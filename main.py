#!/usr/bin/env python3
"""
BTC 15-Min Kalshi Auto-Predictor
Runs 24/7, fires 6 predictions per 15-min market boundary.
Strategy filters applied automatically — only logs BET when all rules pass.

Rules:
  1. Time window   — 9:45am–6:45pm EST, weekdays only (Mon–Fri)
  2. Vol floor     — R.Vol >= 15% at prefetch time
  3. Vol ceiling   — Volume <= 500% of 721m avg (extreme spikes skipped)
  4. Agreement     — F4, F5, F6 must all predict the same direction
  5. Confidence    — F4 >= 75% AND F5 >= 75% AND F6 >= 75%
  6. Autocorr      — autocorr >= -0.35 (skip deep mean-reverting regimes)
  7. Expected value— EV = model_prob - contract_ask >= EV_MIN_THRESHOLD
  8. Prior move    — if prev market moved > PRIOR_SIGMA_THRESHOLD σ, require F6 >= 88%

Simulation model (augmented):
  - GARCH(1,1)       time-varying volatility fitted from 1-min candles
  - Student-t        fat-tailed innovations (df fitted from return data)
  - Jump-diffusion   Merton model: Poisson arrivals, log-normal jump sizes

Anti-lag design:
  - Vol + BRTI pre-fetched in parallel 30s BEFORE each market boundary
  - Only the Kalshi strike is fetched AFTER open (it changes each market)
  - Result fetching runs in a background thread — never blocks predictions
  - Total lag at prediction time: ~200ms (just the strike fetch)
"""

import sys
import math
import time
import threading
import datetime
import dataclasses
import concurrent.futures
import requests
import numpy as np
from scipy import stats, optimize
from colorama import Fore, Style, init
from brti import calculate_brti_fast

init(autoreset=True)

# ── Constants ─────────────────────────────────────────────────────────────────

N_SIMS            = 500_000
STEPS_PER_MIN     = 6
MINUTES_PER_YEAR  = 525_600
BASE_URL          = "https://api.elections.kalshi.com/trade-api/v2"
LOG_FILE          = "kalshi_btc_3.txt"
VOL_CAP           = 2.00
FALLBACK_VOL      = 0.80
VOL_LOW_THRESHOLD = 0.20
VOL_MIN_MINS      = 15
PREFETCH_SECS     = 30
SETTLE_WAIT_SECS  = 60
MARKET_RETRY_SECS = 5
MARKET_MAX_TRIES  = 20

FIRE_OFFSETS_SECS = [40, 60, 120, 180, 240, 270]

# ── Strategy rules ────────────────────────────────────────────────────────────

TRADE_WINDOW_START  = (9,  45)
TRADE_WINDOW_END    = (18, 45)
VOL_FLOOR           = 0.15
VOL_PCT_MAX         = 5.00   # Rule 3: skip if volume > 500% of avg (extreme spikes)
CONF_MIN_F4         = 0.75   # Rule 5: F4 >= 75%
CONF_MIN_F5         = 0.75   # Rule 5: F5 >= 75%
CONF_MIN_F6         = 0.75   # Rule 5: F6 >= 75%
AGREE_FIRES         = {4, 5, 6}
AUTOCORR_MIN        = -0.35  # Rule 6: skip if autocorr < -0.35 (deep mean-reverting regime)

# Rule 7: Minimum expected value per dollar wagered
# EV = model_prob - contract_ask_price.  At 0.05 you need e.g. model=80% and
# Kalshi asking 75¢ or less.  Set to 0.0 to log EV without filtering.
EV_MIN_THRESHOLD    = 0.03

# Rule 8: After a large prior-market move, require higher F6 confidence.
# The 11:15 loss was a textbook post-spike whipsaw — prev market dropped 2.3σ,
# model caught the echo bounce and called ABOVE, then it reversed again.
PRIOR_SIGMA_THRESH  = 2.0   # prior move in σ units that triggers the rule
CONF_MIN_F6_BOOSTED = 0.88  # raised F6 requirement when prior move was large


# ── Module-level state (prior market tracking) ────────────────────────────────

_prior_lock              = threading.Lock()
_prior_market_f1_spot: float | None = None   # spot at F1 of previous market
_prior_market_f6_spot: float | None = None   # spot at F6 of previous market
_prior_market_sigma:   float | None = None   # 1σ from previous market's F1

def get_prior_market() -> tuple:
    with _prior_lock:
        return _prior_market_f1_spot, _prior_market_f6_spot, _prior_market_sigma

def set_prior_market(f1_spot: float, f6_spot: float, sigma: float):
    global _prior_market_f1_spot, _prior_market_f6_spot, _prior_market_sigma
    with _prior_lock:
        _prior_market_f1_spot = f1_spot
        _prior_market_f6_spot = f6_spot
        _prior_market_sigma   = sigma


# ── Model parameters dataclass ────────────────────────────────────────────────

@dataclasses.dataclass
class ModelParams:
    real_vol:    float
    garch_omega: float
    garch_alpha: float
    garch_beta:  float
    garch_var0:  float
    t_df:        float
    jump_lambda: float
    jump_mu:     float
    jump_sigma:  float


# ── Logging ───────────────────────────────────────────────────────────────────

log_lock = threading.Lock()

def log(line: str):
    ts        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_line = f"[{ts}] {line}"
    with log_lock:
        print(full_line)
        with open(LOG_FILE, "a") as f:
            f.write(full_line + "\n")


# ── Session / time helpers ────────────────────────────────────────────────────

def utc_offset() -> int:
    import time as _time
    return 4 if _time.localtime().tm_isdst else 5

def now_est() -> datetime.datetime:
    return datetime.datetime.utcnow() - datetime.timedelta(hours=utc_offset())

def in_trade_window() -> bool:
    t = now_est()
    if t.weekday() >= 5:
        return False
    mins  = t.hour * 60 + t.minute
    start = TRADE_WINDOW_START[0] * 60 + TRADE_WINDOW_START[1]
    end   = TRADE_WINDOW_END[0]   * 60 + TRADE_WINDOW_END[1]
    return start <= mins < end

def session_elapsed_mins() -> int:
    t     = now_est()
    mins  = t.hour * 60 + t.minute
    start = TRADE_WINDOW_START[0] * 60 + TRADE_WINDOW_START[1]
    return max(mins - start, VOL_MIN_MINS) if mins >= start else VOL_MIN_MINS

def next_boundary_utc() -> datetime.datetime:
    now    = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    next_b = (now.minute // 15 + 1) * 15
    if next_b >= 60:
        return now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
    return now.replace(minute=next_b, second=0, microsecond=0)

def sleep_until(target: datetime.datetime):
    secs = (target - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    if secs > 0:
        time.sleep(secs)


# ── Vol calculation ───────────────────────────────────────────────────────────

def compute_vol_from_candles(candles: list) -> float:
    if len(candles) < 6:
        return FALLBACK_VOL
    opens  = np.array([float(c[1]) for c in candles])
    highs  = np.array([float(c[2]) for c in candles])
    lows   = np.array([float(c[3]) for c in candles])
    closes = np.array([float(c[4]) for c in candles])
    n      = len(candles)
    highs  = np.clip(highs, None, np.percentile(highs, 99))
    lows   = np.clip(lows,  np.percentile(lows, 1), None)
    overnight  = np.log(opens[1:] / closes[:-1])
    open_close = np.log(closes[1:] / opens[1:])
    rs = (np.log(highs[1:] / closes[1:]) * np.log(highs[1:] / opens[1:]) +
          np.log(lows[1:]  / closes[1:]) * np.log(lows[1:]  / opens[1:]))
    k   = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = max(float(np.var(overnight, ddof=1)) +
              k * float(np.var(open_close, ddof=1)) +
              (1 - k) * float(np.median(rs)), 1e-10)
    return min(math.sqrt(var * MINUTES_PER_YEAR), VOL_CAP)


def compute_autocorr(ohlc: list, lags: int = 20) -> float | None:
    """
    Lag-1 autocorrelation of the most recent `lags` one-minute log returns.
    Negative  → mean-reverting (caution: momentum bets become contrarian signals)
    Positive  → trending       (momentum bets have structural support)
    Returns None if insufficient data.
    """
    if len(ohlc) < lags + 2:
        return None
    closes  = np.array([float(c[4]) for c in ohlc[-(lags + 1):]])
    returns = np.diff(np.log(closes))
    if len(returns) < 4:
        return None
    r_t  = returns[:-1]
    r_t1 = returns[1:]
    if np.std(r_t) == 0 or np.std(r_t1) == 0:
        return None
    return float(np.corrcoef(r_t, r_t1)[0, 1])


def fetch_vol_and_volume() -> tuple:
    """Returns (realized_vol, volume_pct, volume_label, ohlc_candles)"""
    now_ts    = int(datetime.datetime.utcnow().timestamp())
    since_old = now_ts - (1440 * 60)
    since_new = now_ts - (720 * 60)

    def _candles(since):
        r = requests.get("https://api.kraken.com/0/public/OHLC",
                         params={"pair": "XBTUSD", "interval": 1, "since": since}, timeout=8)
        r.raise_for_status()
        return next((v for k, v in r.json().get("result", {}).items() if k != "last"), [])

    b1, b2 = _candles(since_old), _candles(since_new)
    seen, merged = set(), []
    for c in b1 + b2:
        if c[0] not in seen:
            seen.add(c[0]); merged.append(c)
    ohlc = sorted(merged, key=lambda c: c[0]) if merged else []

    vol_mins = session_elapsed_mins()
    real_vol = compute_vol_from_candles(ohlc[-vol_mins:]) if ohlc else FALLBACK_VOL

    vol_pct = None
    vol_lbl = "n/a"
    if len(ohlc) >= 30:
        vols   = [float(c[6]) for c in ohlc]
        avg    = float(np.mean(vols))
        recent = float(np.mean([float(c[6]) for c in ohlc[-5:]]))
        if avg > 0:
            vol_pct = recent / avg
            pct     = vol_pct * 100
            flag    = ("⚠ thin" if vol_pct < VOL_LOW_THRESHOLD else
                       "~ low"  if vol_pct < 0.5 else
                       "✓ ok"   if vol_pct < 1.5 else "↑ high")
            vol_lbl = f"{pct:.0f}% of 721m avg ({flag})"

    return real_vol, vol_pct, vol_lbl, ohlc


def fit_model_params(ohlc: list, real_vol: float) -> ModelParams:
    var0_default = real_vol ** 2 / MINUTES_PER_YEAR
    default = ModelParams(
        real_vol=real_vol, garch_omega=var0_default*0.05,
        garch_alpha=0.10, garch_beta=0.85, garch_var0=var0_default,
        t_df=5.0, jump_lambda=0.003, jump_mu=0.0, jump_sigma=0.015,
    )
    if len(ohlc) < 60:
        return default
    closes  = np.array([float(c[4]) for c in ohlc])
    returns = np.diff(np.log(closes))
    if len(returns) < 30:
        return default

    try:
        df, _, _ = stats.t.fit(returns, floc=0)
        t_df = float(np.clip(df, 2.5, 30.0))
    except Exception:
        t_df = default.t_df

    std = float(np.std(returns))
    if std > 0:
        jump_mask   = np.abs(returns) > 3.0 * std
        n_jumps     = int(np.sum(jump_mask))
        jump_lambda = max(float(n_jumps / len(returns)), 1e-4)
        if n_jumps >= 3:
            jump_sizes = returns[jump_mask]
            jump_mu    = float(np.mean(jump_sizes))
            jump_sigma = float(np.std(jump_sizes))
        else:
            jump_mu, jump_sigma = default.jump_mu, default.jump_sigma
    else:
        jump_lambda, jump_mu, jump_sigma = default.jump_lambda, default.jump_mu, default.jump_sigma

    diffusion_ret = np.where(np.abs(returns) > 3.0 * std, 0.0, returns) if std > 0 else returns
    n             = len(diffusion_ret)
    var_init      = float(np.var(diffusion_ret)) or var0_default

    try:
        def neg_loglik(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
                return 1e15
            sigma2    = np.empty(n)
            sigma2[0] = var_init
            for t in range(1, n):
                sigma2[t] = omega + alpha * diffusion_ret[t-1]**2 + beta * sigma2[t-1]
            if np.any(sigma2 <= 0):
                return 1e15
            return float(0.5 * np.sum(np.log(sigma2) + diffusion_ret**2 / sigma2))

        result = optimize.minimize(
            neg_loglik, x0=[var_init*0.05, 0.10, 0.85], method='L-BFGS-B',
            bounds=[(1e-12, var_init), (0.001, 0.4), (0.5, 0.999)],
        )
        if result.success and result.fun < 1e14:
            omega, alpha, beta = result.x
            sigma2    = np.empty(n)
            sigma2[0] = var_init
            for t in range(1, n):
                sigma2[t] = omega + alpha * diffusion_ret[t-1]**2 + beta * sigma2[t-1]
            garch_var0 = float(sigma2[-1])
        else:
            omega, alpha, beta = default.garch_omega, default.garch_alpha, default.garch_beta
            garch_var0 = var_init
    except Exception:
        omega, alpha, beta = default.garch_omega, default.garch_alpha, default.garch_beta
        garch_var0 = var_init

    return ModelParams(
        real_vol=real_vol, garch_omega=float(omega), garch_alpha=float(alpha),
        garch_beta=float(beta), garch_var0=float(garch_var0), t_df=t_df,
        jump_lambda=jump_lambda, jump_mu=jump_mu, jump_sigma=jump_sigma,
    )


def fetch_deribit_vol() -> float | None:
    try:
        r = requests.get("https://www.deribit.com/api/v2/public/get_index_price",
                         params={"index_name": "dvol_btc"}, timeout=6)
        r.raise_for_status()
        dvol = r.json().get("result", {}).get("index_price")
        return dvol / 100.0 if dvol else None
    except Exception:
        return None


# ── Kalshi API ────────────────────────────────────────────────────────────────

def find_series_ticker() -> str:
    r = requests.get(f"{BASE_URL}/series/", params={"limit": 200},
                     headers={"accept": "application/json"}, timeout=8)
    r.raise_for_status()
    for s in r.json().get("series") or []:
        if "btc15m" in (s.get("ticker") or "").lower():
            return s["ticker"]
    raise ValueError("btc15m series not found")

def get_fresh_market(series_ticker: str):
    """Get freshly opened market (>14 min remaining). Returns ticker, strike, close_time, mins_rem."""
    for attempt in range(MARKET_MAX_TRIES):
        try:
            r = requests.get(f"{BASE_URL}/markets/",
                             params={"series_ticker": series_ticker, "status": "open", "limit": 100},
                             headers={"accept": "application/json"}, timeout=8)
            r.raise_for_status()
            markets = r.json().get("markets", [])
            now     = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
            fresh   = [m for m in markets if m.get("close_time") and
                       (datetime.datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")) - now
                        ).total_seconds() > 840]
            if fresh:
                m          = min(fresh, key=lambda m: m["close_time"])
                close_time = datetime.datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))
                mins_rem   = (close_time - now).total_seconds() / 60
                strike     = m.get("floor_strike")
                if strike:
                    return m["ticker"], float(strike), close_time, mins_rem
        except Exception as e:
            if attempt == MARKET_MAX_TRIES - 1:
                raise
        time.sleep(MARKET_RETRY_SECS)
    raise ValueError("No fresh market found after retries")

def fetch_kalshi_prices(ticker: str) -> dict | None:
    """
    Fetch live YES bid/ask from Kalshi for a given market ticker.
    Returns dict with yes_bid, yes_ask, no_bid, no_ask (all as 0-1 fractions)
    or None on failure.
    """
    try:
        r = requests.get(f"{BASE_URL}/markets/{ticker}",
                         headers={"accept": "application/json"}, timeout=5)
        r.raise_for_status()
        m = r.json().get("market", {})
        # Kalshi prices are in cents (0-100); convert to fractions
        def to_frac(v):
            return float(v) / 100.0 if v is not None else None
        yes_bid = to_frac(m.get("yes_bid"))
        yes_ask = to_frac(m.get("yes_ask"))
        no_bid  = to_frac(m.get("no_bid"))
        no_ask  = to_frac(m.get("no_ask"))
        if yes_bid is not None and yes_ask is not None:
            return {"yes_bid": yes_bid, "yes_ask": yes_ask,
                    "no_bid":  no_bid,  "no_ask":  no_ask}
    except Exception:
        pass
    return None

def get_market_result(ticker: str) -> str | None:
    r = requests.get(f"{BASE_URL}/markets/{ticker}",
                     headers={"accept": "application/json"}, timeout=8)
    r.raise_for_status()
    result = r.json().get("market", {}).get("result", "")
    return result if result in ("yes", "no") else None


# ── Simulation ────────────────────────────────────────────────────────────────

def run_simulation(spot: float, horizon_mins: float,
                   params: ModelParams, vol_scale: float = 1.0) -> np.ndarray:
    total_steps = max(1, round(horizon_mins))
    persistence = params.garch_alpha + params.garch_beta
    uncond_var  = (params.garch_omega / (1.0 - persistence)
                   if persistence < 1.0 else params.garch_var0)
    sigma2_path = np.array([
        uncond_var + (persistence ** k) * (params.garch_var0 - uncond_var)
        for k in range(total_steps)
    ], dtype=np.float64)
    sigma2_path = np.maximum(sigma2_path * vol_scale, 1e-12)
    sigma_path  = np.sqrt(sigma2_path)
    drift_path  = -0.5 * sigma2_path

    half = N_SIMS // 2
    if params.t_df < 29.0:
        raw_half  = stats.t.rvs(df=params.t_df, size=(half, total_steps))
        raw_half /= math.sqrt(params.t_df / (params.t_df - 2.0))
    else:
        raw_half = np.random.standard_normal((half, total_steps))
    raw = np.concatenate([raw_half, -raw_half], axis=0)

    log_returns = drift_path[np.newaxis, :] + sigma_path[np.newaxis, :] * raw

    if params.jump_lambda > 0.0 and params.jump_sigma > 0.0:
        n_jumps      = np.random.poisson(params.jump_lambda, (N_SIMS, total_steps))
        jump_sizes   = np.random.normal(params.jump_mu, params.jump_sigma, (N_SIMS, total_steps))
        jump_returns = np.where(n_jumps > 0, jump_sizes, 0.0)
        jump_comp    = params.jump_lambda * (
            math.exp(params.jump_mu + 0.5 * params.jump_sigma ** 2) - 1.0)
        log_returns += jump_returns - jump_comp

    return np.exp(np.log(spot) + log_returns.sum(axis=1))


# ── Result fetcher (background thread) ───────────────────────────────────────

def fetch_result_async(ticker: str, strike: float, prediction: str,
                       close_time: datetime.datetime, p_above: float, p_below: float):
    settle_time = close_time + datetime.timedelta(seconds=SETTLE_WAIT_SECS)
    sleep_until(settle_time)
    for attempt in range(10):
        try:
            result = get_market_result(ticker)
            if result:
                actual  = "ABOVE" if result == "yes" else "BELOW"
                correct = prediction.split("#")[0] == actual
                log(f"  RESULT [{ticker}] #{prediction.split('#')[1] if '#' in prediction else '?'} "
                    f"Actual: {actual} | "
                    f"{'✓ CORRECT' if correct else '✗ WRONG'} "
                    f"(predicted {prediction.split('#')[0]} at {max(p_above, p_below)*100:.1f}%)")
                return
        except Exception:
            pass
        time.sleep(15)
    log(f"  RESULT [{ticker}] ERROR: could not fetch result")


# ── Pre-fetch worker ──────────────────────────────────────────────────────────

def prefetch_all() -> dict:
    result = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        f_vol  = ex.submit(fetch_vol_and_volume)
        f_brti = ex.submit(calculate_brti_fast, False)
        f_dvol = ex.submit(fetch_deribit_vol)

        ohlc = []
        try:
            real_vol, vol_pct, vol_lbl, ohlc = f_vol.result(timeout=10)
            result["real_vol"] = real_vol
            result["vol_pct"]  = vol_pct
            result["vol_lbl"]  = vol_lbl
        except Exception as e:
            result["real_vol"] = FALLBACK_VOL
            result["vol_lbl"]  = f"fallback ({e})"

        try:
            result["spot"] = f_brti.result(timeout=10)
        except Exception:
            result["spot"] = None

        try:
            result["impl_vol"] = f_dvol.result(timeout=8)
        except Exception:
            result["impl_vol"] = None

    try:
        result["model"] = fit_model_params(ohlc, result.get("real_vol", FALLBACK_VOL))
    except Exception:
        result["model"] = None

    # Autocorrelation regime signal from recent returns
    try:
        result["autocorr"] = compute_autocorr(ohlc, lags=20)
    except Exception:
        result["autocorr"] = None

    return result


# ── Kraken spot fetch ─────────────────────────────────────────────────────────

def fetch_kraken_spot() -> float:
    r = requests.get("https://api.kraken.com/0/public/Ticker",
                     params={"pair": "XBTUSD"}, timeout=5)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken error: {data['error']}")
    return float(next(iter(data["result"].values()))["c"][0])


# ── Single fire ───────────────────────────────────────────────────────────────

def fire_prediction(ticker: str, strike: float, close_time: datetime.datetime,
                    fire_num: int, vol_str: str,
                    params: ModelParams, vol_scale: float) -> tuple | None:
    now      = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    mins_rem = (close_time - now).total_seconds() / 60

    try:
        spot = fetch_kraken_spot()
    except Exception as e:
        log(f"  FIRE #{fire_num} ERROR fetching spot: {e}")
        return None

    display_vol = params.real_vol * math.sqrt(vol_scale)
    sigma_move  = spot * display_vol * math.sqrt(mins_rem / MINUTES_PER_YEAR)
    finals      = run_simulation(spot, mins_rem, params, vol_scale)
    p_above     = float((finals > strike).mean())
    p_below     = 1.0 - p_above
    prediction  = "ABOVE" if p_above >= p_below else "BELOW"

    log(f"  FIRE #{fire_num} [{mins_rem:.1f}min left] Spot:${spot:,.2f} | "
        f"PREDICT {prediction} | P(above):{p_above*100:.1f}% P(below):{p_below*100:.1f}% | "
        f"1σ:±${sigma_move:,.0f}")

    threading.Thread(
        target=fetch_result_async,
        args=(ticker, strike, f"{prediction}#{fire_num}", close_time, p_above, p_below),
        daemon=True
    ).start()

    return prediction, p_above, p_below, spot, sigma_move


# ── One market cycle ──────────────────────────────────────────────────────────

def run_market_cycle(series_ticker: str, prefetched: dict, boundary: datetime.datetime):
    real_vol  = prefetched.get("real_vol", FALLBACK_VOL)
    vol_pct   = prefetched.get("vol_pct")
    impl_vol  = prefetched.get("impl_vol")
    vol_lbl   = prefetched.get("vol_lbl", "n/a")
    model     = prefetched.get("model")
    autocorr  = prefetched.get("autocorr")

    if impl_vol:
        final_vol = 0.5 * real_vol + 0.5 * impl_vol
        vol_str   = f"real:{real_vol*100:.1f}% impl:{impl_vol*100:.1f}% blend:{final_vol*100:.1f}%"
    else:
        final_vol = real_vol
        vol_str   = f"real:{real_vol*100:.1f}% (no Deribit)"

    vol_scale = (final_vol / model.real_vol) ** 2 if (model and model.real_vol > 0) else 1.0

    if model is None:
        model = ModelParams(
            real_vol=real_vol, garch_omega=real_vol**2/MINUTES_PER_YEAR*0.05,
            garch_alpha=0.10, garch_beta=0.85, garch_var0=real_vol**2/MINUTES_PER_YEAR,
            t_df=30.0, jump_lambda=0.0, jump_mu=0.0, jump_sigma=0.0,
        )

    # ── Rule 1: Time window ────────────────────────────────────────────────
    if not in_trade_window():
        est    = now_est()
        reason = "weekend" if est.weekday() >= 5 else f"{est.strftime('%I:%M%p')} EST, window 9:45am–6:45pm"
        log(f"  STRATEGY SKIP — outside window ({reason})")
        sleep_until(boundary + datetime.timedelta(seconds=FIRE_OFFSETS_SECS[-1]))
        return

    # ── Rule 2: Vol floor ──────────────────────────────────────────────────
    if real_vol < VOL_FLOOR:
        log(f"  STRATEGY SKIP — R.Vol {real_vol*100:.1f}% < {VOL_FLOOR*100:.0f}% floor")
        sleep_until(boundary + datetime.timedelta(seconds=FIRE_OFFSETS_SECS[-1]))
        return

    # ── Rule 3: Vol ceiling ────────────────────────────────────────────────
    if vol_pct is not None and vol_pct > VOL_PCT_MAX:
        log(f"  STRATEGY SKIP — Volume {vol_pct*100:.0f}% > {VOL_PCT_MAX*100:.0f}% ceiling "
            f"(extreme spike — unreliable market conditions)")
        sleep_until(boundary + datetime.timedelta(seconds=FIRE_OFFSETS_SECS[-1]))
        return

    # ── Fire all 6 predictions ─────────────────────────────────────────────
    ticker       = None
    strike       = None
    close_time   = None
    fire_results = {}   # fire_num -> (prediction, p_above, p_below, spot, sigma_move)

    for fire_num, offset in enumerate(FIRE_OFFSETS_SECS, start=1):
        sleep_until(boundary + datetime.timedelta(seconds=offset))
        if ticker is None:
            try:
                ticker, strike, close_time, mins_rem = get_fresh_market(series_ticker)
                log(f"MARKET  {ticker} | Strike:${strike:,.2f} | Vol:{vol_str} | Volume:{vol_lbl}")
            except Exception as e:
                log(f"ERROR getting market: {e}")
                return
        result = fire_prediction(ticker, strike, close_time, fire_num, vol_str, model, vol_scale)
        if result is not None:
            fire_results[fire_num] = result

    # ── Rules 4 & 5: Agreement and confidence ─────────────────────────────
    f4 = fire_results.get(4)
    f5 = fire_results.get(5)
    f6 = fire_results.get(6)

    if f4 is None or f5 is None or f6 is None:
        log("  STRATEGY SKIP — missing F4/F5/F6 data")
        return

    dir4, pa4, pb4, spot4, sigma4 = f4
    dir5, pa5, pb5, spot5, sigma5 = f5
    dir6, pa6, pb6, spot6, sigma6 = f6

    conf4 = max(pa4, pb4)
    conf5 = max(pa5, pb5)
    conf6 = max(pa6, pb6)

    if not (dir4 == dir5 == dir6):
        log(f"  STRATEGY SKIP — F4/F5/F6 disagree (F4:{dir4} F5:{dir5} F6:{dir6})")
        _update_prior(fire_results)
        return

    if conf4 < CONF_MIN_F4 or conf5 < CONF_MIN_F5 or conf6 < CONF_MIN_F6:
        log(f"  STRATEGY SKIP — confidence too low "
            f"(F4:{conf4*100:.1f}% need ≥{CONF_MIN_F4*100:.0f}%, "
            f"F5:{conf5*100:.1f}% need ≥{CONF_MIN_F5*100:.0f}%, "
            f"F6:{conf6*100:.1f}% need ≥{CONF_MIN_F6*100:.0f}%)")
        _update_prior(fire_results)
        return

    direction  = dir6
    model_prob = pa6 if direction == "ABOVE" else pb6

    # ── Rule 6: Autocorrelation regime ────────────────────────────────────
    if autocorr is not None and autocorr < AUTOCORR_MIN:
        log(f"  STRATEGY SKIP — autocorr={autocorr:+.2f} < {AUTOCORR_MIN:+.2f} "
            f"(deep mean-reverting regime — momentum signal unreliable)")
        _update_prior(fire_results)
        return

    # ── Rule 7: Expected value ─────────────────────────────────────────────
    kalshi_prices = fetch_kalshi_prices(ticker)
    ev = None
    if kalshi_prices:
        # Cost to enter the bet from our perspective
        contract_ask = (kalshi_prices["yes_ask"] if direction == "ABOVE"
                        else kalshi_prices["no_ask"])
        if contract_ask and contract_ask > 0:
            ev = model_prob - contract_ask
            regime_str = (f"autocorr={autocorr:+.2f}" if autocorr is not None else "autocorr=n/a")
            log(f"  📊 Kalshi: YES {kalshi_prices['yes_bid']*100:.0f}¢/{kalshi_prices['yes_ask']*100:.0f}¢  "
                f"NO {kalshi_prices['no_bid']*100:.0f}¢/{kalshi_prices['no_ask']*100:.0f}¢  "
                f"| EV:{ev*100:+.1f}¢  model:{model_prob*100:.1f}%  ask:{contract_ask*100:.0f}¢  "
                f"| {regime_str}")
            if ev < EV_MIN_THRESHOLD:
                log(f"  STRATEGY SKIP — EV {ev*100:+.1f}¢ < {EV_MIN_THRESHOLD*100:.0f}¢ threshold")
                _update_prior(fire_results)
                return
    else:
        # Log regime even if Kalshi price unavailable; don't skip on EV
        regime_str = (f"autocorr={autocorr:+.2f}" if autocorr is not None else "autocorr=n/a")
        log(f"  ⚠ Kalshi prices unavailable — skipping EV rule  | {regime_str}")

    # ── Rule 8: Prior market magnitude ────────────────────────────────────
    prev_f1, prev_f6, prev_sigma = get_prior_market()
    if prev_f1 and prev_f6 and prev_sigma and prev_sigma > 0:
        prev_move_sigma = abs(prev_f6 - prev_f1) / prev_sigma
        if prev_move_sigma >= PRIOR_SIGMA_THRESH:
            if conf6 < CONF_MIN_F6_BOOSTED:
                log(f"  STRATEGY SKIP — prior market moved {prev_move_sigma:.1f}σ "
                    f"(reversal risk), need F6 ≥ {CONF_MIN_F6_BOOSTED*100:.0f}% "
                    f"(have {conf6*100:.1f}%)")
                _update_prior(fire_results)
                return
            log(f"  ⚠ Prior move {prev_move_sigma:.1f}σ — raised F6 threshold, passed at {conf6*100:.1f}%")

    # ── All rules passed — log the bet ────────────────────────────────────
    ev_str = f" EV:{ev*100:+.1f}¢" if ev is not None else ""
    log(f"  ★ STRATEGY BET → {direction} "
        f"| F4:{max(pa4,pb4)*100:.1f}% F5:{conf5*100:.1f}% F6:{conf6*100:.1f}%"
        f"{ev_str} | R.Vol:{real_vol*100:.1f}%")

    _update_prior(fire_results)


def _update_prior(fire_results: dict):
    """Store this market's F1/F6 spot and sigma for Rule 6 next market."""
    f1 = fire_results.get(1)
    f6 = fire_results.get(6)
    if f1 and f6:
        set_prior_market(f1_spot=f1[3], f6_spot=f6[3], sigma=f6[4])


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print(Fore.CYAN + Style.BRIGHT + "\n  ₿  BTC Kalshi Auto-Predictor")
    print(Fore.WHITE + "  Running 24/7 — 6 predictions per 15-min market")
    print(Fore.WHITE + f"  Fire times: {FIRE_OFFSETS_SECS} seconds after boundary")
    print(Fore.YELLOW + "  Strategy rules active:")
    print(Fore.YELLOW + f"    1. Window:      Mon–Fri 9:45am–6:45pm EST")
    print(Fore.YELLOW + f"    2. Vol floor:   R.Vol ≥ {VOL_FLOOR*100:.0f}%")
    print(Fore.YELLOW + f"    3. Vol ceiling: Volume ≤ {VOL_PCT_MAX*100:.0f}% of avg (extreme spike filter)")
    print(Fore.YELLOW + f"    4. Agreement:   F4/F5/F6 same direction")
    print(Fore.YELLOW + f"    5. Confidence:  F4 ≥ {CONF_MIN_F4*100:.0f}% AND F5 ≥ {CONF_MIN_F5*100:.0f}% AND F6 ≥ {CONF_MIN_F6*100:.0f}%")
    print(Fore.YELLOW + f"    6. Autocorr:    autocorr ≥ {AUTOCORR_MIN:+.2f} (regime filter)")
    print(Fore.YELLOW + f"    7. EV:          model_prob − ask ≥ {EV_MIN_THRESHOLD*100:.0f}¢")
    print(Fore.YELLOW + f"    8. Prior move:  if prev mkt > {PRIOR_SIGMA_THRESH:.0f}σ → F6 ≥ {CONF_MIN_F6_BOOSTED*100:.0f}%")
    print(Fore.WHITE + f"  Logging to: {LOG_FILE}")
    print(Fore.WHITE + "  Press Ctrl+C to stop\n")

    log("=" * 60)
    log("BTC Kalshi Auto-Predictor started — 6 fires per market")
    log("=" * 60)

    series_ticker = None

    while True:
        try:
            if series_ticker is None:
                try:
                    series_ticker = find_series_ticker()
                    log(f"Series ticker: {series_ticker}")
                except Exception as e:
                    log(f"ERROR finding ticker: {e} — retrying in 60s")
                    time.sleep(60)
                    continue

            boundary    = next_boundary_utc()
            prefetch_at = boundary - datetime.timedelta(seconds=PREFETCH_SECS)
            now_utc     = datetime.datetime.now(datetime.timezone.utc)
            wait_secs   = (prefetch_at - now_utc).total_seconds()

            if wait_secs > 0:
                log(f"Next market in {(boundary - now_utc).total_seconds()/60:.1f}min — "
                    f"pre-fetching in {wait_secs:.0f}s")
                time.sleep(wait_secs)

            log("Pre-fetching vol + BRTI...")
            prefetched = prefetch_all()
            dvol_str   = ("n/a" if not prefetched.get("impl_vol")
                          else f"{prefetched['impl_vol']*100:.1f}%")
            spot_str   = (f"${prefetched['spot']:,.2f}" if prefetched.get('spot')
                          else "n/a (BRTI failed)")
            m          = prefetched.get("model")
            model_str  = (f"GARCH(α={m.garch_alpha:.2f},β={m.garch_beta:.2f}) "
                          f"t-df={m.t_df:.1f} λ={m.jump_lambda:.4f}") if m else "fallback GBM"
            ac         = prefetched.get("autocorr")
            ac_str     = f"autocorr={ac:+.2f}" if ac is not None else "autocorr=n/a"
            log(f"Pre-fetch complete: spot={spot_str}  "
                f"vol={prefetched.get('real_vol', 0)*100:.1f}%  "
                f"dvol={dvol_str}  [{model_str}  {ac_str}]")

            if prefetched.get('spot') is None:
                log("WARNING: BRTI spot unavailable — skipping this market")
                sleep_until(boundary + datetime.timedelta(seconds=FIRE_OFFSETS_SECS[-1] + 10))
                continue

            sleep_until(boundary)
            log("─" * 60)
            run_market_cycle(series_ticker, prefetched, boundary)

        except KeyboardInterrupt:
            log("Stopped by user.")
            sys.exit(0)
        except Exception as e:
            log(f"Unexpected error: {e} — retrying in 30s")
            time.sleep(30)


if __name__ == "__main__":
    main()
