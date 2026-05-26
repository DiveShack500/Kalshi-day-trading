#----------------------Betting credentials-------------------------------#
KALSHI_API_KEY_ID  = "YOUR_KALSHI_API_KEY"
KALSHI_PRIVATE_KEY = b"""YOUR_KALSHI_SECRET_KEY"""
#----------------------Betting credentials-------------------------------#
##########################################################################
#!/usr/bin/env python3
"""
BTC 15-Min Kalshi Auto-Predictor
Runs 24/7, fires 9 predictions per 15-min market boundary.
Strategy filters applied automatically — only logs BET when all rules pass.

Rules:
  1. Time window   — 9:45am–6:45pm EST, every day including weekends
  2. Vol floor     — R.Vol >= 15% at prefetch time
  3. Vol ceiling   — Volume <= 500% of 721m avg (extreme spikes skipped)
  4. Vol accel     — skip if 15-min vol > 1.5x 60-min vol at prefetch time
                     (regime shift actively in progress — model calibration stale)
  5. Agreement     — F7, F8, F9 must all predict the same direction
  6. Confidence    — F7 >= 75% AND F8 >= 75% AND F9 >= 75%
  7. Autocorr      — autocorr >= -0.25 (skip deep mean-reverting regimes)
  8. Expected value— EV = model_prob - contract_ask, logged for reference only.
                     No longer a filter — fixed payouts mean edge size is irrelevant.
  9. Prior move    — if prev market moved > PRIOR_SIGMA_THRESHOLD σ, require F9 >= 88%
                     if prev market moved > PRIOR_SIGMA_HARD_BLOCK σ, skip unconditionally
 10. Deceleration  — compare velocity ($/min) F1→F5 (early) vs F5→F9 (late);
                     skip if late velocity < 40% of early velocity in bet direction
                     (momentum fading before the bet is placed)
 11. Reversal guard— if spot reverses > 0.40σ between F8 and F9, skip
                     (momentum already shifting before bet is placed)
 12. Kalshi drift  — track YES ask at F7, F8, F9; skip if total drift > 5¢ against
                     direction OR if both F7→F8 and F8→F9 windows move against us
                     (acceleration). Detects crowd pricing in reversal in real time.

Fire schedule (seconds after boundary):
  F1:40   F2:60   F3:120  F4:180  F5:240  F6:270  ← observation only
  F7:360  F8:480  F9:600                           ← decision fires (5 min remaining)

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
import time as _time
import time
import threading
import datetime
import dataclasses
import concurrent.futures
import uuid
import base64
import requests
import numpy as np
from scipy import stats, optimize
from colorama import Fore, Style, init
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from brti import calculate_brti_fast

init(autoreset=True)

# ── Kalshi credentials ────────────────────────────────────────────────────────



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

FIRE_OFFSETS_SECS = [40, 60, 120, 180, 240, 270, 360, 480, 600]

# F1–F6 are observation-only fires (first 4.5 min of market)
# F7–F9 are the decision fires (~9 min, ~7.5 min, ~5 min remaining)
DECISION_FIRES    = (7, 8, 9)

# ── Bet sizing ────────────────────────────────────────────────────────────────

BET_AMOUNT_DOLLARS = 100.00  # dollars to risk per bet (contracts calculated from price)
BALANCE_FLOOR    = 40.00  # stop all betting if balance drops to or below this
MAX_YES_PRICE    = 97     # cents — refuse to buy YES above this
MIN_YES_PRICE    = 3      # cents — refuse to buy NO above equivalent ceiling

# ── Strategy rules ────────────────────────────────────────────────────────────

TRADE_WINDOW_START  = (9,  45)
TRADE_WINDOW_END    = (18, 45)
VOL_FLOOR           = 0.15
VOL_PCT_MAX         = 5.00

VOL_ACCEL_SHORT_MINS  = 15
VOL_ACCEL_LONG_MINS   = 60
VOL_ACCEL_THRESHOLD   = 1.5

CONF_MIN_F7         = 0.75
CONF_MIN_F8         = 0.75
CONF_MIN_F9         = 0.75
AGREE_FIRES         = {7, 8, 9}
AUTOCORR_MIN        = -0.25

EV_MIN_THRESHOLD    = 0.00

PRIOR_SIGMA_THRESH     = 2.0
CONF_MIN_F9_BOOSTED    = 0.88
PRIOR_SIGMA_HARD_BLOCK = 3.0

DECEL_THRESHOLD    = 0.40
DECEL_MIN_VELOCITY = 5.0

REVERSAL_GUARD_SIGMA = 0.40

KALSHI_DRIFT_THRESHOLD = 0.05


# ── Module-level state (prior market tracking) ────────────────────────────────

_prior_lock              = threading.Lock()
_prior_market_f1_spot: float | None = None
_prior_market_f6_spot: float | None = None
_prior_market_sigma:   float | None = None

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
    return 4 if _time.localtime().tm_isdst else 5

def now_est() -> datetime.datetime:
    return datetime.datetime.utcnow() - datetime.timedelta(hours=utc_offset())

def in_trade_window() -> bool:
    t     = now_est()
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


def compute_vol_acceleration(ohlc: list) -> tuple | None:
    if len(ohlc) < VOL_ACCEL_LONG_MINS + 2:
        return None
    short_candles = ohlc[-VOL_ACCEL_SHORT_MINS:]
    long_candles  = ohlc[-VOL_ACCEL_LONG_MINS:]
    if len(short_candles) < 6 or len(long_candles) < 6:
        return None
    short_vol = compute_vol_from_candles(short_candles)
    long_vol  = compute_vol_from_candles(long_candles)
    if long_vol <= 0:
        return None
    ratio = short_vol / long_vol
    return short_vol, long_vol, ratio


def fetch_vol_and_volume() -> tuple:
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


# ── Kalshi auth ───────────────────────────────────────────────────────────────

def _kalshi_auth_headers(method: str, path: str) -> dict:
    key     = serialization.load_pem_private_key(KALSHI_PRIVATE_KEY, password=None)
    ts      = str(int(_time.time() * 1000))
    msg     = f"{ts}{method.upper()}{path}".encode("utf-8")
    sig_b64 = base64.b64encode(
        key.sign(msg, asym_padding.PSS(
            mgf=asym_padding.MGF1(hashes.SHA256()),
            salt_length=asym_padding.PSS.MAX_LENGTH
        ), hashes.SHA256())
    ).decode()
    return {
        "Content-Type":            "application/json",
        "KALSHI-ACCESS-KEY":       KALSHI_API_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": sig_b64,
    }


# ── Kalshi API ────────────────────────────────────────────────────────────────

def get_balance() -> float:
    """Fetch current account balance in dollars. Raises on failure."""
    path = "/trade-api/v2/portfolio/balance"
    r    = requests.get(f"https://api.elections.kalshi.com{path}",
                        headers=_kalshi_auth_headers("GET", path), timeout=8)
    r.raise_for_status()
    return r.json().get("balance", 0) / 100


def place_order(ticker: str, direction: str,
                yes_ask: float, no_ask: float) -> dict | None:
    """
    Place a limit buy order on Kalshi.
    - Checks live balance first; halts program entirely if at or below BALANCE_FLOOR.
    - direction == "ABOVE" → buy YES at yes_ask price
    - direction == "BELOW" → buy NO  at no_ask price
    - Bets BET_AMOUNT_DOLLARS, rounded down to whole contracts.
    """
    # ── Balance check — halt program if floor reached ──────────────────────
    try:
        balance = get_balance()
        log(f"  💰 Balance: ${balance:,.2f}")
        if balance <= BALANCE_FLOOR:
            log(f"  🛑 BALANCE FLOOR HIT — ${balance:,.2f} at or below "
                f"${BALANCE_FLOOR:.2f} minimum. Shutting down.")
            sys.exit(1)
    except Exception as e:
        log(f"  ⚠ Could not verify balance before order: {e} — skipping order for safety")
        return None

    # ── Pick side and price ────────────────────────────────────────────────
    if direction == "ABOVE":
        side       = "yes"
        price_frac = yes_ask
    else:
        side       = "no"
        price_frac = no_ask

    price_cents = round(price_frac * 100)

    if side == "yes" and price_cents > MAX_YES_PRICE:
        log(f"  ⚠ ORDER SKIPPED — YES price {price_cents}¢ > {MAX_YES_PRICE}¢ ceiling")
        return None
    if side == "no" and price_cents > (100 - MIN_YES_PRICE):
        log(f"  ⚠ ORDER SKIPPED — NO price {price_cents}¢ > {100 - MIN_YES_PRICE}¢ ceiling")
        return None

    # ── Calculate contracts from dollar amount ─────────────────────────────
    contracts = int(BET_AMOUNT_DOLLARS / price_frac)
    if contracts < 1:
        log(f"  ⚠ ORDER SKIPPED — ${BET_AMOUNT_DOLLARS:.2f} not enough to buy "
            f"even 1 contract at {price_cents}¢")
        return None

    actual_payout = float(contracts)
    actual_profit = actual_payout - (contracts * price_frac)

    # ── Place the order ────────────────────────────────────────────────────
    path    = "/trade-api/v2/portfolio/orders"
    payload = {
        "ticker":          ticker,
        "client_order_id": str(uuid.uuid4()),
        "action":          "buy",
        "side":            side,
        "type":            "limit",
        "count":           contracts,
        "yes_price":       price_cents if side == "yes" else (100 - price_cents),
    }

    try:
        r = requests.post(f"https://api.elections.kalshi.com{path}",
                          headers=_kalshi_auth_headers("POST", path),
                          json=payload, timeout=8)
        if r.status_code == 201:
            order = r.json().get("order", {})
            log(f"  ✅ ORDER PLACED — {side.upper()} x{contracts} @ {price_cents}¢ "
                f"| ${actual_payout:.2f} payout (+${actual_profit:.2f} profit) "
                f"| order_id: {order.get('order_id', '?')} "
                f"| status: {order.get('status', '?')}")
            return order
        else:
            log(f"  ❌ ORDER FAILED — HTTP {r.status_code}: {r.text[:200]}")
            return None
    except Exception as e:
        log(f"  ❌ ORDER EXCEPTION — {type(e).__name__}: {e}")
        return None


def find_series_ticker() -> str:
    r = requests.get(f"https://api.elections.kalshi.com/trade-api/v2/series/",
                     params={"limit": 200},
                     headers={"accept": "application/json"}, timeout=8)
    r.raise_for_status()
    for s in r.json().get("series") or []:
        if "btc15m" in (s.get("ticker") or "").lower():
            return s["ticker"]
    raise ValueError("btc15m series not found")

def get_fresh_market(series_ticker: str):
    for attempt in range(MARKET_MAX_TRIES):
        try:
            r = requests.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/",
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
    try:
        r = requests.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}",
                         headers={"accept": "application/json"}, timeout=5)
        if r.status_code != 200:
            log(f"  ⚠ Kalshi price fetch HTTP {r.status_code}: {r.text[:120]}")
            return None
        m = r.json().get("market", {})

        def to_frac(v):
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        yes_bid = to_frac(m.get("yes_bid_dollars"))
        yes_ask = to_frac(m.get("yes_ask_dollars"))
        no_bid  = to_frac(m.get("no_bid_dollars"))
        no_ask  = to_frac(m.get("no_ask_dollars"))

        if yes_bid is not None and yes_ask is not None:
            return {"yes_bid": yes_bid, "yes_ask": yes_ask,
                    "no_bid":  no_bid,  "no_ask":  no_ask}

        price_keys = {k: v for k, v in m.items()
                      if any(x in k.lower() for x in ("bid", "ask", "price", "yes", "no"))}
        log(f"  ⚠ Kalshi price fields null. Price-related keys in response: {price_keys}")
        return None
    except Exception as e:
        log(f"  ⚠ Kalshi price fetch exception: {type(e).__name__}: {e}")
        return None

def get_market_result(ticker: str) -> str | None:
    r = requests.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}",
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

    try:
        result["autocorr"] = compute_autocorr(ohlc, lags=20)
    except Exception:
        result["autocorr"] = None

    try:
        result["vol_accel"] = compute_vol_acceleration(ohlc)
    except Exception:
        result["vol_accel"] = None

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
    vol_accel = prefetched.get("vol_accel")

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
        reason = f"{est.strftime('%I:%M%p')} EST, window 9:45am–6:45pm"
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

    # ── Rule 4: Vol acceleration ───────────────────────────────────────────
    if vol_accel is not None:
        short_vol, long_vol, accel_ratio = vol_accel
        accel_str = (f"vol {VOL_ACCEL_SHORT_MINS}m:{short_vol*100:.1f}% "
                     f"vs {VOL_ACCEL_LONG_MINS}m:{long_vol*100:.1f}% "
                     f"(ratio:{accel_ratio:.2f}x)")
        if accel_ratio > VOL_ACCEL_THRESHOLD:
            log(f"  STRATEGY SKIP — Vol acceleration: {accel_str} "
                f"> {VOL_ACCEL_THRESHOLD:.1f}x threshold (regime shift in progress)")
            sleep_until(boundary + datetime.timedelta(seconds=FIRE_OFFSETS_SECS[-1]))
            return
        log(f"  ✓ Vol accel OK: {accel_str}")
    else:
        log(f"  ⚠ Vol accel unavailable (insufficient candle history) — proceeding")

    # ── Fire all 9 predictions ─────────────────────────────────────────────
    ticker           = None
    strike           = None
    close_time       = None
    fire_results     = {}
    kalshi_snapshots = {}

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
        if fire_num in (7, 8, 9) and ticker is not None:
            snap = fetch_kalshi_prices(ticker)
            if snap:
                kalshi_snapshots[fire_num] = snap

    # ── Rules 5 & 6: Agreement and confidence ─────────────────────────────
    f7 = fire_results.get(7)
    f8 = fire_results.get(8)
    f9 = fire_results.get(9)

    if f7 is None or f8 is None or f9 is None:
        log("  STRATEGY SKIP — missing F7/F8/F9 data")
        return

    dir7, pa7, pb7, spot7, sigma7 = f7
    dir8, pa8, pb8, spot8, sigma8 = f8  # noqa
    dir9, pa9, pb9, spot9, sigma9 = f9  # noqa

    conf7 = max(pa7, pb7)
    conf8 = max(pa8, pb8)
    conf9 = max(pa9, pb9)

    if not (dir7 == dir8 == dir9):
        log(f"  STRATEGY SKIP — F7/F8/F9 disagree (F7:{dir7} F8:{dir8} F9:{dir9})")
        _update_prior(fire_results)
        return

    if conf7 < CONF_MIN_F7 or conf8 < CONF_MIN_F8 or conf9 < CONF_MIN_F9:
        log(f"  STRATEGY SKIP — confidence too low "
            f"(F7:{conf7*100:.1f}% need ≥{CONF_MIN_F7*100:.0f}%, "
            f"F8:{conf8*100:.1f}% need ≥{CONF_MIN_F8*100:.0f}%, "
            f"F9:{conf9*100:.1f}% need ≥{CONF_MIN_F9*100:.0f}%)")
        _update_prior(fire_results)
        return

    direction  = dir9
    model_prob = pa9 if direction == "ABOVE" else pb9

    # ── Rule 7: Autocorrelation regime ────────────────────────────────────
    if autocorr is not None and autocorr < AUTOCORR_MIN:
        log(f"  STRATEGY SKIP — autocorr={autocorr:+.2f} < {AUTOCORR_MIN:+.2f} "
            f"(deep mean-reverting regime — momentum signal unreliable)")
        _update_prior(fire_results)
        return

    # ── Rule 8: Expected value (informational only) ────────────────────────
    kalshi_prices = kalshi_snapshots.get(9) or fetch_kalshi_prices(ticker)
    ev = None
    if kalshi_prices:
        contract_ask = (kalshi_prices["yes_ask"] if direction == "ABOVE"
                        else kalshi_prices["no_ask"])
        if contract_ask and contract_ask > 0:
            ev = model_prob - contract_ask
            regime_str = (f"autocorr={autocorr:+.2f}" if autocorr is not None else "autocorr=n/a")
            log(f"  📊 Kalshi: YES {kalshi_prices['yes_bid']*100:.0f}¢/{kalshi_prices['yes_ask']*100:.0f}¢  "
                f"NO {kalshi_prices['no_bid']*100:.0f}¢/{kalshi_prices['no_ask']*100:.0f}¢  "
                f"| EV:{ev*100:+.1f}¢  model:{model_prob*100:.1f}%  ask:{contract_ask*100:.0f}¢  "
                f"| {regime_str}")
    else:
        regime_str = (f"autocorr={autocorr:+.2f}" if autocorr is not None else "autocorr=n/a")
        log(f"  ⚠ Kalshi prices unavailable  | {regime_str}")

    # ── Rule 9: Prior market magnitude ────────────────────────────────────
    prev_f1, prev_f9, prev_sigma = get_prior_market()
    if prev_f1 and prev_f9 and prev_sigma and prev_sigma > 0:
        prev_move_sigma = abs(prev_f9 - prev_f1) / prev_sigma

        if prev_move_sigma >= PRIOR_SIGMA_HARD_BLOCK:
            log(f"  STRATEGY SKIP — prior market moved {prev_move_sigma:.1f}σ "
                f"≥ {PRIOR_SIGMA_HARD_BLOCK:.0f}σ hard block "
                f"(regime disruption — skip unconditionally)")
            _update_prior(fire_results)
            return

        if prev_move_sigma >= PRIOR_SIGMA_THRESH:
            if conf9 < CONF_MIN_F9_BOOSTED:
                log(f"  STRATEGY SKIP — prior market moved {prev_move_sigma:.1f}σ "
                    f"(reversal risk), need F9 ≥ {CONF_MIN_F9_BOOSTED*100:.0f}% "
                    f"(have {conf9*100:.1f}%)")
                _update_prior(fire_results)
                return
            log(f"  ⚠ Prior move {prev_move_sigma:.1f}σ — raised F9 threshold, passed at {conf9*100:.1f}%")

    # ── Rule 10: Deceleration filter ──────────────────────────────────────
    f1_res = fire_results.get(1)
    f5_res = fire_results.get(5)
    if f1_res and f5_res:
        spot1      = f1_res[3]
        spot5      = f5_res[3]
        time_early = (FIRE_OFFSETS_SECS[4] - FIRE_OFFSETS_SECS[0]) / 60
        time_late  = (FIRE_OFFSETS_SECS[8] - FIRE_OFFSETS_SECS[4]) / 60
        early_vel  = (spot5 - spot1) / time_early
        late_vel   = (spot9 - spot5) / time_late
        sign       = -1.0 if direction == "BELOW" else 1.0
        early_move = early_vel * sign
        late_move  = late_vel  * sign
        if early_move > DECEL_MIN_VELOCITY:
            decel_ratio = late_move / early_move
            if decel_ratio < DECEL_THRESHOLD:
                log(f"  STRATEGY SKIP — Deceleration: early {early_move:+.1f}$/min → "
                    f"late {late_move:+.1f}$/min "
                    f"({decel_ratio*100:.0f}% of early, need ≥{DECEL_THRESHOLD*100:.0f}%)")
                _update_prior(fire_results)
                return
            log(f"  ✓ Momentum OK: early {early_move:+.1f}$/min → late {late_move:+.1f}$/min "
                f"({decel_ratio*100:.0f}% of early)")

    # ── Rule 11: Reversal guard ────────────────────────────────────────────
    if direction == "ABOVE":
        pullback_sigma = (spot8 - spot9) / sigma9
    else:
        pullback_sigma = (spot9 - spot8) / sigma9

    if pullback_sigma > REVERSAL_GUARD_SIGMA:
        log(f"  STRATEGY SKIP — Reversal Guard: F8→F9 pullback={pullback_sigma:.2f}σ "
            f"> {REVERSAL_GUARD_SIGMA:.2f}σ threshold "
            f"(spot moved ${abs(spot9-spot8):,.0f} against {direction} between F8 and F9)")
        _update_prior(fire_results)
        return

    # ── Rule 12: Kalshi order book drift ──────────────────────────────────
    snap7 = kalshi_snapshots.get(7)
    snap8 = kalshi_snapshots.get(8)
    snap9 = kalshi_snapshots.get(9)

    if snap7 and snap8 and snap9:
        ya7 = snap7["yes_ask"]
        ya8 = snap8["yes_ask"]
        ya9 = snap9["yes_ask"]
        drift_78  = ya8 - ya7
        drift_89  = ya9 - ya8
        drift_tot = ya9 - ya7

        drift_str = (f"YES ask {ya7*100:.0f}¢ → {ya8*100:.0f}¢ → {ya9*100:.0f}¢ "
                     f"(F7→F8:{drift_78*100:+.0f}¢  F8→F9:{drift_89*100:+.0f}¢  "
                     f"total:{drift_tot*100:+.0f}¢)")

        if direction == "ABOVE":
            adv_78  = drift_78  < -KALSHI_DRIFT_THRESHOLD
            adv_89  = drift_89  < -KALSHI_DRIFT_THRESHOLD
            adv_tot = drift_tot < -KALSHI_DRIFT_THRESHOLD
            crowd_word = "selling YES"
        else:
            adv_78  = drift_78  > KALSHI_DRIFT_THRESHOLD
            adv_89  = drift_89  > KALSHI_DRIFT_THRESHOLD
            adv_tot = drift_tot > KALSHI_DRIFT_THRESHOLD
            crowd_word = "buying YES"

        accelerating = adv_78 and adv_89

        if accelerating:
            log(f"  STRATEGY SKIP — Kalshi drift accelerating against {direction}: "
                f"{drift_str} ({crowd_word} in both windows)")
            _update_prior(fire_results)
            return
        elif adv_tot:
            log(f"  STRATEGY SKIP — Kalshi drift against {direction}: "
                f"{drift_str} (total {drift_tot*100:+.0f}¢ exceeds {KALSHI_DRIFT_THRESHOLD*100:.0f}¢ threshold)")
            _update_prior(fire_results)
            return
        else:
            accel_note = " ⚡ accelerating with us" if (not adv_78 and not adv_89 and
                         abs(drift_78) > 0.01 and abs(drift_89) > 0.01) else ""
            log(f"  📈 Kalshi drift OK: {drift_str}{accel_note}")

    elif snap7 and snap9:
        ya7 = snap7["yes_ask"]
        ya9 = snap9["yes_ask"]
        drift_tot = ya9 - ya7
        drift_str = (f"YES ask {ya7*100:.0f}¢ → {ya9*100:.0f}¢ "
                     f"(total:{drift_tot*100:+.0f}¢, F8 snapshot unavailable)")
        adverse = (drift_tot < -KALSHI_DRIFT_THRESHOLD if direction == "ABOVE"
                   else drift_tot > KALSHI_DRIFT_THRESHOLD)
        if adverse:
            log(f"  STRATEGY SKIP — Kalshi drift against {direction}: {drift_str}")
            _update_prior(fire_results)
            return
        else:
            log(f"  📈 Kalshi drift OK: {drift_str}")
    else:
        log(f"  ⚠ Kalshi snapshots unavailable — skipping drift rule")

    # ── All rules passed — place the bet ──────────────────────────────────
    ev_str = f" EV:{ev*100:+.1f}¢" if ev is not None else ""
    log(f"  ★ STRATEGY BET → {direction} "
        f"| F7:{conf7*100:.1f}% F8:{conf8*100:.1f}% F9:{conf9*100:.1f}%"
        f"{ev_str} | R.Vol:{real_vol*100:.1f}%")

    # Use the F9 Kalshi snapshot already captured during the fire loop.
    # Fall back to a fresh fetch only if unavailable.
    prices_for_order = kalshi_snapshots.get(9) or fetch_kalshi_prices(ticker)
    if prices_for_order:
        # ── Max bet calculation (informational) ───────────────────────────
        try:
            current_balance = get_balance()
            price_frac      = prices_for_order["yes_ask"] if direction == "ABOVE" else prices_for_order["no_ask"]
            price_cents     = round(price_frac * 100)
            max_contracts   = int(current_balance / price_frac)
            max_cost        = max_contracts * price_frac
            max_payout      = float(max_contracts)
            max_profit      = max_payout - max_cost
            log(f"  💡 Max bet: ${max_cost:.2f} → ${max_payout:.2f} payout (+${max_profit:.2f} profit)")
        except Exception as e:
            log(f"  ⚠ Could not calculate max bet: {e}")

        place_order(
            ticker    = ticker,
            direction = direction,
            yes_ask   = prices_for_order["yes_ask"],
            no_ask    = prices_for_order["no_ask"],
        )
    else:
        log("  ❌ ORDER SKIPPED — could not fetch live prices for order placement")

    _update_prior(fire_results)


def _update_prior(fire_results: dict):
    f1 = fire_results.get(1)
    f9 = fire_results.get(9)
    if f1 and f9:
        set_prior_market(f1_spot=f1[3], f6_spot=f9[3], sigma=f9[4])


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print(Fore.CYAN + Style.BRIGHT + "\n  ₿  BTC Kalshi Auto-Predictor")
    print(Fore.WHITE + "  Running 24/7 — 9 predictions per 15-min market")
    print(Fore.WHITE + f"  Fire times: {FIRE_OFFSETS_SECS} seconds after boundary")
    print(Fore.WHITE + f"  Decision fires: F7/F8/F9 (~9min, ~7.5min, ~5min remaining)")
    print(Fore.YELLOW + "  Strategy rules active:")
    print(Fore.YELLOW + f"    1. Window:      9:45am–6:45pm EST, 7 days a week")
    print(Fore.YELLOW + f"    2. Vol floor:   R.Vol ≥ {VOL_FLOOR*100:.0f}%")
    print(Fore.YELLOW + f"    3. Vol ceiling: Volume ≤ {VOL_PCT_MAX*100:.0f}% of avg (extreme spike filter)")
    print(Fore.YELLOW + f"    4. Vol accel:   {VOL_ACCEL_SHORT_MINS}m vol ≤ {VOL_ACCEL_THRESHOLD:.1f}x "
                        f"{VOL_ACCEL_LONG_MINS}m vol (regime shift — fires at prefetch)")
    print(Fore.YELLOW + f"    5. Agreement:   F7/F8/F9 same direction")
    print(Fore.YELLOW + f"    6. Confidence:  F7 ≥ {CONF_MIN_F7*100:.0f}% AND F8 ≥ {CONF_MIN_F8*100:.0f}% AND F9 ≥ {CONF_MIN_F9*100:.0f}%")
    print(Fore.YELLOW + f"    7. Autocorr:    autocorr ≥ {AUTOCORR_MIN:+.2f} (regime filter)")
    print(Fore.YELLOW + f"    8. EV:          logged only (no filter) — fixed payout makes edge size irrelevant")
    print(Fore.YELLOW + f"    9. Prior move:  if prev mkt > {PRIOR_SIGMA_HARD_BLOCK:.0f}σ → hard skip | "
                        f"> {PRIOR_SIGMA_THRESH:.0f}σ → F9 ≥ {CONF_MIN_F9_BOOSTED*100:.0f}%")
    print(Fore.YELLOW + f"   10. Decel:       late velocity ≥ {DECEL_THRESHOLD*100:.0f}% of early velocity "
                        f"(min early momentum: {DECEL_MIN_VELOCITY:.0f}$/min)")
    print(Fore.YELLOW + f"   11. Rev. guard:  F8→F9 pullback ≤ {REVERSAL_GUARD_SIGMA:.2f}σ")
    print(Fore.YELLOW + f"   12. Kalshi drift: F7→F8→F9 YES ask, skip if total > {KALSHI_DRIFT_THRESHOLD*100:.0f}¢ adverse or accelerating")
    print(Fore.CYAN  + f"  Betting: ${BET_AMOUNT_DOLLARS:.2f} per bet | halt at ${BALANCE_FLOOR:.0f} balance floor")
    print(Fore.WHITE + f"  Logging to: {LOG_FILE}")
    print(Fore.WHITE + "  Press Ctrl+C to stop\n")

    log("=" * 60)
    log("BTC Kalshi Auto-Predictor started — 9 fires per market, live betting ON")
    log("=" * 60)

    # ── Verify credentials and log opening balance ─────────────────────────
    try:
        opening_balance = get_balance()
        log(f"Account balance at startup: ${opening_balance:,.2f}")
        if opening_balance <= BALANCE_FLOOR:
            log(f"STARTUP ABORT — balance ${opening_balance:,.2f} already at or below "
                f"${BALANCE_FLOOR:.2f} floor. Add funds before running.")
            sys.exit(1)
    except Exception as e:
        log(f"STARTUP ABORT — could not fetch balance: {e}")
        sys.exit(1)

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
            va         = prefetched.get("vol_accel")
            va_str     = (f"vaccel={va[2]:.2f}x" if va is not None else "vaccel=n/a")
            log(f"Pre-fetch complete: spot={spot_str}  "
                f"vol={prefetched.get('real_vol', 0)*100:.1f}%  "
                f"dvol={dvol_str}  [{model_str}  {ac_str}  {va_str}]")

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
        except SystemExit:
            raise  # allow sys.exit(1) from balance floor to propagate
        except Exception as e:
            log(f"Unexpected error: {e} — retrying in 30s")
            time.sleep(30)


if __name__ == "__main__":
    main()
