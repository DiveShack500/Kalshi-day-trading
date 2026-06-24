#----------------------Betting credentials-------------------------------#
KALSHI_API_KEY_ID  = "YOUR_API_KEY"
KALSHI_PRIVATE_KEY = b"""YOUR_SECRET_KEY"""
#----------------------Betting credentials-------------------------------#
import datetime
import time
import base64
import uuid
import sys
import math
import threading
import requests
import brti_btc
from collections import deque
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL           = "https://api.elections.kalshi.com"
LOG_FILE           = "btc_live.txt"

BUY_MIN_CENTS      = 98
BUY_MAX_CENTS      = 100
BET_DOLLARS        = 25
MAX_OPPOSING_CENTS = 6

WATCH_SECS_WEEKDAY = 900   # start observing 9 minutes before settlement
WATCH_SECS_WEEKEND = 900   # start observing 9 minutes before settlement
ENTRY_START_SECS   = 300   # earliest entry allowed: T-5:00
MARKET_MIN_SECS    = 0
POLL_MS            = 0.1

# ── Startup fresh-market gate ─────────────────────────────────────────────────
# On startup, ignore the currently open 15-minute market and wait until Kalshi
# rotates to a new BTC 15-minute market. A fresh market usually has close to
# 900 seconds left; this minimum avoids entering a market that changed while the
# script was offline or delayed.
STARTUP_FRESH_MIN_SECS = 850
STARTUP_POLL_SECS      = 5

# ── Climb filter ──────────────────────────────────────────────────────────────
CLIMB_LOOKBACK_SECS = 80
CLIMB_MAX_CENTS     = 10
CLIMB_BYPASS_SECS   = 120

# ── Buffer filter ─────────────────────────────────────────────────────────────
BTC_MIN_BUFFER             = 50.00
BTC_RESET_BUFFER           = 45.00  # sustained tracker resets only if buffer drops below this (prevents jitter resets)
BTC_MIN_PROJECTED_BUFFER   = 95.00
BUFFER_SKIP_PROJECTED_SECS = 90
BUFFER_MIN_SUSTAINED_SECS  = 30   # buffer must be continuously ≥$50 for this long before trigger (mid-window only)

# ── Marginal certainty filter ─────────────────────────────────────────────────
# Blocks early 99¢ triggers that do not have enough BTC clearance. This targets
# near-certain market pricing with only marginal distance from the strike.
MARGINAL_CERTAINTY_MIN_CENTS = 99
MARGINAL_CERTAINTY_MIN_SECS  = 120
MARGINAL_CERTAINTY_BUFFER    = 85.00

# ── ROC filter ────────────────────────────────────────────────────────────────
BTC_MAX_ROC_MOVE      = 300.0
BTC_ROC_LOOKBACK_MINS = 5

# ── Recent volatility / explosive-move filter ─────────────────────────────────
# Scans the broader recent lookback for ANY rolling 3-minute window where BTC's
# high-low range exceeded the threshold. This catches short explosive bursts
# without incorrectly treating the entire lookback as one "3m" window.
# Scan the current market watch window from watch_started_at through the trigger time.
# Inside that elapsed period, find the largest move in any rolling 3-minute slice.
RECENT_RANGE_WINDOW_SECS   = 180   # each tested window is 3 minutes
RECENT_RANGE_MIN_MOVE      = 100.0 # only care once any 3m window moved this much
RECENT_RANGE_BUFFER_MULT   = 1.0   # require buffer >= 1.00 × max rolling 3m range

# ── Mid-window bounce filter ──────────────────────────────────────────────────
BOUNCE_MIN_EXCURSION   = 40.00
BOUNCE_MIN_T_REMAINING = 100

# ── Contested-strike filter ───────────────────────────────────────────────────
# Skip a market once BTC has made two meaningful crossings through the strike.
# A crossing only counts after BTC reaches at least this many dollars beyond
# the strike on the opposite side, so tiny $1–$6 wiggles around the strike are
# ignored. Example: +$20 → -$8 → +$9 counts as two meaningful crossings.
CONTESTED_CROSS_EXCURSION = 7.00
CONTESTED_CROSS_MAX_COUNT = 2

# ── Shallow book filter ───────────────────────────────────────────────────────
SHALLOW_BOOK_MAX_CONTRACTS = 25_000
SHALLOW_BOOK_MIN_RATIO     = 16

# ── Post-climb-skip veto filter ───────────────────────────────────────────────
# After a NO climb-skip, record the BTC spot price at that moment. In the next
# market, the "retracement risk" is how far BTC has fallen since the skip fired.
# If that risk is small (move stalled), allow the trade. If large, require the
# current gap below the new strike to exceed SAFETY_RATIO * retracement_risk.
# Stall threshold scaled from ETH $2.00 × 64 = $128.
POST_CLIMB_STALL_THRESHOLD = 128.00  # retracement risk <= this → move stalled → allow
POST_CLIMB_SAFETY_RATIO    = 0.50    # gap must be > this fraction of retracement risk


# ── BTC spot tracker ──────────────────────────────────────────────────────────

class BtcSpotTracker:
    COINBASE_URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"

    def __init__(self, history_secs: int = 1800):
        self.history  = deque(maxlen=history_secs * 2)
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log("BTC spot tracker started (BRTI)")

    def stop(self):
        self._running = False

    def _fetch_price(self) -> "float | None":
        for method in ("get_live_brti", "get_brti", "calculate_brti_fast"):
            try:
                fn = getattr(brti_btc, method, None)
                if fn:
                    price = fn() if method != "calculate_brti_fast" else fn(verbose=False)
                    if price is not None and float(price) > 10_000:
                        return float(price)
            except Exception:
                pass
        try:
            r = requests.get(self.COINBASE_URL, timeout=5)
            if r.status_code == 200:
                price = float(r.json()["data"]["amount"])
                if price > 10_000:
                    return price
        except Exception:
            pass
        return None

    def _loop(self):
        while self._running:
            try:
                price = self._fetch_price()
                if price is not None:
                    with self._lock:
                        self.history.append((time.time(), float(price)))
            except Exception:
                pass
            time.sleep(2)

    def latest(self) -> "float | None":
        with self._lock:
            return self.history[-1][1] if self.history else None

    def momentum(self, lookback_secs: int = 60) -> "float | None":
        with self._lock:
            if len(self.history) < 2:
                return None
            cutoff = time.time() - lookback_secs
            oldest = next(((ts, px) for ts, px in self.history if ts >= cutoff), None)
            if oldest is None:
                return None
            newest  = self.history[-1]
            elapsed = newest[0] - oldest[0]
            return (newest[1] - oldest[1]) / elapsed if elapsed >= 1 else None

    def distance_from_strike(self, strike: float) -> "float | None":
        price = self.latest()
        return (price - strike) if (price is not None and strike is not None) else None

    def change_over_mins(self, mins: int = 5) -> "tuple[float | None, float | None]":
        with self._lock:
            if len(self.history) < 2:
                return None, None
            cutoff = time.time() - (mins * 60)
            oldest = next((px for ts, px in self.history if ts >= cutoff), None)
            if oldest is None:
                return None, None
            net = self.history[-1][1] - oldest
            return abs(net), net

    def range_over_secs(self, lookback_secs: int) -> "tuple[float | None, float | None, float | None]":
        """Return high-low range, low, high over the requested lookback.

        Unlike change_over_mins(), this catches V-shaped moves where BTC starts
        and ends near the same price but travels a large distance in between.
        """
        with self._lock:
            if len(self.history) < 2:
                return None, None, None
            cutoff = time.time() - lookback_secs
            prices = [px for ts, px in self.history if ts >= cutoff]
            if len(prices) < 2:
                return None, None, None
            low = min(prices)
            high = max(prices)
            return high - low, low, high

    def max_range_window(self, lookback_secs: int,
                         window_secs: int) -> "tuple[float | None, float | None, float | None]":
        """Return the largest high-low move found in any rolling window.

        Example: lookback_secs=900 and window_secs=180 scans the last 15 minutes
        and returns the largest move that occurred inside any 3-minute slice.
        """
        with self._lock:
            if len(self.history) < 2:
                return None, None, None

            cutoff = time.time() - lookback_secs
            points = [(ts, px) for ts, px in self.history if ts >= cutoff]
            if len(points) < 2:
                return None, None, None

            best_range = 0.0
            best_low = None
            best_high = None

            left = 0
            window = deque()

            for ts, px in points:
                window.append((ts, px))
                while window and window[0][0] < ts - window_secs:
                    window.popleft()

                if len(window) < 2:
                    continue

                prices = [p for _, p in window]
                low = min(prices)
                high = max(prices)
                current_range = high - low

                if current_range > best_range:
                    best_range = current_range
                    best_low = low
                    best_high = high

            if best_low is None or best_high is None:
                return None, None, None
            return best_range, best_low, best_high


# ── Mid-window bounce tracker ─────────────────────────────────────────────────

class BounceTrackerV2:
    def __init__(self):
        self.reset(strike=None)

    def reset(self, strike):
        self._strike            = strike
        self._started_below     = None
        self._max_cross_above   = 0.0
        self._t_remaining_cross = 0
        self._max_dip_below     = 0.0
        self._t_remaining_dip   = 0

    def update(self, btc_price: float, t_remaining: float):
        if self._strike is None or btc_price is None:
            return
        above = btc_price > self._strike
        if self._started_below is None:
            self._started_below = not above
        if self._started_below:
            if above:
                exc = btc_price - self._strike
                if exc > self._max_cross_above:
                    self._max_cross_above   = exc
                    self._t_remaining_cross = t_remaining
        else:
            if not above:
                dip = self._strike - btc_price
                if dip > self._max_dip_below:
                    self._max_dip_below   = dip
                    self._t_remaining_dip = t_remaining

    def should_skip(self, side: str) -> "tuple[bool, str]":
        if side == "no":
            if (self._started_below
                    and self._max_cross_above > BOUNCE_MIN_EXCURSION
                    and self._t_remaining_cross > BOUNCE_MIN_T_REMAINING):
                return True, (f"price bounced above strike mid-window "
                              f"(+${self._max_cross_above:,.0f} peak at T-{self._t_remaining_cross:.0f}s)")
        elif side == "yes":
            if (self._started_below is False
                    and self._max_dip_below > BOUNCE_MIN_EXCURSION
                    and self._t_remaining_dip > BOUNCE_MIN_T_REMAINING):
                return True, (f"price dipped below strike mid-window "
                              f"(-${self._max_dip_below:,.0f} peak at T-{self._t_remaining_dip:.0f}s)")
        return False, ""


class ContestedStrikeTracker:
    """Tracks meaningful strike crossings inside the current market.

    A price is considered meaningfully above the strike only at +$threshold or
    higher, and meaningfully below only at -$threshold or lower. Values inside
    that neutral band do not change state. This prevents tiny flickers around
    the strike from counting as crosses.
    """

    def __init__(self, excursion: float = CONTESTED_CROSS_EXCURSION):
        self._excursion = float(excursion)
        self.reset(strike=None)

    def reset(self, strike):
        self._strike = strike
        self._zone = None  # "above" or "below" after leaving neutral band
        self._cross_count = 0
        self._max_above = 0.0
        self._max_below = 0.0
        self._last_cross_t_remaining = None

    def update(self, btc_price: float, t_remaining: float):
        if self._strike is None or btc_price is None:
            return

        dist = btc_price - self._strike
        if dist >= self._excursion:
            new_zone = "above"
            self._max_above = max(self._max_above, dist)
        elif dist <= -self._excursion:
            new_zone = "below"
            self._max_below = max(self._max_below, abs(dist))
        else:
            return

        if self._zone is None:
            self._zone = new_zone
            return

        if new_zone != self._zone:
            self._cross_count += 1
            self._zone = new_zone
            self._last_cross_t_remaining = t_remaining

    def should_skip(self) -> "tuple[bool, str]":
        if self._cross_count < CONTESTED_CROSS_MAX_COUNT:
            return False, ""
        return True, (
            f"contested-strike filter: BTC made {self._cross_count} meaningful "
            f"strike crossings with >=${self._excursion:,.0f} excursions "
            f"(max above +${self._max_above:,.0f}, max below -${self._max_below:,.0f}"
            + (f", last cross at T-{self._last_cross_t_remaining:.0f}s" if self._last_cross_t_remaining is not None else "")
            + ")"
        )


# ── Helpers / logging / auth / API ────────────────────────────────────────────

def get_watch_secs() -> int:
    return WATCH_SECS_WEEKDAY if datetime.datetime.now().weekday() < 5 else WATCH_SECS_WEEKEND

def day_mode() -> str:
    return "weekday" if datetime.datetime.now().weekday() < 5 else "weekend"

def log(msg: str):
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    safe_line = line.encode("utf-8", errors="replace").decode("utf-8")
    print(safe_line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
        f.write(safe_line + "\n")

def _auth(method: str, path: str) -> dict:
    key = serialization.load_pem_private_key(KALSHI_PRIVATE_KEY, password=None)
    ts  = str(int(time.time() * 1000))
    msg = f"{ts}{method.upper()}{path}".encode()
    sig = base64.b64encode(
        key.sign(msg, asym_padding.PSS(
            mgf=asym_padding.MGF1(hashes.SHA256()),
            salt_length=asym_padding.PSS.MAX_LENGTH,
        ), hashes.SHA256())
    ).decode()
    return {"Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": KALSHI_API_KEY_ID,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig}

def api_get(path: str, params: dict = None, auth: bool = False) -> dict:
    headers = _auth("GET", path) if auth else {"accept": "application/json"}
    r = requests.get(f"{BASE_URL}{path}", params=params or {}, headers=headers, timeout=8)
    r.raise_for_status()
    return r.json()

def get_balance() -> float:
    return api_get("/trade-api/v2/portfolio/balance", auth=True).get("balance", 0) / 100

def find_btc_series() -> str:
    data = api_get("/trade-api/v2/series/", params={"limit": 200})
    for s in (data.get("series") or []):
        ticker = (s.get("ticker") or "").lower()
        if "btc15m" in ticker or "kxbtc15m" in ticker:
            return s["ticker"]
    raise ValueError("BTC 15-min series not found")

def _extract_strike(market: dict) -> "float | None":
    import re as _re
    fs = market.get("floor_strike")
    if fs is not None:
        try:
            return float(fs)
        except (TypeError, ValueError):
            pass
    m = _re.search(r'\$?([\d,]+\.?\d*)', market.get("yes_sub_title") or "")
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None

def get_current_btc_market(series_ticker: str) -> "dict | None":
    data = api_get("/trade-api/v2/markets/",
                   params={"series_ticker": series_ticker, "status": "open", "limit": 100})
    now   = datetime.datetime.now(datetime.timezone.utc)
    valid = []
    for m in data.get("markets", []):
        ct = m.get("close_time")
        if not ct:
            continue
        close_dt  = datetime.datetime.fromisoformat(ct.replace("Z", "+00:00"))
        secs_left = (close_dt - now).total_seconds()
        if secs_left >= MARKET_MIN_SECS:
            valid.append((secs_left, m))
    if not valid:
        return None
    valid.sort(key=lambda x: x[0])
    return valid[0][1]

def secs_to_close(market: dict) -> float:
    ct = datetime.datetime.fromisoformat(market["close_time"].replace("Z", "+00:00"))
    return (ct - datetime.datetime.now(datetime.timezone.utc)).total_seconds()

def wait_for_fresh_startup_market(series_ticker: str) -> None:
    """Ignore the market that is already open at startup.

    This prevents the bot from joining a partially elapsed 15-minute BTC market
    when the program is launched mid-candle. Normal trading begins only after a
    different market appears with close to a full 15 minutes remaining.
    """
    startup_market = get_current_btc_market(series_ticker)

    if startup_market is None:
        log("No open BTC market at startup — waiting for first fresh 15-minute market...")
        startup_ticker = None
    else:
        startup_ticker = startup_market.get("ticker")
        startup_secs = secs_to_close(startup_market)
        log(f"Startup market detected: {startup_ticker}  closes in {startup_secs:.0f}s")
        log("Waiting for next fresh 15-minute market before trading...")

    while True:
        time.sleep(STARTUP_POLL_SECS)

        current = get_current_btc_market(series_ticker)
        if current is None:
            continue

        current_ticker = current.get("ticker")
        current_secs = secs_to_close(current)

        if current_ticker == startup_ticker:
            continue

        if current_secs < STARTUP_FRESH_MIN_SECS:
            log(f"Saw new market {current_ticker}, but it only has {current_secs:.0f}s left "
                f"(< {STARTUP_FRESH_MIN_SECS}s). Waiting for a fresher market...")
            startup_ticker = current_ticker
            continue

        log(f"Fresh market detected: {current_ticker}  closes in {current_secs:.0f}s — starting trading loop")
        return

def get_prices(ticker: str) -> "dict | None":
    try:
        m  = api_get(f"/trade-api/v2/markets/{ticker}").get("market", {})
        ya = m.get("yes_ask_dollars")
        na = m.get("no_ask_dollars")
        if ya is None and na is None:
            return None
        return {"yes_ask": float(ya) if ya is not None else None,
                "no_ask":  float(na) if na is not None else None,
                "strike":  _extract_strike(m)}
    except Exception as e:
        log(f"  price fetch error: {e}")
        return None

def get_orderbook_totals(ticker: str) -> "tuple[int | None, int | None]":
    try:
        book = api_get(f"/trade-api/v2/markets/{ticker}/orderbook").get("orderbook_fp", {})
        yes_total = sum(float(qty) for _, qty in book.get("yes_dollars", []))
        no_total  = sum(float(qty) for _, qty in book.get("no_dollars",  []))
        return int(yes_total), int(no_total)
    except Exception as e:
        log(f"  📖 Order book fetch error: {e}")
        return None, None

def log_orderbook_depth(ticker: str, side: str):
    try:
        book = api_get(f"/trade-api/v2/markets/{ticker}/orderbook").get("orderbook_fp", {})
        for s, key in (("yes", "yes_dollars"), ("no", "no_dollars")):
            entries = book.get(key, [])
            if not entries:
                log(f"  📖 Order book ({s.upper()}): empty")
            else:
                total = sum(float(qty) for _, qty in entries)
                top   = entries[-5:][::-1]
                lines = "  ".join(f"{round(float(p)*100)}¢×{round(float(q))}" for p, q in top)
                log(f"  📖 Order book ({s.upper()}): {round(total)} contracts total — {lines}")
    except Exception as e:
        log(f"  📖 Order book fetch error: {e}")

def log_btc_info(tracker: BtcSpotTracker, strike: float, side: str):
    spot = tracker.latest()
    mom  = tracker.momentum(lookback_secs=60)
    if spot is None:
        log("  📊 BTC spot: unavailable")
        return
    dist = (spot - strike) if strike else None
    dist_str = (f"${dist:+,.0f} from strike ({'favors YES' if dist > 0 else 'favors NO'})"
                if dist is not None else "strike unknown")
    if mom is not None:
        toward  = dist is not None and ((dist > 0 and mom < 0) or (dist < 0 and mom > 0))
        mom_str = (f"${mom:+.2f}/s {'↑ rising' if mom > 0 else '↓ falling'} → "
                   f"{'moving TOWARD strike ⚠' if toward else 'moving AWAY from strike ✅'}")
    else:
        mom_str = "unavailable"
    log(f"  📊 BTC ${spot:,.0f}  {dist_str}  momentum {mom_str}")


# ── Filters ───────────────────────────────────────────────────────────────────

def price_climbed_too_fast(side: str, price_history: deque) -> "tuple[bool, str]":
    if not price_history:
        return False, ""
    cutoff = time.time() - CLIMB_LOOKBACK_SECS
    oldest = next(((ts, yc, nc) for ts, yc, nc in price_history if ts >= cutoff), None)
    if oldest is None:
        return False, ""
    _, old_yc, old_nc = oldest
    new_ts, new_yc, new_nc = price_history[-1]
    if side == "yes" and old_yc is not None and new_yc is not None:
        climb, elapsed = new_yc - old_yc, new_ts - oldest[0]
    elif side == "no" and old_nc is not None and new_nc is not None:
        climb, elapsed = new_nc - old_nc, new_ts - oldest[0]
    else:
        return False, ""
    if climb <= CLIMB_MAX_CENTS:
        return False, ""
    return True, f"{side.upper()} climbed {climb}¢ in {elapsed:.0f}s (max {CLIMB_MAX_CENTS}¢/{CLIMB_LOOKBACK_SECS}s)"

def buffer_too_small(side: str, dist: "float | None", mom: "float | None",
                     secs_remaining: float) -> "tuple[bool, str]":
    """Uses pre-snapshotted dist and momentum so the same values are used
    here as everywhere else in the poll loop — no second tracker call."""
    if dist is None:
        return False, ""
    if side == "yes" and dist < BTC_MIN_BUFFER:
        return True, f"BTC buffer too small for YES (${dist:+,.0f}, need +${BTC_MIN_BUFFER:,.0f})"
    if side == "no" and dist > -BTC_MIN_BUFFER:
        return True, f"BTC buffer too small for NO (${dist:+,.0f}, need -${BTC_MIN_BUFFER:,.0f})"
    if secs_remaining <= BUFFER_SKIP_PROJECTED_SECS:
        return False, ""
    if mom is not None:
        moving_toward = (side == "yes" and mom < 0) or (side == "no" and mom > 0)
        if moving_toward:
            erosion   = abs(mom) * secs_remaining
            projected = abs(dist) - erosion
            if projected < BTC_MIN_PROJECTED_BUFFER:
                return True, (f"projected buffer too small for {side.upper()} "
                              f"(${abs(dist):,.0f} - ${erosion:,.0f} erosion = "
                              f"${projected:,.0f} projected, need ${BTC_MIN_PROJECTED_BUFFER:,.0f})")
    return False, ""

def buffer_not_sustained(side: str,
                         buffer_sufficient_since: "float | None",
                         buffer_at_window_open: bool) -> "tuple[bool, str]":
    """Block triggers where the ≥$50 buffer appeared mid-window and has not been
    continuously held for BUFFER_MIN_SUSTAINED_SECS seconds. If the buffer was
    already present on the first poll of the watch window, bypass entirely —
    BTC was already separated before we started watching."""
    if buffer_at_window_open:
        return False, ""
    if buffer_sufficient_since is None:
        return True, f"buffer has not yet reached ${BTC_MIN_BUFFER:,.0f}"
    age = time.time() - buffer_sufficient_since
    if age < BUFFER_MIN_SUSTAINED_SECS:
        return True, (f"{side.upper()} buffer only sustained for {age:.0f}s "
                      f"(need {BUFFER_MIN_SUSTAINED_SECS}s continuously)")
    return False, ""

def marginal_certainty_veto(side: str, price_cents: "int | None",
                             btc_dist: "float | None",
                             secs_remaining: float) -> "tuple[bool, str]":
    if price_cents is None or btc_dist is None:
        return False, ""
    if price_cents < MARGINAL_CERTAINTY_MIN_CENTS:
        return False, ""
    if secs_remaining <= MARGINAL_CERTAINTY_MIN_SECS:
        return False, ""

    buffer = abs(btc_dist)
    if buffer >= MARGINAL_CERTAINTY_BUFFER:
        return False, ""

    return True, (
        f"marginal certainty filter: {side.upper()} {price_cents}¢ at "
        f"T-{secs_remaining:.0f}s with only ${buffer:,.0f} buffer "
        f"(need ${MARGINAL_CERTAINTY_BUFFER:,.0f} for early 99¢ triggers)"
    )

def roc_too_high(tracker: BtcSpotTracker, side: str, strike: float) -> "tuple[bool, str]":
    abs_move, net_move = tracker.change_over_mins(BTC_ROC_LOOKBACK_MINS)
    if abs_move is None or abs_move <= BTC_MAX_ROC_MOVE:
        return False, ""
    moving_toward = (side == "yes" and net_move < 0) or (side == "no" and net_move > 0)
    if moving_toward:
        direction = "falling" if net_move < 0 else "rising"
        return True, (f"BTC moved ${abs_move:,.0f} in {BTC_ROC_LOOKBACK_MINS}min "
                      f"{direction} toward strike (max ${BTC_MAX_ROC_MOVE:,.0f})")
    return False, ""

def recent_range_veto(tracker: BtcSpotTracker, side: str,
                      btc_dist: "float | None",
                      elapsed_watch_secs: float) -> "tuple[bool, str]":
    if btc_dist is None:
        return False, ""

    scan_secs = max(1, int(elapsed_watch_secs))
    recent_range, low, high = tracker.max_range_window(
        scan_secs,
        RECENT_RANGE_WINDOW_SECS,
    )
    if recent_range is None or recent_range < RECENT_RANGE_MIN_MOVE:
        return False, ""

    buffer = abs(btc_dist)
    required_buffer = RECENT_RANGE_BUFFER_MULT * recent_range
    if buffer >= required_buffer:
        return False, ""

    return True, (
        f"recent volatility filter: BTC max {RECENT_RANGE_WINDOW_SECS // 60}m range "
        f"${recent_range:,.0f} (${low:,.0f}–${high:,.0f}) found during current "
        f"watch window ({scan_secs}s elapsed); {side.upper()} buffer ${buffer:,.0f} "
        f"< {RECENT_RANGE_BUFFER_MULT:.2f}× range = ${required_buffer:,.0f}"
    )

def shallow_book(side: str, yes_total: "int | None", no_total: "int | None") -> "tuple[bool, str]":
    if yes_total is None or no_total is None:
        return False, ""
    bet_book = yes_total if side == "yes" else no_total
    opp_book = no_total  if side == "yes" else yes_total
    if opp_book == 0:
        return False, ""
    ratio = bet_book / opp_book
    if bet_book < SHALLOW_BOOK_MAX_CONTRACTS and ratio > SHALLOW_BOOK_MIN_RATIO:
        return True, (f"shallow and lopsided {side.upper()} book "
                      f"({bet_book:,} contracts, {ratio:.0f}x opposing — "
                      f"need >{SHALLOW_BOOK_MAX_CONTRACTS:,} or <{SHALLOW_BOOK_MIN_RATIO}x ratio)")
    return False, ""

def post_climb_skip_veto(side: str, btc_dist: "float | None",
                         btc_at_prior_skip: "float | None",
                         current_btc: "float | None") -> "tuple[bool, str]":
    if btc_at_prior_skip is None or side != "no":
        return False, ""
    if btc_dist is None or current_btc is None:
        return False, ""

    retracement_risk = btc_at_prior_skip - current_btc

    if retracement_risk <= 0:
        return False, ""

    if retracement_risk <= POST_CLIMB_STALL_THRESHOLD:
        return False, ""

    gap_below = abs(btc_dist)
    required  = POST_CLIMB_SAFETY_RATIO * retracement_risk

    if gap_below < required:
        return True, (
            f"post-climb-skip veto: BTC fell ${retracement_risk:,.0f} since prior skip "
            f"(from ${btc_at_prior_skip:,.0f}), gap below strike ${gap_below:,.0f} "
            f"< {POST_CLIMB_SAFETY_RATIO:.0%} of retracement risk ${required:,.0f}"
        )
    return False, ""


# ── Order placement ───────────────────────────────────────────────────────────

def _best_bid_from_orderbook(entries) -> "tuple[float | None, float | None]":
    """Return best bid price and qty from a Kalshi orderbook side.

    Kalshi orderbook arrays are bid ladders. The highest bid is normally the
    last element, but we use max() defensively in case ordering changes.
    """
    if not entries:
        return None, None
    try:
        price, qty = max(((float(p), float(q)) for p, q in entries), key=lambda x: x[0])
        return price, qty
    except Exception:
        return None, None


def get_live_executable_quote(ticker: str, side: str) -> "tuple[int | None, float | None, float | None, str]":
    """Get the live executable outcome ask from the bid-only Kalshi orderbook.

    Kalshi orderbook endpoint returns bids only:
    - YES bids are counterparties willing to buy YES.
    - NO bids are counterparties willing to buy NO.

    To buy YES immediately, you sell NO / cross the best NO bid:
        executable YES ask = 1 - best NO bid
        V2 payload: side="bid", price=YES ask

    To buy NO immediately, you sell YES / cross the best YES bid:
        executable NO ask = 1 - best YES bid
        V2 payload: side="ask", price=YES bid

    Returns:
        outcome_ask_cents, kalshi_price, available_qty, debug_note
    """
    try:
        book = api_get(f"/trade-api/v2/markets/{ticker}/orderbook").get("orderbook_fp", {})
        yes_bid, yes_qty = _best_bid_from_orderbook(book.get("yes_dollars", []))
        no_bid, no_qty   = _best_bid_from_orderbook(book.get("no_dollars", []))

        if side == "yes":
            if no_bid is None:
                return None, None, None, "no NO bid available to derive executable YES ask"
            outcome_ask = 1.0 - no_bid
            kalshi_price = outcome_ask
            qty = no_qty
            note = (f"best NO bid {round(no_bid*100)}¢×{round(no_qty or 0)} "
                    f"=> executable YES ask {round(outcome_ask*100)}¢")
        elif side == "no":
            if yes_bid is None:
                return None, None, None, "no YES bid available to derive executable NO ask"
            outcome_ask = 1.0 - yes_bid
            kalshi_price = yes_bid
            qty = yes_qty
            note = (f"best YES bid {round(yes_bid*100)}¢×{round(yes_qty or 0)} "
                    f"=> executable NO ask {round(outcome_ask*100)}¢")
        else:
            return None, None, None, f"unknown side {side!r}"

        return round(outcome_ask * 100), kalshi_price, qty, note
    except Exception as e:
        return None, None, None, f"orderbook quote error: {e}"


def place_order(ticker: str, side: str, ask: float) -> bool:
    """Place an IOC order using Kalshi V2 /portfolio/events/orders semantics.

    This version does NOT trust the earlier market quote once a trigger fires.
    It refreshes the bid-only orderbook immediately before each IOC and derives
    the currently executable outcome ask from the opposite bid ladder.

    IMPORTANT:
    - V2 order "side" is the YES book side, not the outcome name.
    - Buying YES: side="bid", price = executable YES ask.
    - Buying NO:  side="ask", price = best YES bid, because selling YES into
      the YES bid is economically the same as buying NO at 1 - YES bid.
    """
    try:
        balance = get_balance()
        log(f"  Balance: ${balance:,.2f}")
    except Exception as e:
        log(f"  Could not verify balance: {e} — skipping order")
        return False

    trigger_min_cents = BUY_MIN_CENTS
    trigger_max_cents = BUY_MAX_CENTS - 1
    path = "/trade-api/v2/portfolio/events/orders"

    # Once a trigger has passed all filters, buy the live executable quote even
    # if the refreshed price is outside the original trigger band. The 98¢–99¢
    # band is only the signal that starts execution; it is not a post-trigger
    # limit price. A few rapid quote-refresh attempts help with races where the
    # counterpart bid flickers.
    for attempt_num in range(1, 6):
        live_cents, kalshi_price, available_qty, quote_note = get_live_executable_quote(ticker, side)
        log(f"  🔎 Live executable quote attempt {attempt_num}: {quote_note}")

        if live_cents is None or kalshi_price is None:
            time.sleep(0.05)
            continue

        if live_cents < trigger_min_cents:
            log(f"  ✅ Live {side.upper()} ask improved to {live_cents}¢ "
                f"below trigger band {trigger_min_cents}¢–{trigger_max_cents}¢ — buying")
        elif live_cents > trigger_max_cents:
            log(f"  ⚠ Live {side.upper()} ask worsened to {live_cents}¢ "
                f"above trigger band {trigger_min_cents}¢–{trigger_max_cents}¢ — buying anyway")

        contracts = (BET_DOLLARS * 100) // live_cents
        if contracts < 1:
            log(f"  SKIP — not enough funds for 1 contract at {live_cents}¢")
            return False

        # Do not request more contracts than visible executable qty if the top
        # level is thinner than the intended size.
        if available_qty is not None and available_qty > 0:
            contracts = min(int(contracts), int(available_qty))
            if contracts < 1:
                log("  MISS — executable top-level quantity disappeared")
                time.sleep(0.05)
                continue

        outcome_price  = live_cents / 100.0
        fee            = math.ceil(0.07 * contracts * outcome_price * (1 - outcome_price) * 100) / 100
        attempt_cost   = round(contracts * outcome_price, 2)
        attempt_profit = round(float(contracts) - attempt_cost, 2)

        if side == "yes":
            book_side = "bid"
            conversion_note = f"YES buy {live_cents}¢ => YES bid {round(kalshi_price * 100)}¢"
        elif side == "no":
            book_side = "ask"
            conversion_note = f"NO buy {live_cents}¢ => YES ask/bid-cross {round(kalshi_price * 100)}¢"
        else:
            log(f"  ❌ ORDER FAILED — unknown side {side!r}")
            return False

        payload = {
            "ticker":                     ticker,
            "client_order_id":            str(uuid.uuid4()),
            "side":                       book_side,
            "count":                      f"{int(contracts)}.00",
            "price":                      f"{kalshi_price:.4f}",
            "time_in_force":              "immediate_or_cancel",
            "self_trade_prevention_type": "taker_at_cross",
        }

        log(f"  🧾 Submit IOC — {conversion_note}; payload side={book_side} "
            f"price={kalshi_price:.4f} count={contracts}")

        try:
            r = requests.post(f"{BASE_URL}{path}", headers=_auth("POST", path),
                              json=payload, timeout=8)
            log(f"  🧾 Order HTTP {r.status_code}: {r.text[:500]}")

            if r.status_code == 201:
                order      = r.json()
                fill_count = order.get("fill_count", "0")
                remaining  = order.get("remaining_count", "0")
                order_id   = order.get("order_id", "?")
                got_fill   = float(fill_count) > 0
                status     = "filled" if got_fill else "unfilled"
                log(f"  {'✅ ORDER PLACED' if got_fill else '⚠ NO FILL'} — "
                    f"{side.upper()} x{contracts} @ {live_cents}¢  "
                    f"cost=${attempt_cost:.2f}  fee=${fee:.2f}  "
                    f"net profit if win=${attempt_profit - fee:.2f}  "
                    f"order_id={order_id}  filled={fill_count}  remaining={remaining}  status={status}")
                if got_fill:
                    return True
                time.sleep(0.05)
                continue

            if r.status_code == 400 and "invalid_price" in r.text:
                log("  ⚡ MISSED — price invalid or market moved before order filled")
                return False

            log(f"  ❌ ORDER FAILED — HTTP {r.status_code}: {r.text[:300]}")
            time.sleep(0.05)
            continue
        except Exception as e:
            log(f"  ❌ ORDER EXCEPTION — {e}")
            time.sleep(0.05)
            continue

    log(f"  ❌ NO FILL after 5 live quote attempts — giving up on this market")
    return False

def _check_loss_after_settlement(ticker: str, side: str):
    log(f"  Waiting for settlement on {ticker}…")
    for _ in range(60):
        time.sleep(15)
        try:
            data = api_get("/trade-api/v2/portfolio/settlements",
                           params={"limit": 50}, auth=True)
            for s in data.get("settlements", []):
                if s.get("ticker") == ticker:
                    result = s.get("market_result", "?")
                    if result == side:
                        log(f"  ✅ WIN — result={result}")
                    else:
                        log(f"  🛑 LOSS DETECTED — result={result} but bet was {side}. Shutting down.")
                        sys.exit(1)
                    return
        except Exception as e:
            log(f"  settlement check error: {e}")
    log(f"  ⚠ Settlement timeout for {ticker} — continuing")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("Last-Second Buyer — BTC 15-min  LIVE TRADING")
    log(f"Buy: {BUY_MIN_CENTS}¢–{BUY_MAX_CENTS-1}¢  |  "
        f"Price-triggered  |  Bet: ${BET_DOLLARS}  |  "
        f"Watch: {WATCH_SECS_WEEKDAY}s weekdays / {WATCH_SECS_WEEKEND}s weekends  |  "
        f"Entry starts: T-{ENTRY_START_SECS}s  |  "
        f"Max opposing: {MAX_OPPOSING_CENTS}¢  |  "
        f"Climb filter: {CLIMB_MAX_CENTS}¢/{CLIMB_LOOKBACK_SECS}s  |  "
        f"Buffer: ${BTC_MIN_BUFFER:,.0f} (reset threshold ${BTC_RESET_BUFFER:,.0f})  |  "
        f"Sustained buffer: {BUFFER_MIN_SUSTAINED_SECS}s mid-window (bypassed if buffer present at window open)  |  "
        f"Marginal certainty: {MARGINAL_CERTAINTY_MIN_CENTS}¢ needs "
        f"${MARGINAL_CERTAINTY_BUFFER:,.0f} buffer at T>{MARGINAL_CERTAINTY_MIN_SECS}s  |  "
        f"Projected buffer: ${BTC_MIN_PROJECTED_BUFFER:,.0f} (skipped at T<{BUFFER_SKIP_PROJECTED_SECS}s)  |  "
        f"ROC filter: ${BTC_MAX_ROC_MOVE:,.0f}/{BTC_ROC_LOOKBACK_MINS}min (directional)  |  "
        f"Recent range filter: ${RECENT_RANGE_MIN_MOVE:,.0f}/{RECENT_RANGE_WINDOW_SECS//60}min "
        f"inside elapsed watch window needs {RECENT_RANGE_BUFFER_MULT:.2f}x buffer  |  "
        f"Bounce filter: >${BOUNCE_MIN_EXCURSION:,.0f} cross w/ T>{BOUNCE_MIN_T_REMAINING}s  |  "
        f"Contested-strike: {CONTESTED_CROSS_MAX_COUNT}+ crosses with >=${CONTESTED_CROSS_EXCURSION:,.0f} excursion  |  "
        f"Shallow book: <{SHALLOW_BOOK_MAX_CONTRACTS:,} contracts AND >{SHALLOW_BOOK_MIN_RATIO}x ratio  |  "
        f"Post-climb veto: stall<${POST_CLIMB_STALL_THRESHOLD:,.0f} OR gap>{POST_CLIMB_SAFETY_RATIO:.0%}x retracement risk")
    log("=" * 60)

    tracker = BtcSpotTracker()
    tracker.start()
    time.sleep(4)

    spot = tracker.latest()
    if spot is not None and spot < 10_000:
        log(f"ABORT — spot price ${spot:.2f} looks like ETH, not BTC. Check brti_btc module.")
        sys.exit(1)
    if spot is not None:
        log(f"BTC spot confirmed: ${spot:,.0f}")

    try:
        bal = get_balance()
        log(f"Opening balance: ${bal:,.2f}")
    except Exception as e:
        log(f"ABORT — could not fetch balance: {e}")
        sys.exit(1)

    try:
        series_ticker = find_btc_series()
        log(f"BTC series: {series_ticker}")
    except Exception as e:
        log(f"ABORT — {e}")
        sys.exit(1)

    wait_for_fresh_startup_market(series_ticker)

    fired_tickers = set()
    bounce        = BounceTrackerV2()
    contested     = ContestedStrikeTracker()

    # ── Post-climb-skip veto state ─────────────────────────────────────────────
    btc_at_prior_skip: "float | None" = None

    try:
        while True:
            market = get_current_btc_market(series_ticker)

            if market is None:
                log("No open BTC market — retrying in 30s")
                time.sleep(2)
                continue

            ticker    = market["ticker"]
            secs_left = secs_to_close(market)
            strike    = _extract_strike(market)

            if strike is None:
                try:
                    full = api_get(f"/trade-api/v2/markets/{ticker}").get("market", {})
                    strike = _extract_strike(full)
                    if strike:
                        log(f"  🎯 Strike resolved via individual market fetch: ${strike:,.0f}")
                    else:
                        log(f"  ⚠ Strike still unknown. floor_strike={full.get('floor_strike')!r}  "
                            f"yes_sub_title={full.get('yes_sub_title')!r}")
                except Exception as e:
                    log(f"  ⚠ Could not resolve strike: {e}")

            if ticker in fired_tickers:
                wait = max(secs_left - 5, 5)
                log(f"Already fired {ticker} — sleeping {wait:.0f}s for next market")
                time.sleep(wait)
                continue

            strike_str = f"${strike:,.0f}" if strike else "?"
            log(f"Market: {ticker}  strike={strike_str}  closes in {secs_left:.0f}s")

            watch_secs = get_watch_secs()
            wait       = secs_left - watch_secs
            if wait > 0:
                log(f"Sleeping {wait:.0f}s until watch window [{day_mode()}]")
                time.sleep(wait)

            log(f"Watching — trigger: {BUY_MIN_CENTS}¢–{BUY_MAX_CENTS-1}¢ on YES or NO; entries allowed at T-{ENTRY_START_SECS}s")
            bounce.reset(strike=strike)
            contested.reset(strike=strike)
            price_history             = deque(maxlen=200)
            buffer_sufficient_since: "float | None" = None  # tracks sustained buffer age
            buffer_at_window_open     = False                # True if buffer was ≥$50 on first poll
            first_poll                = True                 # used to detect window-open state
            watch_started_at          = time.time()          # when we entered the watch window
            last_logged               = None

            while True:
                secs_left = secs_to_close(market)

                if secs_left < 0:
                    log("  Market closed without trigger — moving on")
                    fired_tickers.add(ticker)
                    break

                prices = get_prices(ticker)
                if prices:
                    ya       = prices["yes_ask"]
                    na       = prices["no_ask"]
                    ya_cents = round(ya * 100) if ya else None
                    na_cents = round(na * 100) if na else None

                    if strike is None and prices.get("strike") is not None:
                        strike = prices["strike"]
                        log(f"  🎯 Strike resolved from market API: ${strike:,.0f}")
                        bounce.reset(strike=strike)
                        contested.reset(strike=strike)

                    price_history.append((time.time(), ya_cents, na_cents))

                    btc_spot = tracker.latest()
                    if strike is not None and btc_spot is not None:
                        bounce.update(btc_spot, secs_left)
                        contested.update(btc_spot, secs_left)
                        contested_skip, contested_reason = contested.should_skip()
                        if contested_skip:
                            log(f"  ⚠ SKIP — {contested_reason}")
                            fired_tickers.add(ticker)
                            break

                    # ── Update sustained buffer tracker ────────────────────────
                    # current_btc_dist and current_btc_mom are snapshotted once here
                    # and reused everywhere below — avoids calling tracker methods
                    # multiple times and getting different values from the background thread.
                    # Start timer when buffer hits BTC_MIN_BUFFER ($50).
                    # Reset timer only when buffer drops below BTC_RESET_BUFFER ($45)
                    # to prevent minor jitter from resetting the streak.
                    current_btc_dist = tracker.distance_from_strike(strike)
                    current_btc_mom  = tracker.momentum(lookback_secs=60)
                    if strike is not None and current_btc_dist is not None:
                        if abs(current_btc_dist) >= BTC_MIN_BUFFER:
                            if first_poll:
                                buffer_at_window_open = True
                            if buffer_sufficient_since is None:
                                buffer_sufficient_since = time.time()
                        elif abs(current_btc_dist) < BTC_RESET_BUFFER:
                            buffer_sufficient_since = None  # genuine dip — reset streak
                        # between BTC_RESET_BUFFER and BTC_MIN_BUFFER: hold streak, don't reset
                    if first_poll:
                        first_poll = False

                    display = f"YES={ya_cents}¢  NO={na_cents}¢"
                    if display != last_logged:
                        log(f"  👁 T-{secs_left:.0f}s — {display}")
                        last_logged   = display
                        dominant_side = "yes" if (ya_cents or 0) > (na_cents or 0) else "no"
                        log_btc_info(tracker, strike, dominant_side)

                    if ya_cents == 100 and (na_cents is None or na_cents < BUY_MIN_CENTS):
                        log("  ⚠ SKIP — YES already at 100¢, no tradeable entry — moving on")
                        fired_tickers.add(ticker)
                        break

                    if na_cents == 100 and (ya_cents is None or ya_cents < BUY_MIN_CENTS):
                        log("  ⚠ SKIP — NO already at 100¢, no tradeable entry — moving on")
                        fired_tickers.add(ticker)
                        break

                    # ── Entry gate ────────────────────────────────────────────
                    # We begin watching at T-540s to build price/volatility context,
                    # but we do not place trades until T-300s or later.
                    if secs_left > ENTRY_START_SECS:
                        time.sleep(POLL_MS)
                        continue

                    # ── YES trigger ───────────────────────────────────────────
                    if ya_cents and BUY_MIN_CENTS <= ya_cents < BUY_MAX_CENTS:
                        if strike is None:
                            log("  ⚠ SKIP — strike unknown, BTC filters blind")
                            fired_tickers.add(ticker); break
                        if current_btc_dist is None:
                            log("  ⚠ SKIP — BTC spot unavailable, cannot validate buffer")
                            fired_tickers.add(ticker); break
                        if na_cents and na_cents > MAX_OPPOSING_CENTS:
                            log(f"  ⚠ SKIP — opposing NO too high ({na_cents}¢ > {MAX_OPPOSING_CENTS}¢)")
                            fired_tickers.add(ticker); break

                        climbed, reason = price_climbed_too_fast("yes", price_history)
                        if climbed:
                            if secs_left <= CLIMB_BYPASS_SECS:
                                log(f"  ✅ Climb filter bypassed — only {secs_left:.0f}s left")
                            else:
                                log(f"  ⚠ SKIP — climb filter: {reason}")
                                fired_tickers.add(ticker)
                                break

                        too_close, reason = buffer_too_small("yes", current_btc_dist, current_btc_mom, secs_left)
                        if too_close:
                            log(f"  ⚠ SKIP — buffer filter: {reason}")
                            fired_tickers.add(ticker); break

                        not_sustained, reason = buffer_not_sustained("yes", buffer_sufficient_since, buffer_at_window_open)
                        if not_sustained:
                            log(f"  ⚠ SKIP — sustained buffer filter: {reason}")
                            fired_tickers.add(ticker); break

                        marginal, reason = marginal_certainty_veto("yes", ya_cents, current_btc_dist, secs_left)
                        if marginal:
                            log(f"  ⚠ SKIP — {reason}")
                            fired_tickers.add(ticker); break

                        too_fast, reason = roc_too_high(tracker, "yes", strike)
                        if too_fast:
                            log(f"  ⚠ SKIP — ROC filter: {reason}")
                            fired_tickers.add(ticker); break

                        range_vetoed, reason = recent_range_veto(tracker, "yes", current_btc_dist, time.time() - watch_started_at)
                        if range_vetoed:
                            log(f"  ⚠ SKIP — {reason}")
                            fired_tickers.add(ticker); break

                        bounced, reason = bounce.should_skip("yes")
                        if bounced:
                            log(f"  ⚠ SKIP — bounce filter: {reason}")
                            fired_tickers.add(ticker); break

                        yes_total, no_total = get_orderbook_totals(ticker)
                        is_shallow, reason = shallow_book("yes", yes_total, no_total)
                        if is_shallow:
                            log(f"  ⚠ SKIP — shallow book filter: {reason}")
                            fired_tickers.add(ticker); break

                        if btc_at_prior_skip is not None:
                            log("  ℹ️  Post-climb-skip veto still armed — carries to next market")

                        buffer_age = time.time() - buffer_sufficient_since if buffer_sufficient_since else 0
                        log(f"  ⏱ Buffer sustained for {buffer_age:.0f}s before trigger")
                        log_btc_info(tracker, strike, "yes")
                        log(f"  ⚡ TRIGGER YES — {ya_cents}¢ at T-{secs_left:.0f}s  opposing={na_cents}¢")
                        log_orderbook_depth(ticker, "yes")
                        filled = place_order(ticker, "yes", ya)
                        fired_tickers.add(ticker)
                        if filled:
                            _check_loss_after_settlement(ticker, "yes")
                        else:
                            log("  No fill — moving on without settlement check")
                        break

                    # ── NO trigger ────────────────────────────────────────────
                    if na_cents and BUY_MIN_CENTS <= na_cents < BUY_MAX_CENTS:
                        if strike is None:
                            log("  ⚠ SKIP — strike unknown, BTC filters blind")
                            fired_tickers.add(ticker); btc_at_prior_skip = None; break
                        if current_btc_dist is None:
                            log("  ⚠ SKIP — BTC spot unavailable, cannot validate buffer")
                            fired_tickers.add(ticker); btc_at_prior_skip = None; break
                        if ya_cents and ya_cents > MAX_OPPOSING_CENTS:
                            log(f"  ⚠ SKIP — opposing YES too high ({ya_cents}¢ > {MAX_OPPOSING_CENTS}¢)")
                            fired_tickers.add(ticker); btc_at_prior_skip = None; break

                        climbed, reason = price_climbed_too_fast("no", price_history)
                        if climbed:
                            if secs_left <= CLIMB_BYPASS_SECS:
                                log(f"  ✅ Climb filter bypassed — only {secs_left:.0f}s left")
                            else:
                                log(f"  ⚠ SKIP — climb filter: {reason}")
                                fired_tickers.add(ticker)
                                btc_at_prior_skip = tracker.latest()
                                log(f"  🚩 Post-climb-skip veto armed — BTC recorded at ${btc_at_prior_skip:,.0f}")
                                break

                        too_close, reason = buffer_too_small("no", current_btc_dist, current_btc_mom, secs_left)
                        if too_close:
                            log(f"  ⚠ SKIP — buffer filter: {reason}")
                            fired_tickers.add(ticker); btc_at_prior_skip = None; break

                        not_sustained, reason = buffer_not_sustained("no", buffer_sufficient_since, buffer_at_window_open)
                        if not_sustained:
                            log(f"  ⚠ SKIP — sustained buffer filter: {reason}")
                            fired_tickers.add(ticker); btc_at_prior_skip = None; break

                        marginal, reason = marginal_certainty_veto("no", na_cents, current_btc_dist, secs_left)
                        if marginal:
                            log(f"  ⚠ SKIP — {reason}")
                            fired_tickers.add(ticker); btc_at_prior_skip = None; break

                        too_fast, reason = roc_too_high(tracker, "no", strike)
                        if too_fast:
                            log(f"  ⚠ SKIP — ROC filter: {reason}")
                            fired_tickers.add(ticker); btc_at_prior_skip = None; break

                        range_vetoed, reason = recent_range_veto(tracker, "no", current_btc_dist, time.time() - watch_started_at)
                        if range_vetoed:
                            log(f"  ⚠ SKIP — {reason}")
                            fired_tickers.add(ticker); btc_at_prior_skip = None; break

                        bounced, reason = bounce.should_skip("no")
                        if bounced:
                            log(f"  ⚠ SKIP — bounce filter: {reason}")
                            fired_tickers.add(ticker); btc_at_prior_skip = None; break

                        yes_total_ob, no_total_ob = get_orderbook_totals(ticker)
                        is_shallow, reason = shallow_book("no", yes_total_ob, no_total_ob)
                        if is_shallow:
                            log(f"  ⚠ SKIP — shallow book filter: {reason}")
                            fired_tickers.add(ticker); btc_at_prior_skip = None; break

                        current_btc = tracker.latest()
                        vetoed, reason = post_climb_skip_veto(
                            "no", current_btc_dist, btc_at_prior_skip, current_btc
                        )
                        if vetoed:
                            log(f"  ⚠ SKIP — {reason}")
                            fired_tickers.add(ticker)
                            btc_at_prior_skip = None
                            break

                        btc_at_prior_skip = None

                        buffer_age = time.time() - buffer_sufficient_since if buffer_sufficient_since else 0
                        log(f"  ⏱ Buffer sustained for {buffer_age:.0f}s before trigger")
                        log_btc_info(tracker, strike, "no")
                        log(f"  ⚡ TRIGGER NO — {na_cents}¢ at T-{secs_left:.0f}s  opposing={ya_cents}¢")
                        log_orderbook_depth(ticker, "no")
                        filled = place_order(ticker, "no", na)
                        fired_tickers.add(ticker)
                        if filled:
                            _check_loss_after_settlement(ticker, "no")
                        else:
                            log("  No fill — moving on without settlement check")
                        break

                time.sleep(POLL_MS)

    except KeyboardInterrupt:
        log("Stopped by user.")
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()
