#!/usr/bin/env python3
"""
BRTI - CME CF Bitcoin Real Time Index Replication
Implements the exact CF Benchmarks methodology (v16.5, Feb 2026):
  - Fetches order books from all 5 constituent exchanges simultaneously
  - Applies dynamic order size cap
  - Builds consolidated order book
  - Computes mid price-volume curve
  - Finds utilized depth (0.5% deviation threshold)
  - Weights by exponential distribution (lambda = 1/0.37)
  - Returns BRTI price

Constituent exchanges: Bitstamp, Coinbase, Gemini, itBit (Paxos), Kraken
Parameters (from CF Benchmarks methodology doc section 6):
  Spacing (s)         = 1 BTC
  Deviation from mid (D) = 0.5%
  Lambda              = 1 / 0.37
  Potentially erroneous data parameter = 5%
"""

import math
import time
import datetime
import concurrent.futures
import requests
import numpy as np

# ── BRTI Parameters (Section 6 of CF Benchmarks methodology) ─────────────────

SPACING        = 1.0          # BTC — granularity of price-volume curve
DEVIATION      = 0.005        # 0.5% — max mid spread before depth is cut off
LAMBDA         = 1.0 / 0.37   # exponential distribution parameter
ERRONEOUS_PCT  = 0.05         # 5% — exchange excluded if mid deviates this much
STALE_SECS     = 30           # seconds before an exchange is considered stale
WINSOR_FRAC    = 0.25         # fraction of size samples to winsorize each side
WINSOR_MULT    = 3.0          # cap = winsorized_mean + WINSOR_MULT * winsorized_std


# ── Order book fetchers (one per exchange) ────────────────────────────────────

def fetch_bitstamp() -> dict:
    resp = requests.get(
        "https://www.bitstamp.net/api/v2/order_book/btcusd/",
        params={"group": 1},
        timeout=8,
    )
    resp.raise_for_status()
    data = resp.json()
    bids = [[float(p), float(s)] for p, s in data["bids"]]
    asks = [[float(p), float(s)] for p, s in data["asks"]]
    return {"exchange": "bitstamp", "bids": bids, "asks": asks,
            "retrieved_at": time.time()}


def fetch_coinbase() -> dict:
    resp = requests.get(
        "https://api.exchange.coinbase.com/products/BTC-USD/book",
        params={"level": 2},
        timeout=8,
    )
    resp.raise_for_status()
    data = resp.json()
    bids = [[float(p), float(s)] for p, s, _ in data["bids"]]
    asks = [[float(p), float(s)] for p, s, _ in data["asks"]]
    return {"exchange": "coinbase", "bids": bids, "asks": asks,
            "retrieved_at": time.time()}


def fetch_gemini() -> dict:
    resp = requests.get(
        "https://api.gemini.com/v1/book/btcusd",
        params={"limit_bids": 0, "limit_asks": 0},
        timeout=8,
    )
    resp.raise_for_status()
    data = resp.json()
    bids = [[float(e["price"]), float(e["amount"])] for e in data["bids"]]
    asks = [[float(e["price"]), float(e["amount"])] for e in data["asks"]]
    return {"exchange": "gemini", "bids": bids, "asks": asks,
            "retrieved_at": time.time()}


def fetch_kraken() -> dict:
    resp = requests.get(
        "https://api.kraken.com/0/public/Depth",
        params={"pair": "XBTUSD", "count": 500},
        timeout=8,
    )
    resp.raise_for_status()
    data = resp.json()
    result = data.get("result", {})
    book = next((v for k, v in result.items() if k != "last"), None)
    if not book:
        raise ValueError("Kraken: no order book in response")
    bids = [[float(p), float(s)] for p, s, _ in book["bids"]]
    asks = [[float(p), float(s)] for p, s, _ in book["asks"]]
    return {"exchange": "kraken", "bids": bids, "asks": asks,
            "retrieved_at": time.time()}


def fetch_paxos() -> dict:
    # Paxos v2 public market data (replaced itBit REST API Nov 2022)
    resp = requests.get(
        "https://api.paxos.com/v2/markets/BTCUSD/order-book",
        params={"depth": 200},
        timeout=8,
    )
    resp.raise_for_status()
    data = resp.json()
    bids = [[float(e["price"]), float(e["amount"])] for e in data.get("bids", [])]
    asks = [[float(e["price"]), float(e["amount"])] for e in data.get("asks", [])]
    return {"exchange": "paxos", "bids": bids, "asks": asks,
            "retrieved_at": time.time()}


FETCHERS = [fetch_bitstamp, fetch_coinbase, fetch_gemini, fetch_kraken, fetch_paxos]


# ── Step 1: Fetch all books simultaneously ────────────────────────────────────

def fetch_all_books(calc_time: float) -> list[dict]:
    """Fetch order books from all exchanges concurrently. Returns valid books only."""
    books = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fn): fn.__name__ for fn in FETCHERS}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                book = future.result()
                # Stale check: discard if retrieved >30s before calc_time
                age = calc_time - book["retrieved_at"]
                if age > STALE_SECS:
                    print(f"  [{book['exchange']}] STALE ({age:.1f}s) — excluded")
                    continue
                # Basic validity checks
                if not book["bids"] or not book["asks"]:
                    print(f"  [{book['exchange']}] empty book — excluded")
                    continue
                books.append(book)
            except Exception as e:
                print(f"  [{name}] fetch error: {e} — excluded")
    return books


# ── Step 2: Potentially erroneous data check ──────────────────────────────────

def filter_erroneous(books: list[dict]) -> list[dict]:
    """
    Exclude any exchange whose mid-price deviates more than 5% from the
    median mid-price across all exchanges (Section 5.3).
    """
    if len(books) < 2:
        return books

    mids = []
    for b in books:
        best_bid = max(p for p, _ in b["bids"])
        best_ask = min(p for p, _ in b["asks"])
        mids.append((best_bid + best_ask) / 2.0)

    median_mid = float(np.median(mids))
    valid = []
    for b, mid in zip(books, mids):
        dev = abs(mid - median_mid) / median_mid
        if dev > ERRONEOUS_PCT:
            print(f"  [{b['exchange']}] mid ${mid:,.2f} deviates {dev*100:.1f}% "
                  f"from median ${median_mid:,.2f} — excluded")
        else:
            valid.append(b)
    return valid


# ── Step 3: Dynamic order size cap (Section 4.1.3) ───────────────────────────

def compute_order_size_cap(books: list[dict]) -> float:
    """
    Compute the dynamic order size cap from all bid and ask sizes.
    Cap = winsorized_mean + 3 * winsorized_std
    """
    all_sizes = []
    for b in books:
        all_sizes.extend(s for _, s in b["bids"])
        all_sizes.extend(s for _, s in b["asks"])

    sizes = sorted(all_sizes)
    n = len(sizes)
    if n < 4:
        return float("inf")

    k = max(1, math.floor(WINSOR_FRAC * n))  # samples to trim each side
    # Winsorize: replace bottom k with sizes[k], top k with sizes[n-k-1]
    winsorized = (
        [sizes[k]] * k
        + sizes[k:n - k]
        + [sizes[n - k - 1]] * k
    )
    w_mean = float(np.mean(winsorized))
    w_std  = float(np.std(winsorized, ddof=1))
    cap    = w_mean + WINSOR_MULT * w_std
    return cap


# ── Step 4: Build consolidated order book ─────────────────────────────────────

def consolidate_books(books: list[dict], cap: float) -> tuple[list, list]:
    """
    Merge all exchange order books into one consolidated book.
    Sizes are capped at `cap`. Levels at the same price are summed.
    Returns (bids, asks) each as list of [price, size] sorted correctly.
    """
    bid_map: dict[float, float] = {}
    ask_map: dict[float, float] = {}

    for b in books:
        for price, size in b["bids"]:
            capped = min(size, cap)
            bid_map[price] = bid_map.get(price, 0.0) + capped
        for price, size in b["asks"]:
            capped = min(size, cap)
            ask_map[price] = ask_map.get(price, 0.0) + capped

    bids = sorted(bid_map.items(), key=lambda x: -x[0])  # descending price
    asks = sorted(ask_map.items(), key=lambda x: x[0])   # ascending price
    return bids, asks


# ── Step 5: Build price-volume curves ─────────────────────────────────────────

def build_price_volume_curves(bids: list, asks: list, spacing: float):
    """
    Build the ask, bid, mid, and mid-spread volume curves at granularity=spacing.
    Returns lists of (volume, ask_price, bid_price, mid_price, mid_spread).
    """
    # Cumulative volume steps
    max_vol = min(
        sum(s for _, s in bids),
        sum(s for _, s in asks),
    )
    if max_vol <= 0:
        return []

    volumes = []
    v = spacing
    while v <= max_vol + spacing:
        volumes.append(v)
        v += spacing

    def marginal_ask(vol: float) -> float:
        """Marginal price to buy `vol` BTC from ask side."""
        cum = 0.0
        for price, size in asks:
            cum += size
            if cum >= vol:
                return price
        return asks[-1][0]

    def marginal_bid(vol: float) -> float:
        """Marginal price to sell `vol` BTC into bid side."""
        cum = 0.0
        for price, size in bids:
            cum += size
            if cum >= vol:
                return price
        return bids[-1][0]

    curves = []
    for vol in volumes:
        ask_p = marginal_ask(vol)
        bid_p = marginal_bid(vol)
        mid_p = (ask_p + bid_p) / 2.0
        mid_s = (ask_p / mid_p) - 1.0  # percentage deviation
        curves.append((vol, ask_p, bid_p, mid_p, mid_s))

    return curves


# ── Step 6: Find utilized depth ───────────────────────────────────────────────

def find_utilized_depth(curves: list, deviation: float, spacing: float) -> float:
    """
    Find the maximum volume for which mid_spread <= deviation.
    If none qualifies, return spacing (minimum).
    """
    utilized = spacing
    for vol, ask_p, bid_p, mid_p, mid_s in curves:
        if mid_s <= deviation:
            utilized = vol
        else:
            break
    return utilized


# ── Step 7: Exponential weighting and BRTI ────────────────────────────────────

def compute_brti(curves: list, utilized_depth: float, lam: float,
                 spacing: float) -> float:
    """
    Weight the mid price-volume curve by normalized exponential PDF
    up to utilized_depth, then sum to get BRTI.
    Eq. 3 from methodology.
    """
    # Filter to utilized depth
    used = [(vol, mid_p) for vol, _, _, mid_p, _ in curves if vol <= utilized_depth]
    if not used:
        return curves[0][3] if curves else 0.0

    # Exponential PDF weights: w(v) = lambda * exp(-lambda * v)
    # Normalized over the discrete set of volumes
    raw_weights = [lam * math.exp(-lam * vol) for vol, _ in used]
    total_weight = sum(raw_weights)
    if total_weight == 0:
        return used[0][1]

    brti = sum(w / total_weight * mid_p for (_, mid_p), w in zip(used, raw_weights))
    return round(brti, 2)


# ── Main calculation ──────────────────────────────────────────────────────────

def calculate_brti(verbose: bool = True) -> float | None:
    """
    Run the full BRTI calculation. Returns the index value or None on failure.
    """
    calc_time = time.time()

    if verbose:
        print("\n  Fetching order books from all 5 constituent exchanges...", flush=True)

    books = fetch_all_books(calc_time)

    if verbose:
        print(f"  Valid books: {[b['exchange'] for b in books]}")

    books = filter_erroneous(books)

    if len(books) == 0:
        print("  BRTI calculation failure: no valid order books")
        return None

    cap = compute_order_size_cap(books)
    if verbose:
        print(f"  Order size cap: {cap:.4f} BTC")

    bids, asks = consolidate_books(books, cap)

    if not bids or not asks:
        print("  BRTI calculation failure: empty consolidated book")
        return None

    curves = build_price_volume_curves(bids, asks, SPACING)

    if not curves:
        print("  BRTI calculation failure: could not build curves")
        return None

    utilized_depth = find_utilized_depth(curves, DEVIATION, SPACING)

    if verbose:
        print(f"  Utilized depth: {utilized_depth:.1f} BTC")

    brti = compute_brti(curves, utilized_depth, LAMBDA, SPACING)

    if verbose:
        exchanges_used = [b["exchange"] for b in books]
        print(f"  Exchanges used: {', '.join(exchanges_used)}")
        print(f"  BRTI: ${brti:,.2f}")

    return brti


if __name__ == "__main__":
    result = calculate_brti(verbose=True)
    if result:
        print(f"\n  Final BRTI: ${result:,.2f}")


# ── Fast BRTI (best bid/ask mid-price from all 5 exchanges) ──────────────────

def fetch_mid_bitstamp() -> float:
    resp = requests.get("https://www.bitstamp.net/api/v2/ticker/btcusd/", timeout=4)
    resp.raise_for_status()
    d = resp.json()
    return (float(d["bid"]) + float(d["ask"])) / 2.0

def fetch_mid_coinbase() -> float:
    resp = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/ticker", timeout=4)
    resp.raise_for_status()
    d = resp.json()
    return (float(d["bid"]) + float(d["ask"])) / 2.0

def fetch_mid_gemini() -> float:
    resp = requests.get("https://api.gemini.com/v1/pubticker/btcusd", timeout=4)
    resp.raise_for_status()
    d = resp.json()
    return (float(d["bid"]) + float(d["ask"])) / 2.0

def fetch_mid_kraken() -> float:
    resp = requests.get("https://api.kraken.com/0/public/Ticker",
                        params={"pair": "XBTUSD"}, timeout=4)
    resp.raise_for_status()
    d = resp.json()
    ticker = next(iter(d["result"].values()))
    return (float(ticker["b"][0]) + float(ticker["a"][0])) / 2.0

def fetch_mid_paxos() -> float:
    resp = requests.get("https://api.paxos.com/v2/markets/BTCUSD/ticker", timeout=4)
    resp.raise_for_status()
    d = resp.json()
    # Paxos may return price as a string, float, or nested dict with a 'value' key
    def parse_price(v):
        if isinstance(v, dict):
            return float(v.get("value") or v.get("price") or next(iter(v.values())))
        return float(v)
    return (parse_price(d["best_bid"]) + parse_price(d["best_ask"])) / 2.0

FAST_FETCHERS = [fetch_mid_bitstamp, fetch_mid_coinbase, fetch_mid_gemini,
                 fetch_mid_kraken, fetch_mid_paxos]

def calculate_brti_fast(verbose: bool = True) -> float | None:
    mids = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fn): fn.__name__ for fn in FAST_FETCHERS}
        for future in concurrent.futures.as_completed(futures, timeout=5):
            name = futures[future]
            try:
                mid = future.result()
                mids.append(mid)
                if verbose:
                    print(f"    {name.replace('fetch_mid_', ''):>10}: ${mid:,.2f}")
            except Exception as e:
                if verbose:
                    print(f"    {name.replace('fetch_mid_', ''):>10}: error — {e}")

    if not mids:
        print("  Fast BRTI failed: no exchanges responded")
        return None

    median = float(np.median(mids))
    valid  = [m for m in mids if abs(m - median) / median <= ERRONEOUS_PCT]
    if not valid:
        valid = mids

    result = round(float(np.mean(valid)), 2)
    if verbose:
        print(f"  Fast BRTI ({len(valid)}/{len(mids)} exchanges): ${result:,.2f}")
    return result
