#!/usr/bin/env python3
"""
Daily top-up fetch for the public Williams %R screen.

Maintains three small, git-friendly consolidated parquet tables (NOT one file
per ticker -- that's what makes a DAILY rewrite affordable in a public repo):

    data/daily_window.parquet   trailing ~900 trading days OHLC, long format
    data/weekly_bars.parquet    trailing ~300 weekly bars (~5.75y)
    data/monthly_bars.parquet   trailing ~90 monthly bars (~7.5y)

Each run:
  1. Batched yf.download() for the whole universe (+ SPY), a handful of large
     calls, NOT 2,000+ per-ticker calls -- that is the forgiving price-history
     path, distinct from the rate-limited per-ticker .info() metadata path.
  2. Upsert new days into daily_window, trim each ticker to DAILY_KEEP rows.
  3. Recompute ONLY the current (forming) week's and month's bar per ticker
     from the tail of daily_window and upsert that one row -- every prior,
     completed period's row is left untouched by construction (it was
     finalised on the last day that period was current), so there is no
     separate "freeze" step to get wrong.

Universe: meta/tickers.csv (ticker,name,sector,industry,mcap,exch).
Run: python3 scripts/fetch.py
ASCII only.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
META_CSV = ROOT / "meta" / "tickers.csv"

DAILY_KEEP = 900
WEEKLY_KEEP = 300
MONTHLY_KEEP = 90
BATCH = 60
SLEEP = 1.0
MIN_FETCH_DAYS = 10
MAX_FETCH_DAYS = 40


def load_universe():
    tickers = pd.read_csv(META_CSV)["ticker"].astype(str).tolist()
    if "SPY" not in tickers:
        tickers.append("SPY")
    return sorted(set(tickers))


def load_table(name, cols):
    p = DATA / name
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame(columns=cols)


def fetch_batch(tickers, period):
    return yf.download(tickers, period=period, interval="1d", auto_adjust=False,
                        group_by="ticker", threads=True, progress=False)


def extract_rows(d, tickers):
    rows = []
    for t in tickers:
        try:
            if isinstance(d.columns, pd.MultiIndex):
                if t not in set(d.columns.get_level_values(0)):
                    continue
                df = d[t].copy()
            else:
                df = d.copy()
        except Exception:
            continue
        df = df.dropna(how="all")
        if df.empty or "Close" not in df.columns:
            continue
        df = df[["High", "Low", "Close"]].dropna()
        if df.empty:
            continue
        df = df.reset_index().rename(columns={"Date": "date"})
        df.insert(0, "ticker", t)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["ticker", "date", "High", "Low", "Close"])


def fetch_universe(tickers, period):
    ok, fail, chunks = 0, [], []
    for i in range(0, len(tickers), BATCH):
        b = tickers[i:i + BATCH]
        try:
            d = fetch_batch(b, period)
        except Exception as e:
            print(f"  batch err: {e}", file=sys.stderr)
            fail += b
            time.sleep(SLEEP * 3)
            continue
        rows = extract_rows(d, b)
        got = set(rows["ticker"].unique()) if len(rows) else set()
        fail += [t for t in b if t not in got]
        ok += len(got)
        chunks.append(rows)
        if (i // BATCH + 1) % 5 == 0:
            print(f"  {min(i + BATCH, len(tickers))}/{len(tickers)} ok={ok} fail={len(fail)}")
        time.sleep(SLEEP)
    fetched = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["ticker", "date", "High", "Low", "Close"])
    return fetched, fail


def upsert_daily(existing, fresh):
    if len(existing):
        combined = pd.concat([existing, fresh], ignore_index=True)
    else:
        combined = fresh
    combined = combined.drop_duplicates(["ticker", "date"], keep="last")
    combined = combined.sort_values(["ticker", "date"])
    combined = combined.groupby("ticker", group_keys=False).tail(DAILY_KEEP)
    return combined.reset_index(drop=True)


def recompute_current_bars(daily_df):
    """Current (forming) weekly and monthly bar per ticker, from the tail of
    daily_window. Only ever the LAST resampled bar -- prior complete periods
    are never touched here, they were finalised on their own last-current day.
    """
    out_w, out_m = [], []
    for t, g in daily_df.groupby("ticker"):
        g = g.set_index("date")[["High", "Low", "Close"]].sort_index().tail(40)
        if g.empty:
            continue
        wk = g.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
        if len(wk):
            r = wk.iloc[-1]
            out_w.append({"ticker": t, "period": wk.index[-1],
                          "High": float(r.High), "Low": float(r.Low), "Close": float(r.Close)})
        mo = g.resample("ME").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
        if len(mo):
            r = mo.iloc[-1]
            out_m.append({"ticker": t, "period": mo.index[-1],
                          "High": float(r.High), "Low": float(r.Low), "Close": float(r.Close)})
    w = pd.DataFrame(out_w, columns=["ticker", "period", "High", "Low", "Close"])
    m = pd.DataFrame(out_m, columns=["ticker", "period", "High", "Low", "Close"])
    return w, m


def upsert_period(existing, updates, keep):
    if len(existing):
        key = existing.set_index(["ticker", "period"]).index
        mask = ~key.isin(updates.set_index(["ticker", "period"]).index) if len(updates) else np.ones(len(existing), dtype=bool)
        combined = pd.concat([existing[mask], updates], ignore_index=True)
    else:
        combined = updates
    combined = combined.sort_values(["ticker", "period"])
    combined = combined.groupby("ticker", group_keys=False).tail(keep)
    return combined.reset_index(drop=True)


def main():
    DATA.mkdir(exist_ok=True)
    tickers = load_universe()
    print(f"universe: {len(tickers)} tickers (incl. SPY)")

    daily = load_table("daily_window.parquet", ["ticker", "date", "High", "Low", "Close"])
    weekly = load_table("weekly_bars.parquet", ["ticker", "period", "High", "Low", "Close"])
    monthly = load_table("monthly_bars.parquet", ["ticker", "period", "High", "Low", "Close"])

    if len(daily):
        last_seen = daily["date"].max()
        gap_days = (pd.Timestamp.now().normalize() - last_seen).days
        period = f"{min(max(gap_days + 5, MIN_FETCH_DAYS), MAX_FETCH_DAYS)}d"
    else:
        period = f"{MAX_FETCH_DAYS}d"
    print(f"fetch period: {period}")

    fresh, fail = fetch_universe(tickers, period)
    if fail:
        print(f"failed tickers ({len(fail)}): {fail[:30]}{'...' if len(fail) > 30 else ''}", file=sys.stderr)
    if "SPY" not in fresh["ticker"].unique() and "SPY" not in daily.get("ticker", pd.Series(dtype=str)).unique():
        print("FATAL: SPY missing from both the fresh fetch and the existing table", file=sys.stderr)
        sys.exit(1)

    daily = upsert_daily(daily, fresh)
    w_updates, m_updates = recompute_current_bars(daily)
    weekly = upsert_period(weekly, w_updates, WEEKLY_KEEP)
    monthly = upsert_period(monthly, m_updates, MONTHLY_KEEP)

    daily.to_parquet(DATA / "daily_window.parquet", index=False)
    weekly.to_parquet(DATA / "weekly_bars.parquet", index=False)
    monthly.to_parquet(DATA / "monthly_bars.parquet", index=False)

    print(f"DONE daily_rows={len(daily)} weekly_rows={len(weekly)} monthly_rows={len(monthly)} "
          f"as_of={daily['date'].max().date() if len(daily) else '?'}")


if __name__ == "__main__":
    main()
