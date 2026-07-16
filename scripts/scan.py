#!/usr/bin/env python3
"""
Williams %R distance-to-oversold screen -- public, self-contained edition.

Same maths as the private-repo version (fan-inversion trigger prices, weekly
and monthly primary lenses, freshness/lingering diagnostics), but reads from
THREE small consolidated parquet tables instead of one file per ticker, so
this repo stays a few MB and cheap to rewrite daily:

    data/daily_window.parquet   trailing ~900 trading days OHLC, long format
    data/weekly_bars.parquet    trailing ~300 weekly bars (~5.75y)
    data/monthly_bars.parquet   trailing ~90 monthly bars (~7.5y)

These are maintained by scripts/fetch.py. This script only reads them.

One deliberate trade-off vs the private version: the monthly lens's
day-resolution FRESHNESS window is 500 trading days here, not 1260, to fit
inside the retained daily window. This only undercounts the display "days
running" figure for an episode that has run continuously for 2+ years
without a single exit above -50 -- a pathological case. Chronic/episode
count/median length stay bar-based (unaffected, see monthly_primary).

Core inversion: %R = (HH - Close) / (HH - LL) * -100 over n bars.
Trigger price for level L (p = |L|/100): T = (1-p)*HH + p*LL.

Run: python3 scripts/scan.py
ASCII only.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"
META_CSV = ROOT / "meta" / "tickers.csv"

N = 14
ZONE = -80.0
RECOVER = -50.0
STALE_DAYS = 7

LOOKBACK_2Y = 504
CHRONIC_SHARE = 25.0

MONTHLY_FRESH_DAYS = 500     # shrunk from the private repo's 1260 -- see module docstring
MONTHLY_LEAD_DAYS = 300
MONTHLY_CHRONIC_BARS = 60
MONTHLY_CHRONIC_SHARE = 25.0


def load_meta():
    df = pd.read_csv(META_CSV)
    meta = {}
    for r in df.itertuples(index=False):
        meta[r.ticker] = {"name": r.name, "sector": r.sector, "industry": r.industry,
                          "mcap": None if pd.isna(r.mcap) else float(r.mcap),
                          "exch": None if pd.isna(r.exch) else r.exch}
    return meta


def wr(hh, ll, close):
    rng = hh - ll
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(rng > 0, (hh - close) / rng * -100.0, np.nan)
    return v


def period_eval_daily(df, freq):
    """Period %R(14) evaluated on every trading day, freq = 'W-FRI' or 'M'."""
    codes = pd.factorize(df.index.to_period(freq))[0]
    g = pd.Series(codes, index=df.index)
    ph_bars = df["High"].groupby(codes).max()
    pl_bars = df["Low"].groupby(codes).min()
    prev_h = ph_bars.rolling(N - 1).max().shift(1)
    prev_l = pl_bars.rolling(N - 1).min().shift(1)
    ph = prev_h.reindex(codes).to_numpy()
    pl = prev_l.reindex(codes).to_numpy()
    ih = df["High"].groupby(g).cummax().to_numpy()
    il = df["Low"].groupby(g).cummin().to_numpy()
    hh = np.maximum(ph, ih)
    ll = np.minimum(pl, il)
    return pd.Series(wr(hh, ll, df["Close"].to_numpy()), index=df.index)


def bar_wr(frame):
    hh = frame["High"].rolling(N).max()
    ll = frame["Low"].rolling(N).min()
    return pd.Series(wr(hh.to_numpy(), ll.to_numpy(), frame["Close"].to_numpy()),
                     index=frame.index)


def episodes(in_zone, gap=1):
    """Runs of True, tolerating up to `gap` consecutive False bars between them."""
    z = in_zone.to_numpy(copy=True)
    if gap >= 1:
        for i in range(1, len(z) - 1):
            if not z[i] and z[i - 1] and z[i + 1]:
                z[i] = True
    runs, start = [], None
    for i, v in enumerate(z):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(z) - 1))
    return runs


def trigger_from_window(win):
    HH, LL = float(win["High"].max()), float(win["Low"].min())
    if HH - LL <= 0:
        return None
    trig = lambda L: round((1 - abs(L) / 100.0) * HH + abs(L) / 100.0 * LL, 4)
    ll_idx = int(win["Low"].to_numpy().argmin())
    trig_age = (len(win) - 1) - ll_idx
    return trig(-80), trig(-90), trig(-100), trig_age


def weekly_primary(wk, wre, px):
    if len(wk) < N:
        return None
    trig = trigger_from_window(wk.tail(N))
    if trig is None:
        return None
    t80, t90, t100, trig_age = trig
    fall80 = (px - t80) / px * 100.0

    wr_now = float(wre.iloc[-1]) if len(wre) and np.isfinite(wre.iloc[-1]) else None
    if wr_now is None:
        return None

    win2y = wre.tail(LOOKBACK_2Y)
    valid = win2y.notna()
    nvalid = int(valid.sum())
    if nvalid < 30:
        return None
    zin = (win2y <= ZONE).fillna(False)
    runs = episodes(zin)
    durs = [e - s + 1 for s, e in runs]
    recov = []
    arr = win2y.to_numpy()
    for s, e in runs:
        after = np.where(arr[s:] > RECOVER)[0]
        if len(after):
            recov.append(int(after[0]))
    share = float(zin[valid].mean() * 100.0)
    fresh = None
    if runs and runs[-1][1] == len(zin) - 1:
        fresh = durs[-1]

    zone = 100 if wr_now <= -99.95 else 90 if wr_now <= -90 else 80 if wr_now <= ZONE else 0

    rnd = lambda v, d=1: (round(float(v), d) if v is not None and np.isfinite(v) else None)
    return {
        "t80": t80, "t90": t90, "t100": t100, "fall80": rnd(fall80),
        "zone": zone, "fresh": fresh, "epi": len(runs),
        "medlen": rnd(np.median(durs), 0) if durs else None,
        "medrec": rnd(np.median(recov), 0) if recov else None,
        "share": rnd(share), "hist": nvalid, "chronic": share > CHRONIC_SHARE,
        "trig_age": None,
    }


def monthly_primary(mo, moe, px):
    if len(mo) < N:
        return None
    trig = trigger_from_window(mo.tail(N))
    if trig is None:
        return None
    t80, t90, t100, trig_age = trig
    fall80 = (px - t80) / px * 100.0

    wr_now = float(moe.iloc[-1]) if len(moe) and np.isfinite(moe.iloc[-1]) else None
    if wr_now is None:
        return None

    day_win = moe.tail(MONTHLY_FRESH_DAYS)
    zin_days = (day_win <= ZONE).fillna(False)
    day_runs = episodes(zin_days)
    fresh = None
    if day_runs and day_runs[-1][1] == len(zin_days) - 1:
        fresh = day_runs[-1][1] - day_runs[-1][0] + 1

    bwr = bar_wr(mo).tail(MONTHLY_CHRONIC_BARS)
    valid = bwr.notna()
    nvalid = int(valid.sum())
    if nvalid < 6:
        return None
    zin_bars = (bwr <= ZONE).fillna(False)
    runs = episodes(zin_bars, gap=0)
    durs = [e - s + 1 for s, e in runs]
    recov = []
    arr = bwr.to_numpy()
    for s, e in runs:
        after = np.where(arr[s:] > RECOVER)[0]
        if len(after):
            recov.append(int(after[0]))
    share = float(zin_bars[valid].mean() * 100.0)

    zone = 100 if wr_now <= -99.95 else 90 if wr_now <= -90 else 80 if wr_now <= ZONE else 0

    rnd = lambda v, d=1: (round(float(v), d) if v is not None and np.isfinite(v) else None)
    return {
        "t80": t80, "t90": t90, "t100": t100, "fall80": rnd(fall80),
        "zone": zone, "fresh": fresh, "epi": len(runs),
        "medlen": rnd(np.median(durs), 0) if durs else None,
        "medrec": rnd(np.median(recov), 0) if recov else None,
        "share": rnd(share), "hist": nvalid, "chronic": share > MONTHLY_CHRONIC_SHARE,
        "trig_age": trig_age,
    }


def scan_ticker(t, df, wk, mo, spy_ret6):
    if len(df) < 30:
        return None, None
    close = df["Close"]
    px = float(close.iloc[-1])
    last_date = df.index[-1]

    wr_d = bar_wr(df.tail(N * 3))
    wr_w_bars = bar_wr(wk) if len(wk) >= N else pd.Series(dtype=float)
    wr_m_bars = bar_wr(mo) if len(mo) >= N else pd.Series(dtype=float)

    dma = close.rolling(200).mean()
    d200 = s200 = None
    if len(dma.dropna()) and np.isfinite(dma.iloc[-1]):
        d200 = (px / float(dma.iloc[-1]) - 1) * 100.0
        if len(dma.dropna()) > 21:
            s200 = bool(dma.iloc[-1] > dma.iloc[-22])
    wma = wk["Close"].rolling(200).mean() if len(wk) else pd.Series(dtype=float)
    w200 = sw200 = None
    if len(wma.dropna()) > 0 and np.isfinite(wma.iloc[-1]):
        w200 = (px / float(wma.iloc[-1]) - 1) * 100.0
        if len(wma.dropna()) > 4:
            sw200 = bool(wma.iloc[-1] > wma.iloc[-5])

    rs6 = None
    if len(close) > 126 and spy_ret6 is not None:
        rs6 = (px / float(close.iloc[-127]) - 1) * 100.0 - spy_ret6

    rnd = lambda v, d=1: (round(float(v), d) if v is not None and np.isfinite(v) else None)
    base = {
        "t": t, "px": rnd(px, 2), "d": last_date.strftime("%Y-%m-%d"),
        "wrD": rnd(wr_d.iloc[-1]) if len(wr_d) else None,
        "wrW": rnd(wr_w_bars.iloc[-1]) if len(wr_w_bars) else None,
        "wrM": rnd(wr_m_bars.iloc[-1]) if len(wr_m_bars) else None,
        "d200": rnd(d200), "s200": s200, "w200": rnd(w200), "sw200": sw200,
        "rs6": rnd(rs6),
    }

    week_row = None
    if len(wk) >= N:
        wre = period_eval_daily(df.tail(LOOKBACK_2Y + 120), "W-FRI")
        wp = weekly_primary(wk, wre, px)
        if wp is not None:
            wp["spark"] = [round(v, 1) if np.isfinite(v) else None
                            for v in wr_w_bars.tail(52).to_numpy()]
            week_row = {**base, **wp}

    month_row = None
    if len(mo) >= N:
        moe = period_eval_daily(df.tail(MONTHLY_FRESH_DAYS + MONTHLY_LEAD_DAYS), "M")
        mp = monthly_primary(mo, moe, px)
        if mp is not None:
            mp["spark"] = [round(v, 1) if np.isfinite(v) else None
                            for v in wr_m_bars.tail(60).to_numpy()]
            month_row = {**base, **mp}

    return week_row, month_row


def build_payload(rows, meta, max_date):
    for r in rows:
        m = meta.get(r["t"], {})
        r["name"] = m.get("name") or r["t"]
        r["sec"] = m.get("sector")
        r["ind"] = m.get("industry")
        r["mcap"] = m.get("mcap")
        r["exch"] = m.get("exch")
        r["stale"] = (max_date - datetime.strptime(r["d"], "%Y-%m-%d")).days > STALE_DAYS
    return {
        "asof": max_date.strftime("%Y-%m-%d") if max_date else "?",
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n": N, "universe": len(rows),
        "rows": rows,
    }


def main():
    meta = load_meta()

    daily = pd.read_parquet(DATA / "daily_window.parquet")
    weekly = pd.read_parquet(DATA / "weekly_bars.parquet")
    monthly = pd.read_parquet(DATA / "monthly_bars.parquet")

    spy_d = daily[daily["ticker"] == "SPY"].set_index("date")["Close"].sort_index()
    if len(spy_d) < 130:
        print("FATAL: SPY history missing or too short in daily_window.parquet", file=sys.stderr)
        sys.exit(1)
    spy_ret6 = (float(spy_d.iloc[-1]) / float(spy_d.iloc[-127]) - 1) * 100.0

    daily_g = {t: g.set_index("date")[["High", "Low", "Close"]].sort_index()
               for t, g in daily.groupby("ticker")}
    weekly_g = {t: g.set_index("period")[["High", "Low", "Close"]].sort_index()
                for t, g in weekly.groupby("ticker")}
    monthly_g = {t: g.set_index("period")[["High", "Low", "Close"]].sort_index()
                 for t, g in monthly.groupby("ticker")}

    tickers = sorted(daily_g.keys())
    week_rows, month_rows, max_date = [], [], None
    for i, t in enumerate(tickers):
        if t == "SPY":
            continue
        try:
            df = daily_g[t]
            wk = weekly_g.get(t, pd.DataFrame(columns=["High", "Low", "Close"]))
            mo = monthly_g.get(t, pd.DataFrame(columns=["High", "Low", "Close"]))
            wr_row, mo_row = scan_ticker(t, df, wk, mo, spy_ret6)
        except Exception as e:
            print(f"  FAIL {t}: {e}", file=sys.stderr)
            continue
        if wr_row is not None:
            week_rows.append(wr_row)
        if mo_row is not None:
            month_rows.append(mo_row)
        d = None
        if wr_row is not None:
            d = datetime.strptime(wr_row["d"], "%Y-%m-%d")
        elif mo_row is not None:
            d = datetime.strptime(mo_row["d"], "%Y-%m-%d")
        if d is not None:
            max_date = d if max_date is None or d > max_date else max_date
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(tickers)}")

    payload_w = build_payload(week_rows, meta, max_date)
    payload_m = build_payload(month_rows, meta, max_date)

    DOCS.mkdir(exist_ok=True)
    (DOCS / "screen.json").write_text(json.dumps(payload_w, separators=(",", ":")))
    (DOCS / "screen_monthly.json").write_text(json.dumps(payload_m, separators=(",", ":")))

    tpl = (ROOT / "template.html").read_text()
    html = (tpl.replace("__PAYLOAD_WEEKLY__", json.dumps(payload_w, separators=(",", ":")))
               .replace("__PAYLOAD_MONTHLY__", json.dumps(payload_m, separators=(",", ":"))))
    (DOCS / "index.html").write_text(html)

    def summarise(label, rows):
        nz = sum(1 for r in rows if r["zone"] > 0 and not r["stale"])
        fr = sum(1 for r in rows if r["zone"] > 0 and not r["stale"]
                 and r["fresh"] is not None and r["fresh"] <= 3)
        print(f"{label}: {len(rows)} tickers | in zone {nz} | fresh(<=3d) {fr}")

    summarise("WEEKLY ", week_rows)
    summarise("MONTHLY", month_rows)
    print(f"as-of {payload_w['asof']} -> {DOCS / 'index.html'}")


if __name__ == "__main__":
    main()
