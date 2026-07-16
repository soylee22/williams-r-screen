# The Oversold Ledger

Live: https://soylee22.github.io/williams-r-screen/

A weekly + monthly Williams %R(14) oversold-buy-zone screen over ~2,200
US/LSE/EU tickers, published as a static GitHub Pages dashboard and refreshed
automatically every weekday morning. Self-contained: this repo fetches its
own price data via yfinance, no dependency on any other repo.

## What it is

Two lenses on the same universe, switchable as a tab in the dashboard:

- **Weekly** -- weekly %R(14) evaluated on every daily close, trigger prices
  from the fan inversion `T = (1-p)*HH + p*LL`, freshness and chronic/episode
  stats on trading days over a trailing 2 years.
- **Monthly** -- same maths on a 14-month window, freshness stays
  day-resolution (over a trailing ~500 days here; see the honest trade-off
  note below), but chronic/episode/median-length are computed on MONTHLY
  BARS over a trailing 5 years (a single monthly episode already spans ~21
  days, so a day-based share would misread as chronic almost immediately).

Full methodology is printed on the dashboard itself (Method panel, both tabs).

## How it stays live

`.github/workflows/scan.yml` runs weekdays at 09:30 UTC (prior US session
settled by then):

1. `scripts/fetch.py` -- batched `yf.download()` top-up (a handful of large
   calls, not one call per ticker) into three small consolidated parquet
   tables: `data/daily_window.parquet` (~900 trading days), `data/weekly_bars.parquet`
   (~300 weeks), `data/monthly_bars.parquet` (~90 months). Only the CURRENT
   forming week/month bar is recomputed and upserted each run; every
   completed period's row is untouched by construction, finalised on the
   last day it was current.
2. `scripts/scan.py` -- the screen itself, writes `docs/index.html` (the
   Pages site), `docs/screen.json`, `docs/screen_monthly.json`.
3. Commits `data/` + `docs/` and pushes.

The dashboard header always shows "as of" (the data date) and a compiled
timestamp, so a stale run is visible at a glance, not silent.

## Honest trade-off vs. keeping full history

The private tool this was published from keeps unlimited per-ticker history.
This repo keeps only what the screen needs, on purpose, so daily rewrites
stay a few MB: `daily_window` at ~900 days covers every day-resolution
calculation (200dma, RS, weekly freshness/chronic, the monthly lens's live
value and triggers). The one place this bites: the monthly lens's "days in
episode" figure is capped at ~500 trading days of lookback instead of years,
so it undercounts only if a single episode has run continuously, with no
exit, for 2+ years -- a pathological case. Chronic, episode count and median
length are unaffected; they read off monthly bars, not the daily window.

## Running locally

```bash
pip install -r requirements.txt
python3 scripts/fetch.py    # top up data/
python3 scripts/scan.py     # rebuild docs/index.html
open docs/index.html
```

## Repo layout

```
scripts/fetch.py     daily price top-up
scripts/scan.py       the screen (weekly + monthly), renders docs/
template.html          dashboard template (embeds both payloads)
meta/tickers.csv       universe: ticker,name,sector,industry,mcap,exch
data/*.parquet          the three consolidated price tables (committed)
docs/index.html         published dashboard (Pages source)
```

## Known limitation

`meta/tickers.csv` (name/sector/market cap) is a one-time seed, not
refreshed by the daily workflow. Sector/name/mcap drift slowly; re-seed
manually if a name looks stale.

## Read before buying

Not a signal. Correlated firing in a correction, oversold-gets-more-oversold
above a rising trend, and post-earnings gaps as information rather than
noise are all printed as caveats on the dashboard itself. This surfaces
candidates; it is not a backtested edge.
