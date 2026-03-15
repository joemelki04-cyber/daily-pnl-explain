# Daily PnL Explain Engine

A starter middle-office / market-risk side project that explains daily mark-to-market PnL for a simple multi-commodity energy portfolio.

## What this project does

This repo calculates daily PnL for a small energy book and splits it into:

- **Market Move PnL**
- **New Trade PnL**
- **FX PnL**
- **Unexplained PnL**

It outputs:

- trade-level explain
- desk-level summary
- control summary
- HTML report with a waterfall chart

## Why this matters

Middle-office and market-risk teams need to reconcile desk PnL movements every day and isolate whether changes came from:

- market moves
- new trades
- FX translation
- booking/data issues

This repo is designed to look like a small but realistic control/reporting workflow.

## Scope

Version 1 supports:

- end-of-day explain only
- linear products only
- Brent futures
- Henry Hub futures
- German power forwards
- one reporting currency: USD

Version 1 does **not** yet include:

- amendments/cancellations
- realized cash PnL
- options Greeks / vol explain
- intraday explain

## Folder structure

```text
daily-pnl-explain/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ raw/
│     ├─ trades.csv
│     ├─ market_prices.csv
│     └─ fx_rates.csv
├─ src/
│  ├─ valuation.py
│  └─ pnl_explain.py
└─ reports/
