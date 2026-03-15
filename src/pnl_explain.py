from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from valuation import (
    load_inputs,
    prepare_trades,
    prepare_market_prices,
    prepare_fx_rates,
    get_previous_available_date,
    value_trades,
)


def _coalesce(df: pd.DataFrame, col: str) -> pd.Series:
    return df[f"{col}_today"].combine_first(df[f"{col}_prev"])


def build_explain(
    data_dir: str | Path,
    valuation_date: str,
    prev_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    trades, market_prices, fx_rates = load_inputs(data_dir)

    trades = prepare_trades(trades)
    market_prices = prepare_market_prices(market_prices)
    fx_rates = prepare_fx_rates(fx_rates)

    valuation_date = pd.Timestamp(valuation_date).normalize()
    prev_date = (
        pd.Timestamp(prev_date).normalize()
        if prev_date is not None
        else get_previous_available_date(market_prices, valuation_date)
    )

    prev_vals = value_trades(trades, market_prices, fx_rates, prev_date)
    today_vals = value_trades(trades, market_prices, fx_rates, valuation_date)

    rename_prev = {
        "book": "book_prev",
        "commodity": "commodity_prev",
        "instrument_type": "instrument_type_prev",
        "contract_month": "contract_month_prev",
        "side": "side_prev",
        "quantity": "quantity_prev",
        "signed_quantity": "signed_quantity_prev",
        "unit": "unit_prev",
        "trade_price": "trade_price_prev",
        "trade_date": "trade_date_prev",
        "status": "status_prev",
        "counterparty": "counterparty_prev",
        "currency": "currency_prev",
        "settlement_price": "settlement_price_prev",
        "usd_fx": "usd_fx_prev",
        "mtm_local": "mtm_local_prev",
        "mtm_usd": "mtm_usd_prev",
    }
    rename_today = {
        "book": "book_today",
        "commodity": "commodity_today",
        "instrument_type": "instrument_type_today",
        "contract_month": "contract_month_today",
        "side": "side_today",
        "quantity": "quantity_today",
        "signed_quantity": "signed_quantity_today",
        "unit": "unit_today",
        "trade_price": "trade_price_today",
        "trade_date": "trade_date_today",
        "status": "status_today",
        "counterparty": "counterparty_today",
        "currency": "currency_today",
        "settlement_price": "settlement_price_today",
        "usd_fx": "usd_fx_today",
        "mtm_local": "mtm_local_today",
        "mtm_usd": "mtm_usd_today",
    }

    prev_vals = prev_vals.rename(columns=rename_prev)
    today_vals = today_vals.rename(columns=rename_today)

    merged = prev_vals.merge(today_vals, on="trade_id", how="outer")

    meta_cols = [
        "book",
        "commodity",
        "instrument_type",
        "contract_month",
        "side",
        "unit",
        "counterparty",
        "currency",
    ]
    for col in meta_cols:
        merged[col] = _coalesce(merged, col)

    # canonical numeric fields
    merged["quantity"] = merged["quantity_today"].combine_first(merged["quantity_prev"])
    merged["signed_quantity"] = merged["signed_quantity_today"].combine_first(merged["signed_quantity_prev"])
    merged["trade_price"] = merged["trade_price_today"].combine_first(merged["trade_price_prev"])
    merged["trade_date"] = pd.to_datetime(
        merged["trade_date_today"].combine_first(merged["trade_date_prev"])
    ).dt.normalize()

    exists_prev = merged["mtm_usd_prev"].notna()
    exists_today = merged["mtm_usd_today"].notna()

    legacy_mask = exists_prev & exists_today
    new_trade_mask = ~exists_prev & exists_today
    missing_today_mask = exists_prev & ~exists_today

    merged["actual_pnl"] = merged["mtm_usd_today"].fillna(0.0) - merged["mtm_usd_prev"].fillna(0.0)
    merged["market_move_pnl"] = 0.0
    merged["new_trade_pnl"] = 0.0
    merged["fx_pnl"] = 0.0

    # Legacy trades:
    # market_move_pnl = yesterday position * today's-vs-yesterday's market move, translated at yesterday FX
    merged.loc[legacy_mask, "market_move_pnl"] = (
        merged.loc[legacy_mask, "signed_quantity"]
        * (
            merged.loc[legacy_mask, "settlement_price_today"]
            - merged.loc[legacy_mask, "settlement_price_prev"]
        )
        * merged.loc[legacy_mask, "usd_fx_prev"]
    )

    # FX bucket using today's local MtM exposed to the FX change
    merged.loc[legacy_mask, "fx_pnl"] = (
        merged.loc[legacy_mask, "mtm_local_today"]
        * (
            merged.loc[legacy_mask, "usd_fx_today"]
            - merged.loc[legacy_mask, "usd_fx_prev"]
        )
    )

    # New trades booked today
    merged.loc[new_trade_mask, "new_trade_pnl"] = merged.loc[new_trade_mask, "mtm_usd_today"]

    merged["total_explained_pnl"] = (
        merged["market_move_pnl"] + merged["new_trade_pnl"] + merged["fx_pnl"]
    )
    merged["unexplained_pnl"] = merged["actual_pnl"] - merged["total_explained_pnl"]

    merged["explain_type"] = "LEGACY"
    merged.loc[new_trade_mask, "explain_type"] = "NEW_TRADE"
    merged.loc[missing_today_mask, "explain_type"] = "MISSING_TODAY"

    explain_cols = [
        "trade_id",
        "book",
        "commodity",
        "instrument_type",
        "contract_month",
        "side",
        "quantity",
        "unit",
        "currency",
        "trade_date",
        "trade_price",
        "settlement_price_prev",
        "settlement_price_today",
        "usd_fx_prev",
        "usd_fx_today",
        "mtm_usd_prev",
        "mtm_usd_today",
        "actual_pnl",
        "market_move_pnl",
        "new_trade_pnl",
        "fx_pnl",
        "total_explained_pnl",
        "unexplained_pnl",
        "explain_type",
    ]
    explain_df = merged[explain_cols].copy()
    explain_df.insert(0, "prev_valuation_date", prev_date)
    explain_df.insert(1, "valuation_date", valuation_date)

    desk_summary = (
        explain_df.groupby(["valuation_date", "book", "commodity"], dropna=False)[
            ["actual_pnl", "market_move_pnl", "new_trade_pnl", "fx_pnl", "total_explained_pnl", "unexplained_pnl"]
        ]
        .sum()
        .reset_index()
        .sort_values(["book", "commodity"])
    )

    totals = explain_df[
        ["actual_pnl", "market_move_pnl", "new_trade_pnl", "fx_pnl", "total_explained_pnl", "unexplained_pnl"]
    ].sum()

    total_actual = float(totals["actual_pnl"])
    total_unexplained = float(totals["unexplained_pnl"])

    allowed_unexplained = max(5000.0, 0.02 * abs(total_actual))
    explained_pct = 100.0 if abs(total_actual) < 1e-12 else 100.0 * (
        1.0 - abs(total_unexplained) / abs(total_actual)
    )

    control_summary = pd.DataFrame(
        [
            {
                "prev_valuation_date": prev_date,
                "valuation_date": valuation_date,
                "total_actual_pnl": total_actual,
                "total_market_move_pnl": float(totals["market_move_pnl"]),
                "total_new_trade_pnl": float(totals["new_trade_pnl"]),
                "total_fx_pnl": float(totals["fx_pnl"]),
                "total_explained_pnl": float(totals["total_explained_pnl"]),
                "total_unexplained_pnl": total_unexplained,
                "explained_pct": explained_pct,
                "allowed_unexplained_abs": allowed_unexplained,
                "control_status": "PASS" if abs(total_unexplained) <= allowed_unexplained else "BREACH",
            }
        ]
    )

    return explain_df, desk_summary, control_summary, valuation_date, prev_date


def build_waterfall(control_row: pd.Series) -> go.Figure:
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "total"],
            x=["Market Move", "New Trades", "FX", "Unexplained", "Actual Total"],
            y=[
                control_row["total_market_move_pnl"],
                control_row["total_new_trade_pnl"],
                control_row["total_fx_pnl"],
                control_row["total_unexplained_pnl"],
                control_row["total_actual_pnl"],
            ],
            text=[
                f'{control_row["total_market_move_pnl"]:,.2f}',
                f'{control_row["total_new_trade_pnl"]:,.2f}',
                f'{control_row["total_fx_pnl"]:,.2f}',
                f'{control_row["total_unexplained_pnl"]:,.2f}',
                f'{control_row["total_actual_pnl"]:,.2f}',
            ],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Daily PnL Explain Waterfall (USD)",
        showlegend=False,
        template="plotly_white",
        yaxis_title="USD",
    )
    return fig


def write_outputs(
    explain_df: pd.DataFrame,
    desk_summary: pd.DataFrame,
    control_summary: pd.DataFrame,
    output_dir: str | Path,
    valuation_date: pd.Timestamp,
    prev_date: pd.Timestamp,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explain_path = output_dir / f"daily_pnl_explain_{valuation_date.date()}.csv"
    desk_path = output_dir / f"desk_summary_{valuation_date.date()}.csv"
    control_path = output_dir / f"control_summary_{valuation_date.date()}.csv"
    html_path = output_dir / f"daily_pnl_report_{valuation_date.date()}.html"

    explain_df.to_csv(explain_path, index=False)
    desk_summary.to_csv(desk_path, index=False)
    control_summary.to_csv(control_path, index=False)

    control_row = control_summary.iloc[0]
    fig = build_waterfall(control_row)
    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    top_contributors = (
        explain_df.assign(abs_actual=explain_df["actual_pnl"].abs())
        .sort_values("abs_actual", ascending=False)
        .drop(columns="abs_actual")
        .head(10)
    )

    unexplained_exceptions = explain_df.loc[explain_df["unexplained_pnl"].abs() > 0.01].copy()
    if unexplained_exceptions.empty:
        unexplained_exceptions = pd.DataFrame(
            [{"message": "No unexplained PnL exceptions above 0.01 USD"}]
        )

    fmt = lambda x: f"{x:,.2f}"

    html = f"""
    <html>
      <head>
        <title>Daily PnL Explain Report - {valuation_date.date()}</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 32px; }}
          h1, h2 {{ margin-top: 28px; }}
          table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
          th {{ background-color: #f5f5f5; }}
          .meta {{ margin-bottom: 24px; }}
        </style>
      </head>
      <body>
        <h1>Daily PnL Explain Report</h1>
        <div class="meta">
          <p><strong>Previous valuation date:</strong> {prev_date.date()}</p>
          <p><strong>Current valuation date:</strong> {valuation_date.date()}</p>
          <p><strong>Control status:</strong> {control_row["control_status"]}</p>
        </div>

        {fig_html}

        <h2>Control Summary</h2>
        {control_summary.to_html(index=False, float_format=fmt)}

        <h2>Desk Summary</h2>
        {desk_summary.to_html(index=False, float_format=fmt)}

        <h2>Top Trade Contributors</h2>
        {top_contributors.to_html(index=False, float_format=fmt)}

        <h2>Unexplained PnL Exceptions</h2>
        {unexplained_exceptions.to_html(index=False, float_format=fmt)}
      </body>
    </html>
    """

    html_path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily PnL Explain Engine")
    parser.add_argument("--date", required=True, help="Current valuation date, e.g. 2026-03-15")
    parser.add_argument("--prev-date", required=False, help="Previous valuation date, optional")
    parser.add_argument("--data-dir", default="data/raw", help="Folder containing CSV inputs")
    parser.add_argument("--output-dir", default="reports", help="Folder for CSV/HTML outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    explain_df, desk_summary, control_summary, valuation_date, prev_date = build_explain(
        data_dir=args.data_dir,
        valuation_date=args.date,
        prev_date=args.prev_date,
    )

    write_outputs(
        explain_df=explain_df,
        desk_summary=desk_summary,
        control_summary=control_summary,
        output_dir=args.output_dir,
        valuation_date=valuation_date,
        prev_date=prev_date,
    )

    print("\nDaily PnL Explain completed.")
    print(f"Previous valuation date: {prev_date.date()}")
    print(f"Current valuation date:  {valuation_date.date()}")
    print(control_summary.to_string(index=False))


if __name__ == "__main__":
    main()
