from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_inputs(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)

    trades = pd.read_csv(data_dir / "trades.csv")
    market_prices = pd.read_csv(data_dir / "market_prices.csv")
    fx_rates = pd.read_csv(data_dir / "fx_rates.csv")

    return trades, market_prices, fx_rates


def prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades.copy()

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    df["status"] = df["status"].astype(str).str.upper().str.strip()
    df["side"] = df["side"].astype(str).str.upper().str.strip()
    df["commodity"] = df["commodity"].astype(str).str.strip()
    df["contract_month"] = df["contract_month"].astype(str).str.strip()
    df["currency"] = df["currency"].astype(str).str.upper().str.strip()

    if not df["side"].isin(["BUY", "SELL"]).all():
        bad = df.loc[~df["side"].isin(["BUY", "SELL"]), ["trade_id", "side"]]
        raise ValueError(f"Invalid side values found:\n{bad.to_string(index=False)}")

    df["signed_quantity"] = df["quantity"].where(df["side"] == "BUY", -df["quantity"])

    return df


def prepare_market_prices(market_prices: pd.DataFrame) -> pd.DataFrame:
    df = market_prices.copy()

    df["valuation_date"] = pd.to_datetime(df["valuation_date"]).dt.normalize()
    df["commodity"] = df["commodity"].astype(str).str.strip()
    df["contract_month"] = df["contract_month"].astype(str).str.strip()
    df["currency"] = df["currency"].astype(str).str.upper().str.strip()

    return df


def prepare_fx_rates(fx_rates: pd.DataFrame) -> pd.DataFrame:
    df = fx_rates.copy()

    df["valuation_date"] = pd.to_datetime(df["valuation_date"]).dt.normalize()
    df["currency"] = df["currency"].astype(str).str.upper().str.strip()

    return df


def get_previous_available_date(market_prices: pd.DataFrame, valuation_date: str | pd.Timestamp) -> pd.Timestamp:
    valuation_date = pd.Timestamp(valuation_date).normalize()
    available_dates = sorted(pd.to_datetime(market_prices["valuation_date"]).dt.normalize().unique())

    earlier_dates = [pd.Timestamp(d) for d in available_dates if pd.Timestamp(d) < valuation_date]
    if not earlier_dates:
        raise ValueError(f"No market date found before {valuation_date.date()}")

    return earlier_dates[-1]


def value_trades(
    trades: pd.DataFrame,
    market_prices: pd.DataFrame,
    fx_rates: pd.DataFrame,
    valuation_date: str | pd.Timestamp,
) -> pd.DataFrame:
    valuation_date = pd.Timestamp(valuation_date).normalize()

    active_trades = trades.loc[
        (trades["status"] == "ACTIVE") & (trades["trade_date"] <= valuation_date)
    ].copy()

    market_slice = market_prices.loc[
        market_prices["valuation_date"] == valuation_date,
        ["commodity", "contract_month", "settlement_price", "currency"],
    ].rename(columns={"currency": "market_currency"})

    fx_slice = fx_rates.loc[
        fx_rates["valuation_date"] == valuation_date,
        ["currency", "usd_fx"],
    ]

    if active_trades.empty:
        return pd.DataFrame()

    valued = active_trades.merge(
        market_slice,
        on=["commodity", "contract_month"],
        how="left",
        validate="many_to_one",
    )

    missing_prices = valued.loc[
        valued["settlement_price"].isna(),
        ["trade_id", "commodity", "contract_month"],
    ]
    if not missing_prices.empty:
        raise ValueError(
            "Missing market prices for:\n"
            + missing_prices.to_string(index=False)
        )

    currency_mismatch = valued.loc[
        valued["currency"] != valued["market_currency"],
        ["trade_id", "currency", "market_currency"],
    ]
    if not currency_mismatch.empty:
        raise ValueError(
            "Trade currency does not match market price currency for:\n"
            + currency_mismatch.to_string(index=False)
        )

    valued = valued.merge(
        fx_slice,
        on="currency",
        how="left",
        validate="many_to_one",
    )

    missing_fx = valued.loc[
        valued["usd_fx"].isna(),
        ["trade_id", "currency"],
    ]
    if not missing_fx.empty:
        raise ValueError(
            "Missing FX rates for:\n"
            + missing_fx.to_string(index=False)
        )

    valued["mtm_local"] = valued["signed_quantity"] * (valued["settlement_price"] - valued["trade_price"])
    valued["mtm_usd"] = valued["mtm_local"] * valued["usd_fx"]
    valued["valuation_date"] = valuation_date

    ordered_cols = [
        "valuation_date",
        "trade_id",
        "book",
        "commodity",
        "instrument_type",
        "contract_month",
        "side",
        "quantity",
        "signed_quantity",
        "unit",
        "trade_price",
        "trade_date",
        "status",
        "counterparty",
        "currency",
        "settlement_price",
        "usd_fx",
        "mtm_local",
        "mtm_usd",
    ]

    return valued[ordered_cols].copy()
