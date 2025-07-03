"""
Stock Market and Sector Residual Analysis

This script downloads ETF data, calculates residuals using point-in-time robust regression,
and plots cumulative returns.
"""

import os
import sys
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_market_calendars as mcal
from vbase_utils.stats.pit_robust_betas import pit_robust_betas

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define constants
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}
MARKET = "SPY"
ALL_TICKERS = [MARKET] + list(SECTOR_ETFS.keys())


def download_full_etf_history(
    tickers: list[str],
    output_path: str = "data/backtest/etf_full_price_history.csv"
):
    all_data = []
    for ticker in tickers:
        attempt = 0
        while attempt < 5:
            try:
                logger.info(f"Downloading history for {ticker}")
                df = yf.download(ticker, period="max", auto_adjust=False)
                if df.empty or "Close" not in df.columns:
                    raise ValueError(f"No data returned for {ticker}")
                df = df[["Close"]].rename(columns={"Close": ticker})
                all_data.append(df)
                break
            except Exception as e:
                logger.warning(f"⏳ Attempt {attempt+1} failed for {ticker}: {e}")
                attempt += 1
                time.sleep(5 * attempt)  # Exponential backoff
        else:
            logger.error(f"❌ Failed to download data for {ticker} after 5 attempts")

    if not all_data:
        raise ValueError("No data downloaded for any ticker.")
    combined = pd.concat(all_data, axis=1, join="inner")
    combined.index.name = "Date"
    combined.columns = combined.columns.droplevel('Price')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path)
    logger.info(f"✅ Saved historical prices to {output_path}")


def plot_cumulative_residuals(
    residuals: pd.DataFrame, title: str, figsize: Tuple[int, int] = (15, 8)
) -> None:
    """Plot cumulative residuals for a set of assets."""
    cum_residuals = (1 + residuals).cumprod() - 1

    plt.figure(figsize=figsize)
    for col in cum_residuals.columns:
        plt.plot(cum_residuals.index, cum_residuals[col], label=col)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Residual Return")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_sector_backtest_from_csv(csv_path: str, output_dir: str = "data/backtest") -> None:
    """
    Run point-in-time regression backtest on sector ETFs using a CSV file of historical prices.
    """
    logger.info(f"📥 Loading data from {csv_path}")
    df_prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df_prices = df_prices[ALL_TICKERS].dropna()

    # Filter for valid NYSE trading days
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=df_prices.index.min(), end_date=df_prices.index.max())
    trading_days = pd.to_datetime(schedule.index)
    df_prices = df_prices[df_prices.index.isin(trading_days)]

    # Compute returns
    df_rets = df_prices.pct_change().dropna()
    Y = df_rets[list(SECTOR_ETFS.keys())]
    X = df_rets[[MARKET]]
    weekly_rebalance = pd.DatetimeIndex([d for d in Y.index if d.weekday() == 4])

    logger.info("📊 Running pit_robust_betas...")
    results = pit_robust_betas(
        df_asset_rets=Y,
        df_fact_rets=X,
        half_life=126,
        min_timestamps=63,
        rebalance_time_index=weekly_rebalance,
        progress=True
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results["df_betas"].to_csv(Path(output_dir) / "sector_betas.csv")
    results["df_hedge_rets"].to_csv(Path(output_dir) / "hedge_returns.csv")
    results["df_asset_resids"].to_csv(Path(output_dir) / "residual_returns.csv")

    logger.info(f"✅ Backtest results saved to {output_dir}")


if __name__ == "__main__":
    csv_output = "data/backtest/etf_full_price_history.csv"

    # Step 1: Download and save full ETF history
    download_full_etf_history(ALL_TICKERS, output_path=csv_output)

    # Step 2: Run sector backtest on downloaded history
    run_sector_backtest_from_csv(csv_output)
