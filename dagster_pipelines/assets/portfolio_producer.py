"""
This module contains the logic for producing portfolio positions using exponentially weighted robust regressions.
"""
import os
import time
import logging
import pickle
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import HuberT

from .robust_regression import compute_time_weighted_robust_betas


def get_run_logger(partition_date: str) -> logging.Logger:
    """Create a new file-based logger for each run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "run_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"sector_portfolios_run_{partition_date}_{timestamp}.log")

    logger = logging.getLogger(f"sector_portfolios_logger_{partition_date}_{timestamp}")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def download_ticker_with_smart_cache(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str = 'data/cache',
    force_refresh: bool = False
) -> pd.DataFrame:
    """Download ticker data with caching."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}.pkl")

    if os.path.exists(cache_file) and not force_refresh:
        with open(cache_file, 'rb') as f:
            cached_data: pd.DataFrame = pickle.load(f)
        cached_data = cached_data[~cached_data.index.duplicated(keep='last')]
        cached_start, cached_end = cached_data.index.min(), cached_data.index.max()

        fetch_start = min(pd.to_datetime(start), cached_start)
        fetch_end = max(pd.to_datetime(end), cached_end)

        if fetch_start < cached_start or fetch_end > cached_end:
            print(f"Extending cached data for {ticker}")
            new_data = yf.download(
                ticker,
                start=fetch_start.strftime('%Y-%m-%d'),
                end=fetch_end.strftime('%Y-%m-%d'),
                auto_adjust=False
            )
            combined = pd.concat([cached_data, new_data])
            combined = combined[~combined.index.duplicated(keep='last')].sort_index()
            with open(cache_file, 'wb') as f:
                pickle.dump(combined, f)
            return combined.loc[start:end]
        else:
            print(f"Using cached data for {ticker}")
            return cached_data.loc[start:end]

    print(f"Downloading fresh data for {ticker}")
    data: pd.DataFrame = yf.download(ticker, start=start, end=end, auto_adjust=False)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    return data


def produce_sector_portfolios(
    portfolio_date: str,
    logger: logging.Logger,
    half_life: float = 21
) -> pd.DataFrame:
    """
    Generate market-neutral long and short portfolios for 11 sector ETFs hedged against SPY
    using exponentially weighted robust regression.

    Args:
        portfolio_date: The date for which to generate the portfolios.
        logger: Logger object to track events.
        half_life: Half-life in days for exponential weighting (default: 21).

    Returns:
        A DataFrame with portfolio positions.
    """
    schedule = mcal.get_calendar("NYSE").schedule(start_date=portfolio_date, end_date=portfolio_date)
    if schedule.empty:
        logger.warning(f"No trading on {portfolio_date}.")
        raise ValueError(f"No trading on {portfolio_date}.")

    etf_tickers: list[str] = [
        'XLK', 'XLF', 'XLV', 'XLY', 'XLP',
        'XLE', 'XLI', 'XLB', 'XLU', 'XLC', 'XLRE'
    ]
    tickers: list[str] = etf_tickers + ['SPY']

    end_date = pd.to_datetime(portfolio_date)
    start_date = end_date - pd.Timedelta(days=90)

    all_data: list[pd.DataFrame] = []
    for ticker in tickers:
        try:
            data = download_ticker_with_smart_cache(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if 'Close' not in data.columns:
                print(f"'Close' not found for {ticker}. Skipping.")
                continue
            df = data[['Close']].rename(columns={'Close': ticker})
            all_data.append(df)
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

    if not all_data:
        raise ValueError("No data was downloaded for the given tickers.")

    merged_data = pd.concat(all_data, axis=1, join='inner')
    returns = merged_data.pct_change().dropna()
    if returns.empty:
        raise ValueError("No return data available after dropping NaNs.")

    Y = returns[etf_tickers]
    X = returns[['SPY']]
    betas: dict[str, float] = compute_time_weighted_robust_betas(Y, X, half_life=half_life)

    all_positions: list[dict[str, object]] = []
    for etf in etf_tickers:
        beta = betas.get('SPY', np.nan)
        if np.isnan(beta):
            logger.warning(f"Missing beta for {etf}. Skipping.")
            continue

        all_positions.append({"portfolio_name": f"{etf}_long", "sym": etf, "wt": 1})
        all_positions.append({"portfolio_name": f"{etf}_long", "sym": "SPY", "wt": -beta})

        all_positions.append({"portfolio_name": f"{etf}_short", "sym": etf, "wt": -1})
        all_positions.append({"portfolio_name": f"{etf}_short", "sym": "SPY", "wt": beta})

        logger.info(f"Portfolio {etf}: beta = {beta:.4f}")

    position_df = pd.DataFrame(all_positions)
    return position_df
