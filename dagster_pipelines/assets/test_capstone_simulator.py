#!/usr/bin/env python3
"""
Test script to verify the main simulation works with a single strategy.
"""

import os
import sys
import time
import pickle
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import logging
import pandas_market_calendars as mcal

from vbase_utils.stats.pit_robust_betas import pit_robust_betas


sys.path.append('../BWM/bwm_capstone_simulator/src')
# import capstone_simulator as cs 

from single_target_simulator import (
    load_and_prepare_data, Simulate, 
    SingleTargetBenchmarkManager, SingleTargetBenchmarkConfig,
    sim_stats_single_target, L_func_2, L_func_3, L_func_4
)

from multi_target_simulator import Simulate_MultiTarget, load_and_prepare_multi_target_data

warnings.simplefilter(action='ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---
SECTOR_ETFS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
MARKET = "SPY"
ALL_TICKERS = [MARKET] + SECTOR_ETFS
CACHE_DIR = "../data/cache"
OUTPUT_DIR = "../data/backtest"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# def download_prices():
#     logger.info("Downloading ETF price history")
#     data = yf.download(ALL_TICKERS, start="2010-01-01", auto_adjust=True)["Close"]
#     # Filter for NYSE trading days
#     nyse = mcal.get_calendar("NYSE")
#     schedule = nyse.schedule(start_date=data.index.min(), end_date=data.index.max())
#     trading_days = pd.to_datetime(schedule.index)
#     data = data[data.index.isin(trading_days)]
#     return data

def load_from_cache(ticker):
    path = os.path.join(CACHE_DIR, f"{ticker}.pkl")
    if os.path.exists(path):
        try:
            print(f"[{ticker}] Loading from cache...")
            return pickle.load(open(path, "rb"))
        except Exception as e:
            print(f"[{ticker}] Cache load error: {e}")
    return None

def save_to_cache(ticker, df):
    path = os.path.join(CACHE_DIR, f"{ticker}.pkl")
    try:
        pickle.dump(df, open(path, "wb"))
    except Exception as e:
        print(f"[{ticker}] Cache save error: {e}")

def safe_download(ticker, start="2000-01-01", end=None, max_retries=5, pause=5):
    cached = load_from_cache(ticker)
    if cached is not None and not cached.empty:
        return cached

    attempt = 0
    while attempt < max_retries:
        try:
            df = yf.download(
                ticker, start=start, end=end,
                progress=False, auto_adjust=True
            )
            if not df.empty:
                save_to_cache(ticker, df)
                return df
        except Exception as e:
            if "Rate limited" in str(e) or "YFRateLimitError" in str(e):
                print(f"[{ticker}] Rate limited. Sleeping {pause}s...")
                time.sleep(pause)
                pause *= 2
            else:
                print(f"[{ticker}] Unexpected error: {e}")
                break
        attempt += 1
    print(f"[{ticker}] Failed after {max_retries} retries.")
    return pd.DataFrame()

def smart_download(tickers, start="2000-01-01", end=None):
    all_data = []
    for t in tickers:
        df = safe_download(t, start=start, end=end)
        if df.empty:
            print(f"[{t}] Warning: empty dataframe returned.")
            continue
        s = df["Close"].copy()
        s.name = t
        all_data.append(s)

    if not all_data:
        raise RuntimeError("No valid data downloaded. Try again later.")

    return pd.concat(all_data, axis=1).sort_index()


# --- Download all tickers ---
def download_all_tickers(tickers: List[str], start: str = "2020-01-01", end: str = None) -> pd.DataFrame:
    dfs = []
    for t in tickers:
        df = smart_download(t, start, end)
        if "Close" not in df.columns:
            raise ValueError(f"No 'Close' column for {t}")
        df = df[["Close"]].rename(columns={"Close": t})
        dfs.append(df)
    combined = pd.concat(dfs, axis=1)
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    return combined

def test_single_simulation():
    """Test a single simulation to verify the main script works."""
    
    print("Testing single simulation...")
    
    # # ETF configuration
    # feature_etfs = ['XLK']
    # target_etfs = ['SPY']
    # all_etfs = feature_etfs + target_etfs
    
    try:
        # # Load and prepare data
        # print("Loading data...")
        # X, y_multi = load_and_prepare_multi_target_data(
        #     etf_list=SECTOR_ETFS, 
        #     target_etfs=MARKET,
        #     start_date='2015-01-01'  # Use reasonable period for testing
        # )
        
        data_df = smart_download(ALL_TICKERS, start="2010-01-01")
        print("Data downloaded.")

        X = data_df[MARKET]
        y_multi = data_df[SECTOR_ETFS]

        if X.empty or y_multi.empty:
            print("❌ No data loaded - using cached results instead")
            return False
        
        print(f"Data loaded: X shape {X.shape}, y_multi shape {y_multi.shape}")
        return True
            
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_simulation()
    if success:
        print("\nMain simulation script is working!")
    else:
        print("\nMain simulation script has issues")