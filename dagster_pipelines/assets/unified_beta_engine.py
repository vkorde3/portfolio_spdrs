"""
Faster unified beta engine pipeline with three options:

1) engine="robust"
     -> fast_pit_robust_betas (rolling window, skips bad timestamps)

2) engine="kalman"
     -> pit_kalman_betas

3) engine="robust_then_kalman"
     -> fast_pit_robust_betas for initial betas, then pit_kalman_betas
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from vbase_utils.stats.robust_betas import robust_betas
from vbase_utils.stats.pit_kalman_betas import pit_kalman_betas

# ---------------- User configuration ----------------
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

STOCK_SECTOR_MAP = {
    "MSFT": "XLK",
    "AAPL": "XLK",
    "XOM": "XLE",
    "JPM": "XLF",
    "JNJ": "XLV",
    "CAT": "XLI",
    "PG": "XLP",
    "AMZN": "XLY",
    "ECL": "XLB",
    "NEE": "XLU",
    "PLD": "XLRE",
    "GOOGL": "XLC",
}

MARKET = "SPY"
STOCK_RETS_PATH = "data/backtest/us_stocks_1d_rets.csv"

OUTPUT_DIR = "output_unified_engine_fast"
BETAS_CSV = os.path.join(OUTPUT_DIR, "betas.csv")
HEDGE_CSV = os.path.join(OUTPUT_DIR, "hedge_rets.csv")
RESIDS_CSV = os.path.join(OUTPUT_DIR, "residuals.csv")
HEDGE_PLOT = os.path.join(OUTPUT_DIR, "hedge_ratios.png")
# ----------------------------------------------------


# ---------------- Helper: load & clean ----------------
def load_returns(path: str) -> pd.DataFrame:
    path = os.path.expanduser(path)
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
    df = df[~df.index.isna()].sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all")
    return df


def clean_pair(
    df_asset: pd.DataFrame,
    df_fact: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_asset = df_asset.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    df_fact = df_fact.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    idx = (~df_asset.isna().all(axis=1)) & (~df_fact.isna().all(axis=1))
    df_asset = df_asset.loc[idx]
    df_fact = df_fact.loc[idx]
    return df_asset, df_fact


# ---------------- Faster robust PIT (rolling window) ----------------
def fast_pit_robust_betas(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    rebalance_time_index: pd.DatetimeIndex,
    half_life: Optional[float] = 60,
    lambda_: Optional[float] = 0.97,
    min_timestamps: int = 60,
    lookback_days: int = 252,
    progress: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    PIT robust betas using robust_betas on a rolling window:

    - For each rebalance date t:
      * Use last `lookback_days` of data up to t (intersection with min_timestamps).
    - Skips bad timestamps.
    - Returns same keys as pit_robust_betas.
    """
    if df_asset_rets.empty or df_fact_rets.empty:
        raise ValueError("Input DataFrames cannot be empty")

    df_asset_rets = df_asset_rets.sort_index()
    df_fact_rets = df_fact_rets.sort_index()

    if not df_asset_rets.index.equals(df_fact_rets.index):
        common_idx = df_asset_rets.index.intersection(df_fact_rets.index)
        df_asset_rets = df_asset_rets.loc[common_idx]
        df_fact_rets = df_fact_rets.loc[common_idx]

    factors = list(df_fact_rets.columns)
    assets = list(df_asset_rets.columns)

    betas_list: List[pd.DataFrame] = []

    dates = df_asset_rets.index
    iterator = tqdm(rebalance_time_index, desc="PIT robust (fast)", unit="t") if progress else rebalance_time_index

    for t in iterator:
        if t not in dates:
            continue

        # rolling window: use last lookback_days worth of dates
        end_pos = dates.get_loc(t)
        start_pos = max(0, end_pos - lookback_days + 1)
        window_idx = dates[start_pos : end_pos + 1]

        if len(window_idx) < min_timestamps:
            continue

        y_hist = df_asset_rets.loc[window_idx]
        x_hist = df_fact_rets.loc[window_idx]

        y_hist, x_hist = clean_pair(y_hist, x_hist)
        if y_hist.shape[0] < min_timestamps:
            continue

        try:
            beta_matrix = robust_betas(
                df_asset_rets=y_hist,
                df_fact_rets=x_hist,
                half_life=half_life,
                lambda_=lambda_,
            )  # [factor x asset]
        except Exception:
            continue

        beta_df = beta_matrix.copy()
        beta_df["timestamp"] = t
        beta_df = beta_df.set_index("timestamp", append=True)
        beta_df = beta_df.reorder_levels(["timestamp", beta_df.index.names[0]])
        betas_list.append(beta_df)

    if not betas_list:
        raise ValueError("No valid robust beta estimates produced; check data and parameters")

    df_betas = pd.concat(betas_list).sort_index()
    df_betas.index.set_names(["timestamp", "factor"], inplace=True)

    # hedge weights and residuals
    df_hedge_weights = -1.0 * df_betas.shift(1)

    df_fact_stacked = df_fact_rets.stack().to_frame("ret")
    df_fact_stacked.index.names = ["timestamp", "factor"]

    df_hedge_rets_by_fact = df_hedge_weights.multiply(df_fact_stacked["ret"], axis=0)
    df_hedge_rets = df_hedge_rets_by_fact.groupby("timestamp").sum(min_count=1)
    df_hedge_rets = df_hedge_rets.reindex(df_asset_rets.index).fillna(0.0)

    df_asset_resids = df_asset_rets + df_hedge_rets
    if df_asset_resids.index.name is None:
        df_asset_resids.index.name = "timestamp"

    return {
        "df_betas": df_betas,
        "df_hedge_rets_by_fact": df_hedge_rets_by_fact,
        "df_hedge_rets": df_hedge_rets,
        "df_asset_resids": df_asset_resids,
    }


# ---------------- Unified dispatcher ----------------
def pit_betas_with_strategy(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    engine: str = "robust",
    rebalance_time_index: Optional[pd.DatetimeIndex] = None,
    min_timestamps: int = 60,
    progress: bool = True,
    # robust params
    half_life: Optional[float] = 60,
    lambda_: Optional[float] = 0.97,
    lookback_days: int = 252,
    # kalman params
    base_q: float = 0.01,
    base_r: float = 1.0,
    vix: Optional[pd.Series] = None,
    liquidity: Optional[Dict[str, pd.Series]] = None,
    vix_k: float = 0.10,
    vix_v0: float = 15.0,
    liquidity_multiplier: float = 1.0,
    outlier_clip_std: float = 3.0,
    initial_betas_override: Optional[pd.DataFrame] = None,
    chunk_size: int = 500,
) -> Dict[str, pd.DataFrame]:
    if df_asset_rets.empty or df_fact_rets.empty:
        raise ValueError("Input DataFrames cannot be empty")

    if not isinstance(df_asset_rets.index, pd.DatetimeIndex):
        raise ValueError("df_asset_rets must have a DatetimeIndex")
    if not isinstance(df_fact_rets.index, pd.DatetimeIndex):
        raise ValueError("df_fact_rets must have a DatetimeIndex")

    common_idx = df_asset_rets.index.intersection(df_fact_rets.index)
    df_asset_rets = df_asset_rets.loc[common_idx]
    df_fact_rets = df_fact_rets.loc[common_idx]
    df_asset_rets, df_fact_rets = clean_pair(df_asset_rets, df_fact_rets)

    if rebalance_time_index is None:
        # coarser grid speeds things up a lot
        rebalance_time_index = df_asset_rets.resample("ME").last().index

    engine = engine.lower()

    if engine == "robust":
        return fast_pit_robust_betas(
            df_asset_rets=df_asset_rets,
            df_fact_rets=df_fact_rets,
            rebalance_time_index=rebalance_time_index,
            half_life=half_life,
            lambda_=lambda_,
            min_timestamps=min_timestamps,
            lookback_days=lookback_days,
            progress=progress,
        )

    if engine == "kalman":
        return pit_kalman_betas(
            df_asset_rets=df_asset_rets,
            df_fact_rets=df_fact_rets,
            rebalance_time_index=rebalance_time_index,
            min_timestamps=min_timestamps,
            base_q=base_q,
            base_r=base_r,
            vix=vix,
            liquidity=liquidity,
            vix_k=vix_k,
            vix_v0=vix_v0,
            liquidity_multiplier=liquidity_multiplier,
            outlier_clip_std=outlier_clip_std,
            initial_betas=initial_betas_override,
            chunk_size=chunk_size,
            progress=progress,
        )

    if engine == "robust_then_kalman":
        robust_res = fast_pit_robust_betas(
            df_asset_rets=df_asset_rets,
            df_fact_rets=df_fact_rets,
            rebalance_time_index=rebalance_time_index,
            half_life=half_life,
            lambda_=lambda_,
            min_timestamps=min_timestamps,
            lookback_days=lookback_days,
            progress=progress,
        )
        df_betas_robust = robust_res["df_betas"]

        last_per_factor = df_betas_robust.groupby(level="factor").tail(1)
        last_per_factor = last_per_factor.reset_index("factor")
        if "timestamp" in last_per_factor.columns:
            last_per_factor = last_per_factor.drop(columns="timestamp")
        initial_betas = last_per_factor.set_index("factor").T

        initial_betas_use = initial_betas_override if initial_betas_override is not None else initial_betas

        return pit_kalman_betas(
            df_asset_rets=df_asset_rets,
            df_fact_rets=df_fact_rets,
            rebalance_time_index=rebalance_time_index,
            min_timestamps=min_timestamps,
            base_q=base_q,
            base_r=base_r,
            vix=vix,
            liquidity=liquidity,
            vix_k=vix_k,
            vix_v0=vix_v0,
            liquidity_multiplier=liquidity_multiplier,
            outlier_clip_std=outlier_clip_std,
            initial_betas=initial_betas_use,
            chunk_size=chunk_size,
            progress=progress,
        )

    raise ValueError("engine must be one of 'robust', 'kalman', or 'robust_then_kalman'.")


# ---------------- Main pipeline ----------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_all = load_returns(STOCK_RETS_PATH)

    asset_cols = [s for s in STOCK_SECTOR_MAP.keys() if s in df_all.columns]
    factor_cols = [MARKET] + [etf for etf in SECTOR_ETFS.keys() if etf in df_all.columns]

    if not asset_cols:
        raise ValueError("No mapped stocks found in returns file")
    if MARKET not in df_all.columns:
        raise ValueError(f"Market ticker {MARKET} not found in returns file")

    df_asset = df_all[asset_cols]
    df_factor = df_all[factor_cols]

    common_idx = df_asset.index.intersection(df_factor.index)
    df_asset = df_asset.loc[common_idx]
    df_factor = df_factor.loc[common_idx]

    # choose engine and speed parameters
    engine = "robust_then_kalman"   # "robust", "kalman", or "robust_then_kalman"
    lookback_days = 252             # smaller = faster, noisier
    min_timestamps = 60

    # monthly rebalance
    rebalance_dates = df_asset.resample("ME").last().index

    result = pit_betas_with_strategy(
        df_asset_rets=df_asset,
        df_fact_rets=df_factor,
        engine=engine,
        rebalance_time_index=rebalance_dates,
        min_timestamps=min_timestamps,
        progress=True,
        half_life=60,
        lambda_=0.97,
        lookback_days=lookback_days,
        base_q=0.01,
        base_r=1.0,
    )

    df_betas = result["df_betas"]
    df_hedge = result["df_hedge_rets"]
    df_resids = result["df_asset_resids"]

    df_betas.to_csv(BETAS_CSV)
    df_hedge.to_csv(HEDGE_CSV)
    df_resids.to_csv(RESIDS_CSV)

    beta_mean = df_betas.mean(axis=1).unstack("factor")

    plt.figure(figsize=(12, 6))
    for factor in beta_mean.columns:
        plt.plot(beta_mean.index, beta_mean[factor], label=str(factor))
    plt.axhline(0.0, color="black", linewidth=0.5)
    plt.title(f"PIT Hedge Ratios (fast, {engine})")
    plt.ylabel("Average beta")
    plt.xlabel("Date")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(HEDGE_PLOT, dpi=150)
    plt.close()

    print(f"Engine: {engine}")
    print("Betas shape:", df_betas.shape)
    print("Residuals shape:", df_resids.shape)


if __name__ == "__main__":
    main()
