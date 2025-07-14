# """
# Residual Momentum Tactical Allocation Strategy (Weekly)

# Uses SPY residuals from 11 sector ETFs, rebalances every 4 weeks using
# a momentum signal from prior weekly residuals.

# Assumes input is already weekly, e.g., from pit_robust_betas.
# """

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import product

# Constants
RESIDUAL_CSV = "data/backtest/residual_returns.csv"
K_LONG = 3
K_SHORT = 3
REBALANCE_EVERY = 4  # Every 4 weeks = Monthly

def residual_momentum_rotation(df: pd.DataFrame, lookback: int, exclude_recent: int) -> pd.Series:
    """
    Compute residual momentum rotation strategy returns.
    Assumes df is weekly returns (e.g. Friday rebalance).
    """
    rebal_dates = df.index[::REBALANCE_EVERY]  # every 4 weeks
    portfolio_rets = []

    for i in range(lookback + exclude_recent, len(rebal_dates) - 1):
        rebalance_date = rebal_dates[i]
        next_date = rebal_dates[i + 1]

        lookback_end = rebal_dates[i - exclude_recent]
        lookback_start = rebal_dates[i - (exclude_recent + lookback)]

        lookback_window = df.loc[lookback_start:lookback_end]

        # Skip if missing data
        if lookback_window.shape[0] < lookback:
            continue

        momentum = lookback_window.sum()
        sorted_momentum = momentum.sort_values(ascending=False)
        long_sectors = sorted_momentum.head(K_LONG).index
        short_sectors = sorted_momentum.tail(K_SHORT).index

        fwd_rets = df.loc[rebalance_date:next_date]
        if fwd_rets.empty:
            continue

        long_ret = fwd_rets[long_sectors].mean(axis=1)
        short_ret = fwd_rets[short_sectors].mean(axis=1)
        strat_ret = long_ret - short_ret
        portfolio_rets.append(strat_ret)

    if not portfolio_rets:
        return pd.Series(dtype=float)

    return pd.concat(portfolio_rets).sort_index()


def evaluate(returns: pd.Series):
    """Return performance metrics: Sharpe, CAGR, Max Drawdown."""
    if returns.empty:
        return None

    ann_return = (1 + returns.mean()) ** 52 - 1
    ann_vol = returns.std() * np.sqrt(52)
    sharpe = ann_return / ann_vol if ann_vol else np.nan
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min()

    return {
        "Sharpe": sharpe,
        "CAGR": ann_return,
        "MaxDrawdown": max_dd,
        "TotalReturn": cum_returns.iloc[-1] - 1,
    }


def run_grid_search(csv_path: str):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.dropna(how="all").dropna(axis=1)  # Clean leading NaNs
    df = df[df.index.weekday == 4]  # Friday returns only (weekly freq)

    results = []

    for lookback, exclude_recent in product(range(8, 27, 2), range(1, 5)):
        ret_series = residual_momentum_rotation(df, lookback, exclude_recent)
        metrics = evaluate(ret_series)
        if metrics:
            results.append({
                "Lookback": lookback,
                "ExcludeRecent": exclude_recent,
                **metrics
            })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("Sharpe", ascending=False)
    df_results.to_csv("data/backtest/grid_search_results.csv", index=False)
    print("📁 Grid search results saved to data/backtest/grid_search_results.csv")
    return df_results


def plot_top_strategies(df_results: pd.DataFrame, top_n: int = 5):
    """
    Plot Sharpe Ratio for top N configurations.
    """
    top = df_results.head(top_n)
    labels = [f"L{r['Lookback']}_X{r['ExcludeRecent']}" for _, r in top.iterrows()]
    plt.figure(figsize=(10, 4))
    plt.bar(labels, top["Sharpe"], color="steelblue")
    plt.title("Top Sharpe Ratios by Parameter Combination")
    plt.ylabel("Sharpe Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results_df = run_grid_search(RESIDUAL_CSV)
    print(results_df.head(10))  # Preview top strategies
    plot_top_strategies(results_df)
    print("✅ Grid search completed and results plotted.")