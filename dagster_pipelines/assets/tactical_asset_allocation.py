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
OUTPUT_DIR = "data/backtest"
RESIDUAL_CSV = "data/backtest/residual_returns.csv"
K_LONG = 3
K_SHORT = 3
REBALANCE_EVERY = 4  # Every 4 weeks = Monthly

def residual_momentum_rotation(df: pd.DataFrame, lookback: int, exclude_recent: int, return_positions: bool = False):
    """
    Rotate into long/short positions based on residual momentum.

    Parameters
    ----------
    df : pd.DataFrame
        Residual returns DataFrame (sectors as columns, weekly dates as index).
    lookback : int
        Lookback window in weeks.
    exclude_recent : int
        Number of most recent weeks to exclude from ranking.
    return_positions : bool
        If True, also return DataFrame of long/short tickers selected each rebalance.

    Returns
    -------
    ret_series : pd.Series
        Strategy returns over time.
    positions (optional) : pd.DataFrame
        Long and short tickers at each rebalance date.
    """
    returns = df.copy()
    ret_series = pd.Series(index=returns.index, dtype=float)
    positions_list = []

    for i in range(lookback + exclude_recent, len(returns)):
        # Define lookback window (excluding most recent weeks)
        window = returns.iloc[i - lookback - exclude_recent : i - exclude_recent]

        # Compute mean residual return for ranking
        scores = window.mean()

        # Select top 3 long and bottom 3 short
        longs = scores.nlargest(3).index.tolist()
        shorts = scores.nsmallest(3).index.tolist()

        # Compute portfolio return this week (equal-weight long/short)
        week_ret = returns.iloc[i]
        long_ret = week_ret[longs].mean()
        short_ret = week_ret[shorts].mean()
        ret_series.iloc[i] = 0.5 * (long_ret - short_ret)  # long/short spread

        # Save positions if requested
        if return_positions:
            positions_list.append({
                "Date": returns.index[i],
                "Longs": ",".join(longs),
                "Shorts": ",".join(shorts)
            })

    ret_series = ret_series.dropna()

    if return_positions:
        positions_df = pd.DataFrame(positions_list).set_index("Date")
        return ret_series, positions_df
    else:
        return ret_series


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
    """
    Run grid search across lookback/exclude parameters, evaluate metrics,
    save results table, return series, and sector selections.
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(how="all").dropna(axis=1)
    df = df[df.index.weekday == 4]

    results = []
    ret_dict = {}
    positions_dict = {}  # store long/short tickers at each rebalance

    for lookback, exclude_recent in product(range(8, 27, 2), range(1, 5)):
        ret_series, positions = residual_momentum_rotation(df, lookback, exclude_recent, return_positions=True)
        # 👆 modify your residual_momentum_rotation so it can optionally return (returns, positions)

        metrics = evaluate(ret_series)
        if metrics:
            config_name = f"L{lookback}_X{exclude_recent}"
            ret_dict[config_name] = ret_series
            positions_dict[config_name] = positions  # DataFrame of long/short tickers
            results.append({
                "Lookback": lookback,
                "ExcludeRecent": exclude_recent,
                **metrics
            })

    # Save metrics
    df_results = pd.DataFrame(results).sort_values("Sharpe", ascending=False)
    df_results.to_csv("data/backtest/grid_search_results.csv", index=False)
    print("📁 Grid search results saved to data/backtest/grid_search_results.csv")

    # Save return series
    df_returns = pd.DataFrame(ret_dict)
    df_returns.to_csv("data/backtest/grid_search_returns.csv")
    print("📁 Strategy return series saved to data/backtest/grid_search_returns.csv")

    # Save long/short selections
    all_positions = pd.concat(positions_dict, axis=1)
    all_positions.to_csv("data/backtest/grid_search_positions.csv")
    print("📁 Long/short selections saved to data/backtest/grid_search_positions.csv")

    return df_results, df_returns, all_positions


def plot_top_strategies(df_results: pd.DataFrame, ret_df: pd.DataFrame, top_n: int = 5):
    """
    Plot Sharpe Ratios and Cumulative Returns for top N configurations.
    """
    # --- Sharpe Ratio bar plot ---
    top = df_results.head(top_n)
    # labels = [f"L{int(r['Lookback'])}_X{int(r['ExcludeRecent'])}" for _, r in top.iterrows()]
    
    # plt.figure(figsize=(10, 4))
    # plt.bar(labels, top["Sharpe"], color="steelblue")
    # plt.title("Top Sharpe Ratios by Parameter Combination")
    # plt.ylabel("Sharpe Ratio")
    # plt.grid(True, axis="y")
    # plt.tight_layout()
    # plt.show()

    # --- Cumulative Returns plot ---
    plt.figure(figsize=(12, 6))
    for _, row in top.iterrows():
        col_name = f"L{int(row['Lookback'])}_X{int(row['ExcludeRecent'])}"
        if col_name in ret_df.columns:
            cum_returns = (1 + ret_df[col_name]).cumprod()
            plt.plot(cum_returns, label=col_name)
        else:
            print(f"⚠️ Warning: {col_name} not found in return DataFrame")
    
    plt.title(f"Cumulative Returns of Top {top_n} Strategies")
    plt.ylabel("Cumulative Return (Growth of $1)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sector_rotation_performance.png")
    plt.close()


if __name__ == "__main__":
    results_df, returns_df, all_positions = run_grid_search(RESIDUAL_CSV)
    print(results_df.head(10))  # Preview top strategies
    plot_top_strategies(results_df, returns_df)
    print("✅ Grid search completed and results plotted.")