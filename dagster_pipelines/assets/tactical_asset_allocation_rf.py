import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_market_calendars as mcal
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from vbase_utils.stats.pit_robust_betas import pit_robust_betas
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
SPY = "SPY"
ALL_TICKERS = [SPY] + SECTORS
K_LONG = 3
K_SHORT = 3
ROLLING_WINDOW = 252  # For feature calculation and RF training
OUTPUT_DIR = "data/backtest"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def download_prices():
    logger.info("Downloading ETF price history")
    data = yf.download(ALL_TICKERS, start="2015-01-01", auto_adjust=True)["Close"]
    # Filter for NYSE trading days
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=data.index.min(), end_date=data.index.max())
    trading_days = pd.to_datetime(schedule.index)
    data = data[data.index.isin(trading_days)]
    return data

def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    return returns

def compute_residuals(returns: pd.DataFrame) -> pd.DataFrame:
    logger.info("Running pit_robust_betas for residuals")
    Y = returns[SECTORS]
    X = returns[[SPY]]
    weekly_rebalance = pd.DatetimeIndex([d for d in Y.index if d.weekday() == 4])
    
    results = pit_robust_betas(
        df_asset_rets=Y,
        df_fact_rets=X,
        half_life=126,
        min_timestamps=63,
        rebalance_time_index=weekly_rebalance,
        progress=True
    )
    
    residuals = results["df_asset_resids"].dropna(how="all")
    # Save intermediate results
    results["df_betas"].dropna(how="all").to_csv(f"{OUTPUT_DIR}/sector_betas.csv")
    results["df_hedge_rets"].dropna(how="all").to_csv(f"{OUTPUT_DIR}/hedge_returns.csv")
    residuals.to_csv(f"{OUTPUT_DIR}/residual_returns.csv")
    logger.info(f"Saved betas, hedge returns, and residuals to {OUTPUT_DIR}")
    return residuals

def compute_features(series: pd.Series) -> pd.DataFrame:
    """Compute features for RF: lagged residuals and rolling statistics."""
    features = pd.DataFrame(index=series.index)
    # Lagged residuals (1, 3, 5, 10, 20 days)
    for lag in [1, 3, 5, 10, 20]:
        features[f"lag_{lag}"] = series.shift(lag)
    # Rolling mean and std (20, 60 days)
    for window in [20, 60]:
        features[f"mean_{window}"] = series.rolling(window).mean()
        features[f"std_{window}"] = series.rolling(window).std()
    return features.dropna()

def residual_momentum_rf(residuals: pd.DataFrame) -> pd.Series:
    rebal_dates = sorted(set(residuals.index[residuals.index.weekday == 4]))
    if len(rebal_dates) < 3:
        return pd.Series(dtype=float)
    portfolio_rets = []
    sectors = residuals.columns
    min_history = 100  # Minimum days for RF training

    for i in range(2, len(rebal_dates)):
        rebalance_date = rebal_dates[i]
        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else residuals.index[-1]
        forecasts = {}
        for sector in sectors:
            past_data = residuals[sector].loc[:rebalance_date].iloc[-ROLLING_WINDOW:]
            if len(past_data) < min_history or past_data.isna().all():
                continue
            # Prepare features and target
            features = compute_features(past_data)
            target = past_data.shift(-1).dropna()  # Next day's residual
            feature_index = features.index.intersection(target.index)
            if len(feature_index) < min_history:
                continue
            X = features.loc[feature_index]
            y = target.loc[feature_index]
            # Train RF model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            # Predict for next day (use last available features with column names)
            last_features = features.iloc[-1:]  # Keep as DataFrame to preserve column names
            forecast = model.predict(last_features)[0]
            forecasts[sector] = forecast
        if len(forecasts) < K_LONG + K_SHORT:
            continue
        pred_series = pd.Series(forecasts)
        sorted_pred = pred_series.sort_values(ascending=False)
        long_sectors = sorted_pred.head(K_LONG).index
        short_sectors = sorted_pred.tail(K_SHORT).index
        fwd_rets = residuals.loc[rebalance_date:next_date].iloc[1:]
        if fwd_rets.empty:
            continue
        long_ret = fwd_rets[long_sectors].mean(axis=1)
        short_ret = fwd_rets[short_sectors].mean(axis=1)
        strat_ret = long_ret - short_ret
        portfolio_rets.append(strat_ret)
    if not portfolio_rets:
        return pd.Series(dtype=float)
    return pd.concat(portfolio_rets).sort_index()

def compute_sector_indices(residuals: pd.DataFrame) -> pd.DataFrame:
    long_indices = (1 + residuals).cumprod()
    long_indices.columns = [f"{s}_long" for s in long_indices.columns]
    short_indices = (1 - residuals).cumprod()
    short_indices.columns = [f"{s}_short" for s in short_indices.columns]
    indices = pd.concat([long_indices, short_indices], axis=1)
    return indices

def evaluate(returns: pd.Series):
    if returns.empty:
        return None
    ann_return = (1 + returns.mean()) ** 252 - 1
    ann_vol = returns.std() * np.sqrt(252)
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

def main():
    prices = download_prices()
    returns = compute_daily_returns(prices)
    residuals = compute_residuals(returns)
    indices = compute_sector_indices(residuals)
    strategy_returns = residual_momentum_rf(residuals)
    metrics = evaluate(strategy_returns)

    # Save results
    indices.to_csv(f"{OUTPUT_DIR}/sector_indices.csv")
    strategy_returns.to_csv(f"{OUTPUT_DIR}/strategy_returns.csv")
    pd.DataFrame([metrics]).to_csv(f"{OUTPUT_DIR}/strategy_metrics.csv", index=False)

    # Plot
    cum_returns = (1 + strategy_returns).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(cum_returns.index, cum_returns, label="RF Momentum Strategy")
    plt.title("RF-Based Residual Momentum Strategy Performance")
    plt.ylabel("Cumulative Return")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/strategy_performance_rf.png")
    plt.close()

    logger.info("Metrics: %s", metrics)
    logger.info(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()