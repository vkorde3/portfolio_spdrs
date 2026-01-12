"""
Improved backtest on sector-neutral residuals
- Ensemble signals: residual momentum + current residual rank
- Winsorization, z-score normalization
- Per-asset vol scaling, per-name caps, gross exposure cap
- Shrinkage toward previous weights and turnover threshold
- Weekly rebalancing, transaction costs, optional vol targeting
"""

import os
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- USER PARAMETERS ----------------
RESIDS_PATH = "data/backtest/stock_sector_residuals.parquet"
REBALANCE_FREQ = "BME"           # business month end rebalance (monthly)
MOM_LOOKBACKS = [21, 63, 126]   # ensemble lookbacks
TOP_K = 10                      # long-only: number of names to go long each rebalance
PER_NAME_CAP = 0.10             # allow more room for winners (10%)
GROSS_CAP = 1.0                 # keep gross exposure 1.0 (long-only => sum weights = 1)
TURNOVER_THRESHOLD = 0.02       # require >2% weight change to trade
WEIGHT_DECAY = 0.6              # stronger shrink toward previous weights (0.6)
VOL_LOOKBACK = 63
VOL_TARGET = 0.10               # try 10% annual portfolio vol target (optional)
TRANSACTION_COST = 0.0005
WINSOR_PCT = 0.02
ANNUAL_DAYS = 252
MIN_HISTORY = max(max(MOM_LOOKBACKS), VOL_LOOKBACK) + 5
# -------------------------------------------------

def load_residuals(path: str) -> pd.DataFrame:
    path = os.path.expanduser(path)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index().dropna(how="all")
    # ensure numeric dtype
    df = df.astype(float)
    return df

def get_rebalance_dates(df: pd.DataFrame, freq: str = "W-FRI") -> pd.DatetimeIndex:
    # choose last available trading day for each period
    return df.resample(freq).last().index

def winsorize_series(s: pd.Series, pct: float = WINSOR_PCT) -> pd.Series:
    """Winsorize a pandas Series (safe if a scalar is accidentally passed)."""
    # If a scalar snuck in, convert to Series
    if not isinstance(s, pd.Series):
        try:
            s = pd.Series(s)
        except Exception:
            # fallback: return empty series with expected index (no-op)
            return pd.Series(dtype=float)
    low = s.quantile(pct)
    high = s.quantile(1 - pct)
    return s.clip(lower=low, upper=high)

def make_signals(df_resid: pd.DataFrame, dt: pd.Timestamp, lookbacks=MOM_LOOKBACKS):
    """
    Ensemble momentum: compute cumulative residual returns for multiple lookbacks,
    average their z-scores to make a robust momentum signal.
    Returns a Series indexed by tickers (higher -> better).
    """
    # get latest row
    if dt not in df_resid.index:
        row = df_resid.loc[:dt].iloc[-1]
    else:
        row = df_resid.loc[dt]

    # collect z-scored momentums
    z_list = []
    for lb in lookbacks:
        loc = df_resid.index.get_loc(row.name)
        start_loc = max(0, loc - lb + 1)
        window = df_resid.iloc[start_loc: loc + 1]
        if window.shape[0] == 0:
            mom = pd.Series(0.0, index=df_resid.columns)
        else:
            cum = (1 + window).prod(axis=0) - 1
            # winsorize then zscore
            cum = winsorize_series(cum, pct=WINSOR_PCT)
            z = (cum - cum.mean()) / (cum.std(ddof=0) + 1e-12)
            z_list.append(z)
    # average z-scores
    if not z_list:
        combined = pd.Series(0.0, index=df_resid.columns)
    else:
        combined = pd.concat(z_list, axis=1).mean(axis=1)
    # final winsorize + zscore
    combined = winsorize_series(combined, pct=WINSOR_PCT)
    combined_z = (combined - combined.mean()) / (combined.std(ddof=0) + 1e-12)
    combined_z = combined_z.reindex(df_resid.columns).fillna(0.0)
    return combined_z


def volatility_scale_weights(raw_w: pd.Series, df_rets: pd.DataFrame,
                             vol_lookback: int = VOL_LOOKBACK,
                             vol_target: Optional[float] = VOL_TARGET) -> pd.Series:
    """
    Scale raw weights by inverse asset vol and enforce per-name caps and gross exposure.
    """
    # trailing daily std (unannualized), then convert to annual vol
    vol = df_rets.rolling(vol_lookback).std(ddof=0)
    if vol.shape[0] == 0:
        current_vol = pd.Series(np.nan, index=raw_w.index)
    else:
        current_vol = vol.iloc[-1] * np.sqrt(ANNUAL_DAYS)

    # replace problematic values with median
    median_vol = current_vol.median() if not np.isnan(current_vol.median()) else 1.0
    current_vol = current_vol.fillna(median_vol).replace(0, median_vol)
    inv_vol = 1.0 / current_vol
    inv_vol = inv_vol.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    scaled = raw_w * inv_vol
    # enforce per-name cap
    scaled = scaled.clip(lower=-PER_NAME_CAP, upper=PER_NAME_CAP)

    # normalize to gross cap
    gross = scaled.abs().sum()
    if gross > 0:
        scaled = scaled * (GROSS_CAP / gross)

    # optional: scale for portfolio vol target using diagonal approx
    if vol_target is not None:
        # approximate ann vol via sqrt(sum (w^2 * sigma^2))
        approx_ann_vol = np.sqrt((scaled ** 2 * (current_vol ** 2)).sum())
        if approx_ann_vol > 0:
            scale = vol_target / approx_ann_vol
            scaled = scaled * scale
            # re-apply caps and gross normalization
            scaled = scaled.clip(lower=-PER_NAME_CAP, upper=PER_NAME_CAP)
            gross = scaled.abs().sum()
            if gross > 0:
                scaled = scaled * (GROSS_CAP / gross)

    return scaled

def make_weights(df_resid: pd.DataFrame, df_rets: pd.DataFrame,
                                  prev_weights: Optional[pd.Series], dt: pd.Timestamp):
    """
    Build weights using ensemble momentum, but long-only top-K.
    - pick top_k names by combined z
    - equal-weight longs, vol-scale them, apply caps, shrink toward prev_weights
    - enforce turnover threshold
    """
    combined_z = make_signals(df_resid, dt, lookbacks=MOM_LOOKBACKS)
    # pick top_k
    top = combined_z.nlargest(TOP_K)
    if top.sum() == 0 or top.isna().all():
        raw = pd.Series(0.0, index=df_rets.columns)
    else:
        raw = pd.Series(0.0, index=df_rets.columns)
        raw[top.index] = top.values
        # ensure positivity and normalize to sum=1 (long-only)
        raw = raw.clip(lower=0.0)
        if raw.sum() > 0:
            raw = raw / raw.sum()
    # apply volatility scaling (so each long is risk-equalized)
    scaled = volatility_scale_weights(raw, df_rets, vol_lookback=VOL_LOOKBACK, vol_target=None)
    # scaled now has signed weights (but since raw >=0, scaled >=0). Normalize gross to GROSS_CAP
    if scaled.abs().sum() > 0:
        scaled = scaled * (GROSS_CAP / scaled.abs().sum())
    # apply per name cap
    scaled = scaled.clip(lower=0.0, upper=PER_NAME_CAP)
    # renormalize if some cap hit
    if scaled.sum() > 0:
        scaled = scaled / scaled.sum() * GROSS_CAP

    # shrink to previous weights to reduce turnover
    if prev_weights is None:
        prev_weights = pd.Series(0.0, index=df_rets.columns)
    new_w = (1 - WEIGHT_DECAY) * scaled + WEIGHT_DECAY * prev_weights

    # turnover threshold
    delta = new_w - prev_weights
    small = delta.abs() < TURNOVER_THRESHOLD
    final_w = prev_weights.copy()
    final_w[~small] = new_w[~small]

    # ensure final normalization
    if final_w.sum() > 0:
        final_w = final_w / final_w.sum() * GROSS_CAP
    else:
        final_w = final_w.fillna(0.0)

    final_w = final_w.reindex(df_rets.columns).fillna(0.0)
    return final_w

# ---------------- Backtest runner ----------------
def run_backtest(df_resid: pd.DataFrame,
                 rebalance_freq: str = REBALANCE_FREQ,
                 transaction_cost: float = TRANSACTION_COST) -> dict:
    df_resid = df_resid.sort_index().dropna(how="all")
    # need residual returns and also returns history for vol estimation (we use df_resid as residual returns)
    df_rets = df_resid.copy()

    # require enough history
    if df_rets.shape[0] < MIN_HISTORY:
        raise ValueError(f"Not enough history ({df_rets.shape[0]} rows) for lookbacks (need >= {MIN_HISTORY})")

    rebalance_dates = get_rebalance_dates(df_rets, rebalance_freq)
    # only keep rebalance dates after we've accumulated MIN_HISTORY
    rebalance_dates = rebalance_dates[rebalance_dates >= df_rets.index[MIN_HISTORY - 1]]

    prev_w = None
    weight_rows = []
    for dt in rebalance_dates:
        w = make_weights(df_resid, df_rets, prev_w, dt)
        w.name = dt
        weight_rows.append(w)
        prev_w = w

    df_w_reb = pd.DataFrame(weight_rows, index=rebalance_dates).fillna(0.0)
    # forward-fill to daily
    df_w_daily = df_w_reb.reindex(df_rets.index, method="ffill").fillna(0.0)

    # use weights known at t-1 to compute returns at t
    exp_weights = df_w_daily.shift(1).fillna(0.0)

    # daily portfolio returns (gross)
    daily_portfolio_rets = (exp_weights * df_rets).sum(axis=1)

    # turnover and transaction costs
    turnover = df_w_daily.diff().abs().sum(axis=1).fillna(0.0)
    trade_costs = turnover * transaction_cost
    # apply trade costs on the day of rebalancing (we subtract from returns)
    daily_net_rets = daily_portfolio_rets - trade_costs

    daily_net_rets = daily_net_rets.dropna()

    # performance metrics
    total_days = len(daily_net_rets)
    if total_days == 0:
        raise ValueError("No returns after backtest; check inputs")

    # annualized return
    ann_return = (1.0 + daily_net_rets).prod() ** (ANNUAL_DAYS / total_days) - 1.0
    ann_vol = daily_net_rets.std(ddof=0) * np.sqrt(ANNUAL_DAYS)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    # max drawdown
    cum = (1 + daily_net_rets).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    stats = {
        "days": total_days,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "turnover_mean": turnover.mean(),
        "avg_gross_exposure": df_w_daily.abs().mean(axis=1).mean(),
    }

    out = {
        "daily_net_rets": daily_net_rets,
        "weights_rebalance": df_w_reb,
        "weights_daily": df_w_daily,
        "stats": stats,
    }
    return out

# ---------------- Main ----------------
def print_stats(stats: dict):
    print("Backtest stats:")
    print(f"  Days: {stats['days']}")
    print(f"  Annualized Return: {stats['ann_return']:.2%}")
    print(f"  Annual Vol: {stats['ann_vol']:.2%}")
    print(f"  Sharpe: {stats['sharpe']:.2f}")
    print(f"  Max Drawdown: {stats['max_drawdown']:.2%}")
    print(f"  Mean daily turnover: {stats['turnover_mean']:.4f}")
    print(f"  Avg gross exposure: {stats['avg_gross_exposure']:.4f}")

if __name__ == "__main__":
    df_resid = load_residuals(RESIDS_PATH)
    result = run_backtest(df_resid, rebalance_freq=REBALANCE_FREQ, transaction_cost=TRANSACTION_COST)
    stats = result["stats"]
    print_stats(stats)

    # plot cumulative returns
    daily = result["daily_net_rets"]
    cum = (1 + daily).cumprod() - 1
    plt.figure(figsize=(10, 5))
    plt.plot(cum.index, cum, label="Strategy (net)")
    plt.title("Improved Strategy Cumulative Returns")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
