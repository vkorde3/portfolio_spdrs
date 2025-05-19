import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def exponential_weights(
    n: int,
    half_life: Optional[float] = None,
    lambda_: Optional[float] = None
) -> np.ndarray:
    """
    Generate exponential decay weights for n time periods.

    Either half_life or lambda_ must be provided.

    Parameters:
    - n (int): Number of time periods.
    - half_life (Optional[float]): Half-life in time units (e.g., days). Must be positive.
    - lambda_ (Optional[float]): Decay factor (e.g., 0.985). Must be between 0 and 1.

    Returns:
    - weights (np.ndarray): Normalized exponential decay weights.

    Raises:
    - ValueError: If neither half_life nor lambda_ is provided.
    - ValueError: If half_life is not positive or lambda_ is not between 0 and 1.
    """
    if half_life is None and lambda_ is None:
        raise ValueError("Either half_life or lambda_ must be provided.")
    if half_life is not None and half_life <= 0:
        raise ValueError("half_life must be positive.")
    if lambda_ is not None and not (0 < lambda_ < 1):
        raise ValueError("lambda_ must be between 0 and 1.")

    if lambda_ is None:
        lambda_ = np.exp(np.log(0.5) / half_life)
    weights: np.ndarray = lambda_ ** np.arange(n - 1, -1, -1)
    return weights / np.sum(weights)  # normalize


def compute_time_weighted_robust_betas(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    half_life: Optional[float] = None,
    lambda_: Optional[float] = None,
    min_timestamps: int = 10
) -> pd.DataFrame:
    """
    Perform robust regression (RLM) with exponential time-weighting.

    Parameters:
    - Y (pd.DataFrame): (n_timestamps x n_assets) DataFrame of dependent returns.
    - X (pd.DataFrame): (n_timestamps x n_factors) DataFrame of factor returns.
    - half_life (Optional[float]): Half-life in time units (e.g., days). Must be positive.
    - lambda_ (Optional[float]): Decay factor (e.g., 0.985). Must be between 0 and 1.
    - min_timestamps (int): Minimum number of timestamps required for regression (default: 10).

    Returns:
    - beta_matrix (pd.DataFrame): (n_factors + 1) x n_assets DataFrame of betas.

    Raises:
    - ValueError: If inputs are empty, have insufficient data, mismatched rows, excessive NaNs,
                  or near-zero variance in X.
    """
    # Check for empty inputs
    if Y.empty or X.empty:
        logger.error("Input DataFrame Y or X is empty.")
        raise ValueError("Input DataFrame Y or X is empty.")

    # Check for mismatched row counts
    if Y.shape[0] != X.shape[0]:
        logger.error(f"Mismatched row counts: Y has {Y.shape[0]} rows, X has {X.shape[0]} rows.")
        raise ValueError(f"Mismatched row counts: Y has {Y.shape[0]} rows, X has {X.shape[0]} rows.")

    # Align Y and X indices and handle NaNs
    combined: pd.DataFrame = pd.concat([Y, X], axis=1, join='inner')
    if combined.empty:
        logger.error("No overlapping timestamps between Y and X.")
        raise ValueError("No overlapping timestamps between Y and X.")

    Y_clean: pd.DataFrame = combined[Y.columns].dropna()
    X_clean: pd.DataFrame = combined[X.columns].loc[Y_clean.index]

    n_timestamps, n_assets = Y_clean.shape
    _, n_factors = X_clean.shape

    # Log data cleaning results
    if len(Y) != n_timestamps:
        dropped_rows = len(Y) - n_timestamps
        logger.warning(
            f"Dropped {dropped_rows} rows due to NaNs or index misalignment. "
            f"Remaining timestamps: {n_timestamps}"
        )
        # Check for excessive NaN dropping
        if dropped_rows > 0.5 * len(Y):
            logger.error(f"Excessive data loss: {dropped_rows} rows dropped (>50% of {len(Y)}).")
            raise ValueError(f"Excessive data loss: {dropped_rows} rows dropped (>50% of {len(Y)}).")

    # Check minimum timestamps
    if n_timestamps < min_timestamps:
        logger.error(
            f"Insufficient data: {n_timestamps} timestamps available, "
            f"minimum required is {min_timestamps}."
        )
        raise ValueError(
            f"Insufficient data: {n_timestamps} timestamps available, "
            f"minimum required is {min_timestamps}."
        )

    # Check for near-zero variance in X
    if X_clean.var().min() < 1e-10:
        logger.error("One or more factors in X have near-zero variance.")
        raise ValueError("One or more factors in X have near-zero variance.")

    # Calculate weights
    weights: np.ndarray = exponential_weights(n_timestamps, half_life=half_life, lambda_=lambda_)
    sqrt_weights: np.ndarray = np.sqrt(weights)

    beta_matrix: pd.DataFrame = pd.DataFrame(index=["Intercept"] + list(X_clean.columns), columns=Y_clean.columns)

    for asset in Y_clean.columns:
        y: np.ndarray = Y_clean[asset].values
        X_weighted: pd.DataFrame = X_clean.multiply(sqrt_weights, axis=0)
        y_weighted: np.ndarray = y * sqrt_weights

        Xw_const: pd.DataFrame = sm.add_constant(X_weighted)
        rlm_model = sm.RLM(y_weighted, Xw_const, M=sm.robust.norms.HuberT())
        rlm_results = rlm_model.fit()

        beta_matrix[asset] = rlm_results.params

    return beta_matrix


# Recommendations for half-life based on horizon
"""
| Horizon (days) | Recommended half-life (days) |
|----------------|------------------------------|
| 30             | 10                           |
| 60             | 20                           |
| 90             | 30                           |
| 180            | 60                           |
| 365            | 120                          |

For a 90-day horizon, a half-life of 30 days is recommended.
This balances responsiveness to recent data with stability from historical data.
"""
