import numpy as np
import pandas as pd
import statsmodels.api as sm

def exponential_weights(n, half_life=None, lambda_=None):
    """
    Generate exponential decay weights for n time periods.
    Either half_life or lambda_ must be provided.
    """
    if lambda_ is None:
        lambda_ = np.exp(np.log(0.5) / half_life)
    weights = lambda_ ** np.arange(n-1, -1, -1)
    return weights / np.sum(weights)  # normalize

def compute_time_weighted_robust_betas(Y, X, half_life=None, lambda_=None):
    """
    Perform robust regression (RLM) with exponential time-weighting.

    Parameters:
    - Y: (n_timestamps x n_assets) DataFrame of dependent returns
    - X: (n_timestamps x n_factors) DataFrame of factor returns
    - half_life: half-life in time units (e.g., weeks)
    - lambda_: optional, decay factor (e.g., 0.985)

    Returns:
    - beta_matrix: (n_factors + 1) x n_assets DataFrame
    """
    n_timestamps, n_assets = Y.shape
    _, n_factors = X.shape

    # Calculate weights
    weights = exponential_weights(n_timestamps, half_life=half_life, lambda_=lambda_)
    sqrt_weights = np.sqrt(weights)

    beta_matrix = pd.DataFrame(index=["Intercept"] + list(X.columns), columns=Y.columns)

    for asset in Y.columns:
        y = Y[asset].values
        X_weighted = X.multiply(sqrt_weights, axis=0)
        y_weighted = y * sqrt_weights

        Xw_const = sm.add_constant(X_weighted)
        rlm_model = sm.RLM(y_weighted, Xw_const, M=sm.robust.norms.HuberT())
        rlm_results = rlm_model.fit()

        beta_matrix[asset] = rlm_results.params

    return beta_matrix
