import numpy as np
import pandas as pd
import unittest
from robust_regression import compute_time_weighted_robust_betas

class TestComputeTimeWeightedRobustBetas(unittest.TestCase):
    def test_single_asset(self):
        """Test beta estimation for a single ETF with known beta."""
        np.random.seed(42)
        n_timestamps = 100
        spy_returns = np.random.normal(0, 0.01, n_timestamps)
        asset_returns = 1.5 * spy_returns + np.random.normal(0, 0.005, n_timestamps)

        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': spy_returns})

        beta_matrix = compute_time_weighted_robust_betas(Y, X, half_life=30)
        self.assertAlmostEqual(beta_matrix.loc['SPY', 'Asset1'], 1.5, delta=0.1)

    def test_multiple_assets(self):
        """Test beta estimation for multiple ETFs with known betas."""
        np.random.seed(42)
        n_timestamps = 100
        spy_returns = np.random.normal(0, 0.01, n_timestamps)
        asset1_returns = 1.2 * spy_returns + np.random.normal(0, 0.005, n_timestamps)
        asset2_returns = 0.8 * spy_returns + np.random.normal(0, 0.005, n_timestamps)

        Y = pd.DataFrame({'Asset1': asset1_returns, 'Asset2': asset2_returns})
        X = pd.DataFrame({'SPY': spy_returns})

        beta_matrix = compute_time_weighted_robust_betas(Y, X, half_life=30)
        self.assertAlmostEqual(beta_matrix.loc['SPY', 'Asset1'], 1.2, delta=0.1)
        self.assertAlmostEqual(beta_matrix.loc['SPY', 'Asset2'], 0.8, delta=0.1)

    def test_lambda_parameter(self):
        """Test beta estimation using lambda_ instead of half_life."""
        np.random.seed(42)
        n_timestamps = 100
        spy_returns = np.random.normal(0, 0.01, n_timestamps)
        asset_returns = 1.5 * spy_returns + np.random.normal(0, 0.005, n_timestamps)

        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': spy_returns})

        lambda_ = 0.985  # Corresponds to a half-life of ~46 days
        beta_matrix = compute_time_weighted_robust_betas(Y, X, lambda_=lambda_)
        self.assertAlmostEqual(beta_matrix.loc['SPY', 'Asset1'], 1.5, delta=0.1)

    def test_empty_data(self):
        """Test handling of empty input DataFrames."""
        Y = pd.DataFrame()
        X = pd.DataFrame()
        with self.assertRaises(ValueError):
            compute_time_weighted_robust_betas(Y, X, half_life=30)

    def test_mismatched_timestamps(self):
        """Test handling of Y and X with different row counts."""
        np.random.seed(42)
        spy_returns = np.random.normal(0, 0.01, 100)
        asset_returns = 1.5 * spy_returns[:90] + np.random.normal(0, 0.005, 90)

        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': spy_returns})

        with self.assertRaises(ValueError):
            compute_time_weighted_robust_betas(Y, X, half_life=30)

    def test_invalid_half_life(self):
        """Test handling of negative or zero half_life."""
        np.random.seed(42)
        n_timestamps = 100
        spy_returns = np.random.normal(0, 0.01, n_timestamps)
        asset_returns = 1.5 * spy_returns

        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': spy_returns})

        with self.assertRaises(ValueError):
            compute_time_weighted_robust_betas(Y, X, half_life=0)
        with self.assertRaises(ValueError):
            compute_time_weighted_robust_betas(Y, X, half_life=-1)

    def test_invalid_lambda(self):
        """Test handling of invalid lambda_ values."""
        np.random.seed(42)
        n_timestamps = 100
        spy_returns = np.random.normal(0, 0.01, n_timestamps)
        asset_returns = 1.5 * spy_returns + np.random.normal(0, 0.005, n_timestamps)

        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': spy_returns})

        with self.assertRaises(ValueError):
            compute_time_weighted_robust_betas(Y, X, lambda_=0)
        with self.assertRaises(ValueError):
            compute_time_weighted_robust_betas(Y, X, lambda_=1.5)

    def test_no_variation_in_x(self):
        """Test handling of X with zero variance."""
        np.random.seed(42)
        n_timestamps = 100
        spy_returns = np.ones(n_timestamps) * 0.01  # Constant returns
        asset_returns = 1.5 * spy_returns + np.random.normal(0, 0.005, n_timestamps)

        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': spy_returns})

        with self.assertRaises(Exception):  # Expect numerical instability
            compute_time_weighted_robust_betas(Y, X, half_life=30)

    def test_outlier_heavy_data(self):
        """Test robustness with significant outliers."""
        np.random.seed(42)
        n_timestamps = 100
        spy_returns = np.random.normal(0, 0.01, n_timestamps)
        asset_returns = 1.5 * spy_returns + np.random.normal(0, 0.005, n_timestamps)
        asset_returns[::10] = asset_returns[::10] + 0.1  # Add large outliers every 10th point

        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': spy_returns})

        beta_matrix = compute_time_weighted_robust_betas(Y, X, half_life=30)
        self.assertAlmostEqual(beta_matrix.loc['SPY', 'Asset1'], 1.5, delta=0.2)  # Allow larger delta due to outliers

    def test_short_time_series(self):
        """Test with a small number of timestamps."""
        np.random.seed(42)
        n_timestamps = 5
        spy_returns = np.random.normal(0, 0.01,5)
        asset_returns = 1.5 * spy_returns + np.random.normal(0, 0.005, n_timestamps)

        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': spy_returns})

        beta_matrix = compute_time_weighted_robust_betas(Y, X, half_life=2)
        self.assertAlmostEqual(beta_matrix.loc['SPY', 'Asset1'], 1.5, delta=0.3)  # Larger delta due to small sample

    def test_multiple_factors(self):
        """Test with multiple factors in X."""
        np.random.seed(42)
        n_timestamps = 100
        spy_returns = np.random.normal(0, 0.01, n_timestamps)
        iwm_returns = np.random.normal(0, 0.01, n_timestamps)
        asset_returns = 1.2 * spy_returns + 0.5 * iwm_returns + np.random.normal(0, 0.005, n_timestamps)

        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': spy_returns, 'IWM': iwm_returns})

        beta_matrix = compute_time_weighted_robust_betas(Y, X, half_life=30)
        self.assertAlmostEqual(beta_matrix.loc['SPY', 'Asset1'], 1.2, delta=0.1)
        self.assertAlmostEqual(beta_matrix.loc['IWM', 'Asset1'], 0.5, delta=0.1)

    def test_insufficient_timestamps(self):
        """Test handling of insufficient timestamps after cleaning."""
        np.random.seed(42)
        n_timestamps = 5
        spy_returns = np.random.normal(0, 0.01, n_timestamps)
        asset_returns = 1.5 * spy_returns + np.random.normal(0, 0.005, n_timestamps)

        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': spy_returns})

        with self.assertRaises(ValueError):
            compute_time_weighted_robust_betas(Y, X, half_life=2, min_timestamps=10)

if __name__ == '__main__':
    unittest.main()