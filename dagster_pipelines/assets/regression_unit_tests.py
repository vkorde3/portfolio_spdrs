import numpy as np
import pandas as pd
import unittest
from robust_regression import compute_time_weighted_robust_betas

class TestComputeTimeWeightedRobustBetas(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set random seed and create common variables."""
        np.random.seed(42)
        cls.n_timestamps = 100
        cls.spy_returns = np.random.normal(0, 0.01, cls.n_timestamps)

    def test_single_asset(self):
        """Test beta estimation for a single ETF with known beta."""
        asset_returns = 1.5 * self.spy_returns + np.random.normal(0, 0.005, self.n_timestamps)
        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': self.spy_returns})
        beta_matrix = compute_time_weighted_robust_betas(Y, X, half_life=30)
        self.assertAlmostEqual(beta_matrix.loc['SPY', 'Asset1'], 1.5, delta=0.1)

    def test_multiple_assets(self):
        """Test beta estimation for multiple ETFs with known betas."""
        asset_returns_1 = 1.2 * self.spy_returns + np.random.normal(0, 0.005, self.n_timestamps)
        asset_returns_2 = 0.8 * self.spy_returns + np.random.normal(0, 0.005, self.n_timestamps)
        Y = pd.DataFrame({'Asset1': asset_returns_1, 'Asset2': asset_returns_2})
        X = pd.DataFrame({'SPY': self.spy_returns})
        beta_matrix = compute_time_weighted_robust_betas(Y, X, half_life=30)
        self.assertAlmostEqual(beta_matrix.loc['SPY', 'Asset1'], 1.2, delta=0.1)
        self.assertAlmostEqual(beta_matrix.loc['SPY', 'Asset2'], 0.8, delta=0.1)

    def test_multiple_factors(self):
        """Test beta estimation with multiple factors in X."""
        iwm_returns = np.random.normal(0, 0.01, self.n_timestamps)
        asset_returns = 1.2 * self.spy_returns + 0.5 * iwm_returns + np.random.normal(0, 0.005, self.n_timestamps)
        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': self.spy_returns, 'IWM': iwm_returns})
        beta_matrix = compute_time_weighted_robust_betas(Y, X, half_life=30)
        self.assertAlmostEqual(beta_matrix.loc['SPY', 'Asset1'], 1.2, delta=0.1)
        self.assertAlmostEqual(beta_matrix.loc['IWM', 'Asset1'], 0.5, delta=0.1)

    def test_lambda_parameter(self):
        """Test beta estimation using lambda_ instead of half_life."""
        asset_returns = 1.5 * self.spy_returns + np.random.normal(0, 0.005, self.n_timestamps)
        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': self.spy_returns})
        beta_matrix = compute_time_weighted_robust_betas(Y, X, lambda_=0.985)
        self.assertAlmostEqual(beta_matrix.loc['SPY', 'Asset1'], 1.5, delta=0.1)

    def test_empty_data(self):
        """Test handling of empty input DataFrames."""
        Y = pd.DataFrame()
        X = pd.DataFrame()
        with self.assertRaises(ValueError):
            compute_time_weighted_robust_betas(Y, X, half_life=30)

    def test_mismatched_timestamps(self):
        """Test handling of Y and X with different row counts."""
        asset_returns = 1.5 * self.spy_returns[:90] + np.random.normal(0, 0.005, 90)
        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': self.spy_returns})
        with self.assertRaises(ValueError):
            compute_time_weighted_robust_betas(Y, X, half_life=30)

    def test_invalid_half_life(self):
        """Test handling of negative or zero half_life."""
        asset_returns = 1.5 * self.spy_returns
        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': self.spy_returns})
        for invalid_half_life in [0, -1]:
            with self.assertRaises(ValueError):
                compute_time_weighted_robust_betas(Y, X, half_life=invalid_half_life)

    def test_invalid_lambda(self):
        """Test handling of invalid lambda_ values."""
        asset_returns = 1.5 * self.spy_returns + np.random.normal(0, 0.005, self.n_timestamps)
        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': self.spy_returns})
        for invalid_lambda in [0, 1.5]:
            with self.assertRaises(ValueError):
                compute_time_weighted_robust_betas(Y, X, lambda_=invalid_lambda)

    def test_no_variation_in_x(self):
        """Test handling of X with zero variance."""
        spy_constant = np.ones(self.n_timestamps) * 0.01  # Constant returns
        asset_returns = 1.5 * spy_constant + np.random.normal(0, 0.005, self.n_timestamps)
        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': spy_constant})
        with self.assertRaises(ValueError):
            compute_time_weighted_robust_betas(Y, X, half_life=30)

    def test_outlier_heavy_data(self):
        """Test robustness with significant outliers."""
        asset_returns = 1.5 * self.spy_returns + np.random.normal(0, 0.005, self.n_timestamps)
        asset_returns[::10] += 0.1  # Add large outliers every 10th point
        Y = pd.DataFrame({'Asset1': asset_returns})
        X = pd.DataFrame({'SPY': self.spy_returns})
        beta_matrix = compute_time_weighted_robust_betas(Y, X, half_life=30)
        self.assertAlmostEqual(beta_matrix.loc['SPY', 'Asset1'], 1.5, delta=0.2)  # Allow larger delta due to outliers

    def test_insufficient_timestamps(self):
        """Test handling of insufficient timestamps after cleaning."""
        short_spy_returns = np.random.normal(0, 0.01, 5)
        short_asset_returns = 1.5 * short_spy_returns + np.random.normal(0, 0.005, 5)
        Y = pd.DataFrame({'Asset1': short_asset_returns})
        X = pd.DataFrame({'SPY': short_spy_returns})
        with self.assertRaises(ValueError):
            compute_time_weighted_robust_betas(Y, X, half_life=2, min_timestamps=10)

if __name__ == '__main__':
    unittest.main()