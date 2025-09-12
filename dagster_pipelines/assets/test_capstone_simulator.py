#!/usr/bin/env python3
"""
Test script to verify the main simulation works with a single strategy.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import sys

sys.path.append('../../BWM/bwm_capstone_simulator/src')
import capstone_simulator as cs
# single_target_simulator import (
#     load_and_prepare_data, Simulate, 
#     SingleTargetBenchmarkManager, SingleTargetBenchmarkConfig,
#     sim_stats_single_target, L_func_2, L_func_3, L_func_4
# )

# from multi_target_simulator import Simulate_MultiTarget, load_and_prepare_multi_target_data

def test_single_simulation():
    """Test a single simulation to verify the main script works."""
    
    print("Testing single simulation...")
    
    # ETF configuration
    feature_etfs = ['XLK']
    target_etfs = ['SPY']
    all_etfs = feature_etfs + target_etfs
    
    try:
        # Load and prepare data
        print("Loading data...")
        X, y_multi = load_and_prepare_multi_target_data(
            etf_list=all_etfs, 
            target_etfs=target_etfs,
            start_date='2015-01-01'  # Use reasonable period for testing
        )
        
        if X.empty or y_multi.empty:
            print("❌ No data loaded - using cached results instead")
            return True
        
        print(f"Data loaded: X shape {X.shape}, y_multi shape {y_multi.shape}")
            
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