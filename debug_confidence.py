#!/usr/bin/env python3
"""
Debug script to test confidence calculation with actual log values
"""
import numpy as np
from utils.volatility_adjusted_confidence import (
    compute_heuristic_confidence,
    get_signal_quality_metrics,
)

# From your logs at 23:17:10
test_cases = [
    {
        "symbol": "ENAUSDT",
        "hist": np.array([0.000027] * 50),
        "hist_value": 0.000027,
        "closes": np.array([0.12] * 50),
        "regime": "normal",
    },
    {
        "symbol": "BCHUSDT",
        "hist": np.array([0.089145] * 50),
        "hist_value": 0.089145,
        "closes": np.array([462.66] * 50),
        "regime": "normal",
    },
    {
        "symbol": "PENGUUSDT",
        "hist": np.array([-0.000001] * 50),
        "hist_value": -0.000001,
        "closes": np.array([0.01] * 50),
        "regime": "normal",
    },
]

print("="*80)
print("Testing Confidence Calculation with Actual Log Values")
print("="*80)

for case in test_cases:
    print(f"\n[{case['symbol']}]")
    print(f"  hist_value: {case['hist_value']:.6f}")
    print(f"  closes: {case['closes'][0]:.2f} (all same)")
    print(f"  regime: {case['regime']}")
    
    # Compute confidence
    h_conf = compute_heuristic_confidence(
        hist_value=case['hist_value'],
        hist_values=case['hist'],
        regime=case['regime'],
        closes=case['closes'],
    )
    
    print(f"  → h_conf: {h_conf:.3f}")
    
    # Get metrics
    metrics = get_signal_quality_metrics(
        hist_values=case['hist'],
        regime=case['regime'],
    )
    
    print(f"  Metrics:")
    print(f"    magnitude: {metrics.get('histogram_magnitude', 0):.4f}")
    print(f"    accel: {metrics.get('histogram_acceleration', 0):.4f}")
    print(f"    raw_conf: {metrics.get('raw_confidence', 0):.3f}")
    print(f"    adjusted_conf: {metrics.get('adjusted_confidence', 0):.3f}")
    print(f"    floor: {metrics.get('regime_floor', 0):.2f}")
    print(f"    final_conf: {metrics.get('final_confidence', 0):.3f}")

print("\n" + "="*80)
print("Expected: h_conf should NOT be 0.5 for non-zero histogram values")
print("="*80)
