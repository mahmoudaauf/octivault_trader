#!/usr/bin/env python3
"""
Quick test to verify the hardcoded confidence fix
"""
import numpy as np
import sys

def test_import():
    """Test that volatility_adjusted_confidence imports correctly"""
    print("[TEST 1] Testing module import...")
    try:
        from utils.volatility_adjusted_confidence import (
            compute_heuristic_confidence,
            get_signal_quality_metrics,
            compute_histogram_magnitude,
        )
        print("  ✅ Import successful (talib optional)")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_chop_magnitude():
    """Test magnitude calculation for near-zero histogram (chop)"""
    print("\n[TEST 2] Testing magnitude calculation in chop mode...")
    try:
        from utils.volatility_adjusted_confidence import compute_histogram_magnitude
        
        # Simulate chop: all histogram values near zero
        hist_chop = np.array([-0.000051, -0.000048, -0.000052, 0.000001, -0.000033] * 10)
        mag_chop = compute_histogram_magnitude(hist_chop)
        
        print(f"  Chop histogram: {hist_chop[:5]}...")
        print(f"  Magnitude: {mag_chop:.4f}")
        
        if 0.0 < mag_chop < 0.3:
            print("  ✅ Magnitude in expected range [0.0, 0.3]")
            return True
        else:
            print(f"  ❌ Magnitude {mag_chop:.4f} outside expected range [0.0, 0.3]")
            return False
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False

def test_trend_magnitude():
    """Test magnitude calculation for strong signal (trend)"""
    print("\n[TEST 3] Testing magnitude calculation in trend mode...")
    try:
        from utils.volatility_adjusted_confidence import compute_histogram_magnitude
        
        # Simulate trend: histogram values grow stronger
        hist_trend = np.array([0.001, 0.002, 0.003, 0.004, 0.005] * 10)
        mag_trend = compute_histogram_magnitude(hist_trend)
        
        print(f"  Trend histogram: {hist_trend[:5]}...")
        print(f"  Magnitude: {mag_trend:.4f}")
        
        if mag_trend > 0.5:
            print("  ✅ Magnitude shows strong signal (> 0.5)")
            return True
        else:
            print(f"  ❌ Magnitude {mag_trend:.4f} should be > 0.5 for trend")
            return False
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False

def test_confidence_not_hardcoded():
    """Test that confidence varies, not hardcoded to 0.5"""
    print("\n[TEST 4] Testing that confidence varies (not hardcoded 0.5)...")
    try:
        from utils.volatility_adjusted_confidence import compute_heuristic_confidence
        
        # Test 1: Weak signal (chop)
        hist_weak = np.array([-0.000051] * 50)
        conf_weak = compute_heuristic_confidence(
            hist_value=-0.000051,
            hist_values=hist_weak,
            regime="normal",
        )
        
        # Test 2: Strong signal (trend)
        hist_strong = np.array([0.005] * 50)
        conf_strong = compute_heuristic_confidence(
            hist_value=0.005,
            hist_values=hist_strong,
            regime="normal",
        )
        
        print(f"  Weak signal confidence: {conf_weak:.3f}")
        print(f"  Strong signal confidence: {conf_strong:.3f}")
        
        if conf_weak != conf_strong and conf_weak != 0.5 and conf_strong != 0.5:
            print("  ✅ Confidence varies by signal strength (not hardcoded 0.5)")
            return True
        elif conf_weak == conf_strong == 0.5:
            print("  ❌ Confidence is hardcoded to 0.5 (fallback still active)")
            return False
        else:
            print("  ⚠️ Confidence varies but with unexpected values")
            return False
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False

def test_metrics_calculated():
    """Test that signal quality metrics are properly calculated"""
    print("\n[TEST 5] Testing signal quality metrics calculation...")
    try:
        from utils.volatility_adjusted_confidence import get_signal_quality_metrics
        
        hist = np.array([-0.000051] * 50)
        metrics = get_signal_quality_metrics(hist, regime="normal")
        
        print(f"  Metrics keys: {list(metrics.keys())}")
        print(f"  Histogram magnitude: {metrics.get('histogram_magnitude', 'N/A')}")
        print(f"  Final confidence: {metrics.get('final_confidence', 'N/A')}")
        
        if 'histogram_magnitude' in metrics and 'final_confidence' in metrics:
            mag = metrics['histogram_magnitude']
            conf = metrics['final_confidence']
            
            if mag > 0 and 0 < conf < 1:
                print("  ✅ Metrics calculated correctly")
                return True
            else:
                print(f"  ❌ Metrics have unexpected values: mag={mag}, conf={conf}")
                return False
        else:
            print("  ❌ Missing expected metrics")
            return False
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False

def main():
    print("="*80)
    print("VERIFICATION: Hardcoded Confidence Fix")
    print("="*80)
    
    tests = [
        test_import,
        test_chop_magnitude,
        test_trend_magnitude,
        test_confidence_not_hardcoded,
        test_metrics_calculated,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "="*80)
    print(f"SUMMARY: {sum(results)}/{len(results)} tests passed")
    print("="*80)
    
    if all(results):
        print("✅ All tests passed! Fix is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
