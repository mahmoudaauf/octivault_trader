#!/usr/bin/env python3
"""
DYNAMIC BALANCE THRESHOLD CONFIGURATION
========================================

Scales bucket classification thresholds based on actual account balance.
This ensures the 3-bucket system works correctly regardless of account size.

Author: System Calibration
Date: 2026-04-26
"""

from typing import Dict, Tuple
    """Calculate bucket classification thresholds based on account balance"""
    
    # Percentage-based ranges (relative to initial balance)
    # For small accounts ($100-1000): Use tighter thresholds (0.1%)
    # For medium accounts ($1000-100k): Use moderate thresholds (0.2%)
    # For large accounts (100k+): Use loose thresholds (0.5%)
    
    @staticmethod
    def get_threshold_percentages(initial_balance: float) -> Tuple[float, float, float]:
        """Get appropriate threshold percentages based on account size"""
        if initial_balance < 1000:
            # Small accounts: tight thresholds
            return (0.01, 0.001, -0.01)    # ±1% GAINING/LOSING, ±0.1% STABLE
        elif initial_balance < 100000:
            # Medium accounts: moderate thresholds
            return (0.02, 0.005, -0.02)    # ±0.2% GAINING/LOSING, ±0.5% STABLE
        else:
            # Large accounts: loose thresholds
            return (0.05, 0.01, -0.05)     # ±0.5% GAINING/LOSING, ±1% STABLE
    
    @staticmethod
    def calculate_thresholds(initial_balance: float) -> Dict[str, Dict[str, float]]:
        """
        Calculate bucket thresholds based on initial balance.
        
        Args:
            initial_balance: Current account balance (e.g., 104.04)
        
        Returns:
            Dict with threshold information for each bucket
        """
        
        gaining_threshold = initial_balance * DynamicBalanceThresholds.GAINING_THRESHOLD
        stable_band = initial_balance * DynamicBalanceThresholds.STABLE_BAND
        losing_threshold = initial_balance * DynamicBalanceThresholds.LOSING_THRESHOLD
        
        return {
            "GAINING": {
                "min_balance": initial_balance + gaining_threshold,
                "description": f"Balance > ${initial_balance + gaining_threshold:.2f}"
            },
            "STABLE": {
                "min_balance": initial_balance - stable_band,
                "max_balance": initial_balance + stable_band,
                "description": f"Balance between ${initial_balance - stable_band:.2f} - ${initial_balance + stable_band:.2f}"
            },
            "LOSING": {
                "max_balance": initial_balance + losing_threshold,
                "description": f"Balance < ${initial_balance + losing_threshold:.2f}"
            }
        }
    
    @staticmethod
    def classify_balance(current_balance: float, initial_balance: float) -> Tuple[str, float]:
        """
        Classify current balance into a bucket.
        
        Args:
            current_balance: Current account balance
            initial_balance: Starting account balance
        
        Returns:
            Tuple of (bucket_name, percentage_change)
        """
        
        pct_change = ((current_balance - initial_balance) / initial_balance) * 100
        
        gaining_threshold = initial_balance * DynamicBalanceThresholds.GAINING_THRESHOLD
        stable_band = initial_balance * DynamicBalanceThresholds.STABLE_BAND
        losing_threshold = initial_balance * DynamicBalanceThresholds.LOSING_THRESHOLD
        
        if current_balance > (initial_balance + gaining_threshold):
            return ("GAINING 📈", pct_change)
        elif current_balance < (initial_balance + losing_threshold):
            return ("LOSING 📉", pct_change)
        elif (initial_balance - stable_band) <= current_balance <= (initial_balance + stable_band):
            return ("STABLE ➡️", pct_change)
        else:
            # Default to STABLE if between LOSING and GAINING
            return ("STABLE ➡️", pct_change)
    
    @staticmethod
    def get_classification_ranges(initial_balance: float) -> Dict[str, str]:
        """
        Get human-readable ranges for each bucket.
        
        Args:
            initial_balance: Starting account balance
        
        Returns:
            Dict with readable ranges
        """
        
        thresholds = DynamicBalanceThresholds.calculate_thresholds(initial_balance)
        
        return {
            "GAINING 📈": f"> ${thresholds['GAINING']['min_balance']:.2f} (+0.2% above)",
            "STABLE ➡️": f"${thresholds['STABLE']['min_balance']:.2f} - ${thresholds['STABLE']['max_balance']:.2f} (±0.5%)",
            "LOSING 📉": f"< ${thresholds['LOSING']['max_balance']:.2f} (-0.2% below)"
        }


# Example usage for $104.04 account
if __name__ == "__main__":
    initial = 104.04
    
    print(f"🎯 DYNAMIC BALANCE CLASSIFICATION - Account: ${initial:.2f}")
    print("=" * 80)
    
    thresholds = DynamicBalanceThresholds.calculate_thresholds(initial)
    print("\n📊 BUCKET THRESHOLDS:")
    for bucket, info in thresholds.items():
        print(f"\n{bucket}:")
        for key, value in info.items():
            if isinstance(value, float):
                print(f"  {key}: ${value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    print("\n\n📈 CLASSIFICATION RANGES:")
    ranges = DynamicBalanceThresholds.get_classification_ranges(initial)
    for bucket, range_str in ranges.items():
        print(f"  {bucket}: {range_str}")
    
    # Test classification
    print("\n\n🧪 TEST CLASSIFICATIONS:")
    test_balances = [
        (104.04, "Initial balance"),
        (104.26, "Small gain (+0.2%)"),
        (104.48, "Larger gain (+0.4%)"),
        (104.12, "Tiny gain (+0.08%)"),
        (104.00, "Slight loss (-0.04%)"),
        (103.82, "Loss (-0.2%)"),
        (103.60, "Larger loss (-0.4%)"),
    ]
    
    for balance, description in test_balances:
        bucket, pct = DynamicBalanceThresholds.classify_balance(balance, initial)
        print(f"  ${balance:.2f} ({description:20}) → {bucket} ({pct:+.2f}%)")
    
    print("\n✅ Dynamic threshold system ready!")
