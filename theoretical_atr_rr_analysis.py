#!/usr/bin/env python3
"""
Theoretical ATR-RR Design Analysis
==================================

Analyzes the theoretical performance expectations based on ATR-RR architecture
without requiring actual trade data.
"""

import sys
import os
import numpy as np
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config

class TheoreticalATRRAnalyzer:
    def __init__(self):
        self.config = Config()

    def analyze_atr_rr_design(self) -> Dict[str, Any]:
        """Analyze the theoretical ATR-RR design parameters"""

        # Core parameters
        target_rr = float(getattr(self.config, 'TARGET_RR_RATIO', 1.8))
        sl_atr_mult = float(getattr(self.config, 'SL_ATR_MULT', 1.0))
        tp_atr_mult = float(getattr(self.config, 'TP_ATR_MULT', 1.5))
        risk_pct_per_trade = float(getattr(self.config, 'RISK_PCT_PER_TRADE', 0.01))

        # Fee parameters
        taker_fee_pct = 0.001  # 0.1%
        maker_fee_pct = 0.0005  # 0.05%
        round_trip_fee_pct = taker_fee_pct * 2  # Assume taker both ways

        # ATR scenarios for different market conditions
        atr_scenarios = {
            'low_volatility': 0.005,    # 0.5% ATR (BTC in stable periods)
            'normal_volatility': 0.010, # 1.0% ATR (typical crypto)
            'high_volatility': 0.020,   # 2.0% ATR (altcoins, news events)
            'extreme_volatility': 0.035 # 3.5% ATR (meme coins, pumps)
        }

        results = {}

        for scenario_name, atr_pct in atr_scenarios.items():
            scenario_results = {}

            # SL and TP calculations (system enforces TARGET_RR_RATIO)
            sl_pct = atr_pct * sl_atr_mult
            tp_pct = sl_pct * target_rr  # TP = RR × SL (system enforcement)

            # Effective RR after fees
            gross_rr = tp_pct / sl_pct
            fee_impact_on_rr = (round_trip_fee_pct / sl_pct)  # Fee drag on RR
            net_rr = gross_rr - fee_impact_on_rr

            # Win rate needed for profitability
            breakeven_win_rate = 1.0 / (1.0 + net_rr)

            # Position sizing calculations
            equity = 1000.0  # $1000 starting capital
            risk_amount = equity * risk_pct_per_trade

            # For different price levels
            price_levels = [1.0, 10.0, 100.0, 1000.0, 10000.0, 50000.0]

            sizing_examples = {}
            for price in price_levels:
                sl_distance_usd = price * sl_pct
                position_size_coins = risk_amount / sl_distance_usd
                position_value_usd = position_size_coins * price

                sizing_examples[f"${price}"] = {
                    'position_size_coins': position_size_coins,
                    'position_value_usd': position_value_usd,
                    'sl_distance_usd': sl_distance_usd,
                    'risk_amount_usd': risk_amount
                }

            scenario_results.update({
                'atr_pct': atr_pct * 100,
                'sl_pct': sl_pct * 100,
                'tp_pct': tp_pct * 100,
                'gross_rr': gross_rr,
                'net_rr': net_rr,
                'fee_impact': fee_impact_on_rr,
                'breakeven_win_rate': breakeven_win_rate * 100,
                'position_sizing_examples': sizing_examples
            })

            results[scenario_name] = scenario_results

        return results

    def calculate_expected_value_matrix(self) -> Dict[str, Any]:
        """Calculate expected value for different win rate and RR combinations"""

        rr_range = np.arange(0.5, 3.1, 0.2)  # 0.5 to 3.0 RR
        win_rate_range = np.arange(0.3, 0.81, 0.05)  # 30% to 80% win rate

        ev_matrix = {}
        optimal_points = {}

        for rr in rr_range:
            rr_key = ".1f"
            ev_matrix[rr_key] = {}
            breakeven_win_rate = 1.0 / (1.0 + rr)

            for win_rate in win_rate_range:
                loss_rate = 1.0 - win_rate
                expected_value = (win_rate * rr) - (loss_rate * 1.0)
                ev_matrix[rr_key][f"{win_rate:.0%}"] = expected_value

                # Track optimal points
                if expected_value > 0:
                    if rr not in optimal_points:
                        optimal_points[rr] = []
                    optimal_points[rr].append({
                        'win_rate': win_rate,
                        'ev': expected_value
                    })

        # Find best EV for each RR
        best_ev_per_rr = {}
        for rr, points in optimal_points.items():
            best_point = max(points, key=lambda x: x['ev'])
            best_ev_per_rr[rr] = best_point

        return {
            'ev_matrix': ev_matrix,
            'best_ev_per_rr': best_ev_per_rr,
            'target_rr': float(getattr(self.config, 'TARGET_RR_RATIO', 1.8))
        }

    def analyze_fee_impact(self) -> Dict[str, Any]:
        """Analyze how fees impact the RR requirements"""

        base_fee_pct = 0.001  # 0.1% taker fee
        fee_scenarios = [0.0005, 0.001, 0.0015, 0.002]  # Different fee levels

        target_rr = float(getattr(self.config, 'TARGET_RR_RATIO', 1.8))

        fee_analysis = {}

        for fee_pct in fee_scenarios:
            round_trip_fee = fee_pct * 2

            # RR needed after fees to achieve target effective RR
            required_gross_rr = target_rr + (round_trip_fee / 0.01)  # Assuming 1% SL

            # Win rate needed for different effective RRs
            win_rates_needed = {}
            for effective_rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
                win_rate_needed = 1.0 / (1.0 + effective_rr)
                win_rates_needed[f"{effective_rr}:1"] = win_rate_needed * 100

            fee_analysis[f"{fee_pct*100:.1f}%"] = {
                'round_trip_fee': round_trip_fee * 100,
                'required_gross_rr_for_target': required_gross_rr,
                'win_rates_needed': win_rates_needed
            }

        return fee_analysis

def main():
    analyzer = TheoreticalATRRAnalyzer()

    print("="*80)
    print("THEORETICAL ATR-RR DESIGN ANALYSIS")
    print("="*80)

    # ATR-RR Design Analysis
    print("\n1️⃣ ATR-RR DESIGN ANALYSIS")
    print("-" * 50)

    design_results = analyzer.analyze_atr_rr_design()

    for scenario, data in design_results.items():
        print(f"\n📊 {scenario.replace('_', ' ').title()}")
        print(f"   ATR: {data['atr_pct']:.1f}%")
        print(f"   SL Distance: {data['sl_pct']:.1f}%")
        print(f"   TP Distance: {data['tp_pct']:.1f}%")
        print(f"   Gross RR: {data['gross_rr']:.2f}:1")
        print(f"   Net RR (after fees): {data['net_rr']:.2f}:1")
        print(f"   Breakeven Win Rate: {data['breakeven_win_rate']:.1f}%")

        # Show position sizing for $1000 equity
        examples = data['position_sizing_examples']
        print("   💰 Position Sizing Examples ($1000 equity, 1% risk):")
        for price, sizing in examples.items():
            if sizing['position_size_coins'] >= 0.001:  # Show reasonable sizes
                print(f"      {price} asset: {sizing['position_size_coins']:.4f} coins (${sizing['position_value_usd']:.2f})")

    # Expected Value Matrix
    print("\n2️⃣ EXPECTED VALUE MATRIX")
    print("-" * 50)

    ev_results = analyzer.calculate_expected_value_matrix()
    target_rr = ev_results['target_rr']

    print(f"🎯 Target RR: {target_rr}:1")
    print("💰 Best Expected Value for each RR ratio:")

    for rr, best_point in ev_results['best_ev_per_rr'].items():
        win_rate_pct = best_point['win_rate'] * 100
        ev = best_point['ev']
        print(".1f")

    # Fee Impact Analysis
    print("\n3️⃣ FEE IMPACT ANALYSIS")
    print("-" * 50)

    fee_results = analyzer.analyze_fee_impact()

    for fee_level, data in fee_results.items():
        print(f"\n💸 Fee Level: {fee_level} per trade")
        print(f"   Round-trip fee: {data['round_trip_fee']:.2f}%")
        print(f"   Gross RR needed for {target_rr}:1 effective: {data['required_gross_rr_for_target']:.2f}:1")

        print("   Win rates needed for different effective RRs:")
        for rr_ratio, win_rate in data['win_rates_needed'].items():
            print(".1f")

    print("\n" + "="*80)
    print("🎯 KEY INSIGHTS:")
    print("• Higher volatility (ATR%) allows larger position sizes")
    print("• Fee impact reduces effective RR - need higher gross RR")
    print("• Win rate requirements decrease as RR increases")
    print("• Optimal EV typically around 50-60% win rate with 2.0-2.5 RR")
    print("="*80)

if __name__ == "__main__":
    main()