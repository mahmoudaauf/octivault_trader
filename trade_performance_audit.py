#!/usr/bin/env python3
"""
Trade Performance Audit & RR Analysis
=====================================

Analyzes real trade distribution from SharedState to:
1. Audit real trade distribution from logs/data
2. Calculate real realized RR ratios
3. Estimate actual win rate
4. Simulate expected value
5. Compare against theoretical ATR-RR design
"""

import sys
import os
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config
from core.shared_state import SharedState

class TradePerformanceAuditor:
    def __init__(self):
        self.config = Config()
        self.shared_state = SharedState()
        self.completed_trades = []

    async def extract_completed_trades(self) -> List[Dict[str, Any]]:
        """Extract completed trades with entry/exit data from SharedState"""
        trades = []

        try:
            # Get positions that have been closed
            positions = getattr(self.shared_state, 'positions', {}) or {}
            if isinstance(positions, dict):
                for symbol, pos_data in positions.items():
                    if isinstance(pos_data, dict):
                        status = str(pos_data.get('status', '')).upper()
                        if status == 'CLOSED':
                            # This is a completed trade
                            trade = {
                                'symbol': symbol,
                                'entry_price': float(pos_data.get('entry_price', 0.0) or 0.0),
                                'exit_price': float(pos_data.get('closed_price', 0.0) or 0.0),
                                'quantity': float(pos_data.get('closed_qty', 0.0) or 0.0),
                                'entry_time': float(pos_data.get('opened_at', 0.0) or 0.0),
                                'exit_time': float(pos_data.get('closed_at', 0.0) or 0.0),
                                'pnl': float(pos_data.get('realized_pnl', 0.0) or 0.0),
                                'reason': str(pos_data.get('closed_reason', '') or ''),
                                'tag': str(pos_data.get('closed_tag', '') or '')
                            }

                            if trade['entry_price'] > 0 and trade['exit_price'] > 0:
                                trades.append(trade)

        except Exception as e:
            print(f"Error extracting completed trades: {e}")

        return trades

    def calculate_rr_ratios(self, trades: List[Dict[str, Any]]) -> List[float]:
        """Calculate risk-reward ratios for completed trades"""
        rr_ratios = []

        for trade in trades:
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            quantity = trade['quantity']

            if entry_price <= 0 or exit_price <= 0 or quantity <= 0:
                continue

            # Calculate position value at entry
            entry_value = entry_price * quantity

            # Calculate exit value (assuming market order fill at exit_price)
            exit_value = exit_price * quantity

            # Calculate gross P&L
            gross_pnl = exit_value - entry_value

            # Estimate risk (SL distance based on ATR)
            # Use theoretical ATR-based SL calculation
            atr_pct = self._estimate_atr_pct(trade['symbol'])
            sl_distance_pct = atr_pct * float(getattr(self.config, 'SL_ATR_MULT', 1.0))

            # Risk amount = entry_value * sl_distance_pct
            risk_amount = entry_value * sl_distance_pct

            if risk_amount > 0:
                rr_ratio = abs(gross_pnl) / risk_amount
                rr_ratios.append(rr_ratio)

        return rr_ratios

    def _estimate_atr_pct(self, symbol: str) -> float:
        """Estimate ATR percentage for a symbol (simplified)"""
        # This is a rough estimate - in reality would use actual ATR calculation
        symbol_upper = symbol.upper()

        # Rough ATR estimates based on typical crypto volatility
        atr_estimates = {
            'BTC': 0.005,   # 0.5%
            'ETH': 0.008,   # 0.8%
            'BNB': 0.010,   # 1.0%
            'ADA': 0.015,   # 1.5%
            'SOL': 0.012,   # 1.2%
            'DOT': 0.018,   # 1.8%
            'DOGE': 0.025,  # 2.5%
            'SHIB': 0.030,  # 3.0%
        }

        # Default to 1.5% ATR
        base_atr = 0.015

        for token, atr in atr_estimates.items():
            if token in symbol_upper:
                base_atr = atr
                break

        return base_atr

    def analyze_win_rate(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze win rate from completed trades"""
        if not trades:
            return {'win_rate': 0.0, 'total_trades': 0, 'wins': 0, 'losses': 0}

        wins = 0
        losses = 0

        for trade in trades:
            pnl = trade.get('pnl', 0.0)
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
            # Skip break-even trades

        total_decisive = wins + losses
        win_rate = (wins / total_decisive) * 100 if total_decisive > 0 else 0.0

        return {
            'win_rate': win_rate,
            'total_trades': len(trades),
            'decisive_trades': total_decisive,
            'wins': wins,
            'losses': losses,
            'break_even': len(trades) - total_decisive
        }

    def simulate_expected_value(self, trades: List[Dict[str, Any]], rr_ratios: List[float]) -> Dict[str, Any]:
        """Simulate expected value based on current performance"""
        if not trades or not rr_ratios:
            return {'expected_value': 0.0, 'theoretical_ev': 0.0, 'actual_ev': 0.0}

        win_rate_stats = self.analyze_win_rate(trades)
        win_rate = win_rate_stats['win_rate'] / 100.0

        # Calculate average RR
        avg_rr = np.mean(rr_ratios) if rr_ratios else 0.0

        # Expected value = (Win Rate × Avg RR) - (Loss Rate × 1.0)
        loss_rate = 1.0 - win_rate
        expected_value = (win_rate * avg_rr) - (loss_rate * 1.0)

        # Theoretical EV based on target RR
        target_rr = float(getattr(self.config, 'TARGET_RR_RATIO', 1.8))
        theoretical_win_rate_needed = 1.0 / (1.0 + target_rr)
        theoretical_ev = (theoretical_win_rate_needed * target_rr) - ((1.0 - theoretical_win_rate_needed) * 1.0)

        return {
            'expected_value': expected_value,
            'theoretical_ev': theoretical_ev,
            'actual_win_rate': win_rate * 100,
            'theoretical_win_rate_needed': theoretical_win_rate_needed * 100,
            'average_rr': avg_rr,
            'target_rr': target_rr
        }

    def compare_to_theoretical_design(self, trades: List[Dict[str, Any]], rr_ratios: List[float]) -> Dict[str, Any]:
        """Compare actual performance to theoretical ATR-RR design"""
        analysis = {}

        if not rr_ratios:
            return {'status': 'NO_DATA', 'message': 'No RR data available'}

        # Theoretical targets
        target_rr = float(getattr(self.config, 'TARGET_RR_RATIO', 1.8))
        target_win_rate_needed = 1.0 / (1.0 + target_rr)

        # Actual metrics
        actual_avg_rr = np.mean(rr_ratios)
        win_stats = self.analyze_win_rate(trades)
        actual_win_rate = win_stats['win_rate'] / 100.0

        # RR distribution analysis
        rr_quartiles = np.percentile(rr_ratios, [25, 50, 75])

        analysis['rr_distribution'] = {
            'mean': actual_avg_rr,
            'median': rr_quartiles[1],
            'q25': rr_quartiles[0],
            'q75': rr_quartiles[2],
            'target': target_rr
        }

        analysis['win_rate_analysis'] = {
            'actual': actual_win_rate * 100,
            'theoretical_needed': target_win_rate_needed * 100,
            'gap': (actual_win_rate - target_win_rate_needed) * 100
        }

        # Performance assessment
        rr_achievement = actual_avg_rr / target_rr
        win_rate_achievement = actual_win_rate / target_win_rate_needed

        analysis['performance_score'] = {
            'rr_achievement': rr_achievement,
            'win_rate_achievement': win_rate_achievement,
            'overall_score': (rr_achievement + win_rate_achievement) / 2.0
        }

        # Risk assessment
        if actual_win_rate < target_win_rate_needed:
            analysis['risk_assessment'] = 'HIGH_RISK'
            analysis['recommendation'] = 'Win rate too low for target RR. Consider reducing RR target or improving entry quality.'
        elif actual_avg_rr < target_rr * 0.8:
            analysis['risk_assessment'] = 'MODERATE_RISK'
            analysis['recommendation'] = 'RR ratios below target. Check SL/TP placement and ATR calculations.'
        else:
            analysis['risk_assessment'] = 'LOW_RISK'
            analysis['recommendation'] = 'Performance aligned with theoretical design. Continue monitoring.'

        return analysis

    async def run_audit(self) -> Dict[str, Any]:
        """Run complete trade performance audit"""
        print("🔍 Starting Trade Performance Audit...")

        # Extract trade data
        trades = await self.extract_completed_trades()
        print(f"📊 Found {len(trades)} completed trades")

        if not trades:
            return {
                'status': 'NO_TRADES',
                'message': 'No completed trades found for analysis'
            }

        # Calculate RR ratios
        rr_ratios = self.calculate_rr_ratios(trades)
        print(f"📈 Calculated RR ratios for {len(rr_ratios)} trades")

        # Analyze win rate
        win_stats = self.analyze_win_rate(trades)
        print(f"🎯 Win Rate: {win_stats['win_rate']:.1f}% ({win_stats['wins']}W / {win_stats['losses']}L)")

        # Simulate expected value
        ev_analysis = self.simulate_expected_value(trades, rr_ratios)
        print(f"💰 Expected Value: {ev_analysis['expected_value']:.3f}")
        print(f"🎯 Theoretical EV: {ev_analysis['theoretical_ev']:.3f}")

        # Compare to theoretical design
        comparison = self.compare_to_theoretical_design(trades, rr_ratios)

        # Compile results
        audit_results = {
            'status': 'COMPLETED',
            'trade_count': len(trades),
            'win_rate_stats': win_stats,
            'rr_analysis': {
                'ratios': rr_ratios,
                'average': np.mean(rr_ratios) if rr_ratios else 0.0,
                'distribution': comparison.get('rr_distribution', {})
            },
            'expected_value': ev_analysis,
            'theoretical_comparison': comparison,
            'recommendations': comparison.get('recommendation', 'Continue monitoring performance')
        }

        return audit_results

async def main():
    auditor = TradePerformanceAuditor()
    results = await auditor.run_audit()

    print("\n" + "="*60)
    print("TRADE PERFORMANCE AUDIT RESULTS")
    print("="*60)

    if results['status'] == 'NO_TRADES':
        print("❌ No trades found for analysis")
        return

    print(f"📊 Total Trades Analyzed: {results['trade_count']}")
    print(f"🎯 Win Rate: {results['win_rate_stats']['win_rate']:.1f}%")
    print(f"📈 Average RR: {results['rr_analysis']['average']:.2f}")
    print(f"💰 Expected Value: {results['expected_value']['expected_value']:.3f}")
    print(f"🎯 Theoretical EV: {results['expected_value']['theoretical_ev']:.3f}")

    comparison = results['theoretical_comparison']
    if 'performance_score' in comparison:
        score = comparison['performance_score']
        print(f"🏆 RR Achievement: {score['rr_achievement']:.2f}")
        print(f"🏆 Win Rate Achievement: {score['win_rate_achievement']:.2f}")
        print(f"🏆 Overall Score: {score['overall_score']:.2f}")

    print(f"⚠️  Risk Assessment: {comparison.get('risk_assessment', 'UNKNOWN')}")
    print(f"💡 Recommendation: {results['recommendations']}")

    print("\n" + "="*60)

if __name__ == "__main__":
    asyncio.run(main())