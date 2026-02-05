import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger("BacktestReporter")

class BacktestReporter:
    """
    Generates performance reports and metrics for backtesting.
    """
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def record_step(self, step_data: Dict[str, Any]):
        """Record high-level metrics for a single simulation step."""
        self.history.append(step_data)

    def generate_summary(self) -> Dict[str, Any]:
        """Calculate and return key performance indicators."""
        if not self.history:
            return {"error": "No history recorded"}

        df = pd.DataFrame(self.history)
        
        initial_nav = df['nav'].iloc[0]
        final_nav = df['nav'].iloc[-1]
        total_roi = (final_nav / initial_nav - 1) * 100

        # Calculate Returns
        df['returns'] = df['nav'].pct_change().fillna(0)
        
        # Max Drawdown
        df['cum_max'] = df['nav'].cummax()
        df['drawdown'] = (df['nav'] - df['cum_max']) / df['cum_max']
        max_drawdown = df['drawdown'].min() * 100

        # Sharpe Ratio (Assuming zero risk-free rate for simplicity)
        sharpe = 0
        if df['returns'].std() > 0:
            sharpe = (df['returns'].mean() / df['returns'].std()) * np.sqrt(252 * 288) # 5m bars -> annualize

        summary = {
            "initial_nav": initial_nav,
            "final_nav": final_nav,
            "total_roi_pct": total_roi,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe,
            "total_steps": len(df),
            "profitable_steps_pct": (df['returns'] > 0).mean() * 100
        }
        
        return summary

    def save_csv(self, filename: str = "backtest_results.csv"):
        """Save history to CSV for external analysis."""
        df = pd.DataFrame(self.history)
        df.to_csv(filename, index=False)
        logger.info(f"💾 History saved to {filename}")

    def print_report(self):
        summary = self.generate_summary()
        print("\n" + "="*40)
        print("          BACKTEST REPORT")
        print("="*40)
        print(f"Initial NAV:     {summary['initial_nav']:.2f} USDT")
        print(f"Final NAV:       {summary['final_nav']:.2f} USDT")
        print(f"Total ROI:       {summary['total_roi_pct']:.2f}%")
        print(f"Max Drawdown:    {summary['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio:    {summary['sharpe_ratio']:.2f}")
        print("-" * 40)
        print(f"Total Steps:     {summary['total_steps']}")
        print(f"Win Rate (5m):   {summary['profitable_steps_pct']:.1f}%")
        print("="*40 + "\n")
