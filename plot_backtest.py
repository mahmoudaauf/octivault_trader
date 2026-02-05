import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_backtest(csv_path="backtest_results.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # 1. Equity Curve
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(df['step'], df['nav'], label='NAV (USDT)', color='cyan', linewidth=2)
    plt.title('Backtest Equity Curve', color='white', fontsize=14)
    plt.ylabel('NAV', color='white')
    plt.grid(True, alpha=0.2)
    plt.legend()
    
    # Dark mode aesthetics
    plt.gcf().set_facecolor('#1c1c1c')
    plt.gca().set_facecolor('#1c1c1c')
    plt.gca().tick_params(colors='white')
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['top'].set_color('white') 
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['left'].set_color('white')

    # 2. Drawdown
    df['cum_max'] = df['nav'].cummax()
    df['drawdown'] = (df['nav'] - df['cum_max']) / df['cum_max'] * 100
    
    plt.subplot(2, 1, 2)
    plt.fill_between(df['step'], df['drawdown'], 0, color='red', alpha=0.3, label='Drawdown %')
    plt.ylabel('Drawdown %', color='white')
    plt.xlabel('Step (5m bars)', color='white')
    plt.grid(True, alpha=0.2)
    plt.legend()
    
    plt.gca().set_facecolor('#1c1c1c')
    plt.gca().tick_params(colors='white')
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['top'].set_color('white') 
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['left'].set_color('white')

    plt.tight_layout()
    
    output_png = "backtest_chart.png"
    plt.savefig(output_png, facecolor='#1c1c1c')
    print(f"📈 Chart saved to {output_png}")
    # plt.show() # Can't show in headless env

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "backtest_results.csv"
    plot_backtest(path)
