import pandas as pd
import logging

# Set up logging for backtesting
logger = logging.getLogger("BacktestEngine")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class BacktestEngine:
    def __init__(self, historical_data, initial_balance=10000, risk_per_trade=0.02, trading_fee=0.001, stop_loss_pct=0.02, take_profit_pct=0.05):
        self.historical_data = historical_data  # Historical market data (OHLCV)
        self.balance = initial_balance  # Starting balance for backtest
        self.positions = []  # Stores open positions
        self.orders = []  # Stores trade history
        self.risk_per_trade = risk_per_trade  # Risk per trade as a percentage of balance
        self.trading_fee = trading_fee  # Trading fee per transaction
        self.stop_loss_pct = stop_loss_pct  # Stop-loss percentage
        self.take_profit_pct = take_profit_pct  # Take-profit percentage

    def execute_trade(self, action, price, quantity):
        """
        Simulate trade execution.
        :param action: 'buy' or 'sell'
        :param price: Current price of the asset
        :param quantity: Number of units to buy/sell
        """
        if action == "buy":
            cost = price * quantity * (1 + self.trading_fee)  # Account for trading fee
            if self.balance >= cost:  # Ensure enough balance
                self.balance -= cost  # Deduct the cost of the trade from balance
                self.positions.append((price, quantity))  # Add position to portfolio
                self.orders.append({"action": "buy", "price": price, "quantity": quantity})
                logger.info(f"BUY: {quantity} units at {price} USD. New balance: {self.balance}")
            else:
                logger.warning(f"Not enough balance to execute BUY order at {price} USD.")

        elif action == "sell" and self.positions:
            for pos in self.positions:
                entry_price, entry_quantity = pos
                if entry_quantity >= quantity:
                    self.balance += price * quantity * (1 - self.trading_fee)  # Add to balance
                    self.positions.remove(pos)  # Remove sold position from portfolio
                    self.orders.append({"action": "sell", "price": price, "quantity": quantity})
                    logger.info(f"SELL: {quantity} units at {price} USD. New balance: {self.balance}")
                    break
                else:
                    logger.warning("Not enough position to sell.")

    def apply_stop_loss_take_profit(self, current_price):
        """
        Apply stop-loss and take-profit rules on open positions.
        :param current_price: Current market price
        """
        for pos in self.positions:
            entry_price, quantity = pos
            # Stop-loss logic
            if current_price <= entry_price * (1 - self.stop_loss_pct):
                logger.info(f"STOP-LOSS triggered at {current_price} USD. Selling position.")
                self.execute_trade("sell", current_price, quantity)
            # Take-profit logic
            elif current_price >= entry_price * (1 + self.take_profit_pct):
                logger.info(f"TAKE-PROFIT triggered at {current_price} USD. Selling position.")
                self.execute_trade("sell", current_price, quantity)

    def run_backtest(self):
        """
        Run the backtest simulation on historical data.
        """
        logger.info("Starting backtest...")

        # Iterate through the historical data to simulate trades
        for index, row in self.historical_data.iterrows():
            current_price = row['close']
            
            # Simple trading logic: Buy when the price goes up, sell when it goes down
            if index > 0:
                prev_price = self.historical_data.iloc[index - 1]['close']
                
                if current_price > prev_price:  # Buy signal
                    quantity = (self.balance * self.risk_per_trade) / current_price  # Risk calculation
                    self.execute_trade("buy", current_price, quantity)
                elif current_price < prev_price:  # Sell signal
                    quantity = self.positions[0][1]  # Sell the entire position (could be refined further)
                    self.execute_trade("sell", current_price, quantity)
            
            # Apply stop-loss and take-profit conditions for open positions
            self.apply_stop_loss_take_profit(current_price)

        final_balance = self.balance + sum([pos[1] * self.historical_data['close'].iloc[-1] for pos in self.positions])
        logger.info(f"Final Balance after backtest: {final_balance}")
        return final_balance

# Example usage
if __name__ == "__main__":
    # Load your historical data here (from Binance API, CSV, etc.)
    symbol = "BTCUSDT"
    df = pd.read_csv(f"{symbol}_historical_data.csv")  # Assuming you have the data in CSV format

    backtest = BacktestEngine(df)
    final_balance = backtest.run_backtest()

    print(f"Final Balance after Backtest: {final_balance}")
