import os
import argparse
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import Env, spaces

from core.market_data_feed import MarketDataFeed
from core.shared_state import SharedState
from core.config import Config
from utils.indicators import compute_rsi, compute_macd, compute_ema

MODEL_PATH = "models/rl_strategist"

class TradingEnv(Env):
    def __init__(self, data: pd.DataFrame):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.max_steps = len(data) - 2
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0

        # ✅ Dynamic observation shape based on DataFrame width
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.data.shape[1],),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        return self._get_obs()

    def _get_obs(self):
        obs = self.data.iloc[self.current_step].values
        return np.array(obs, dtype=np.float32).flatten()

    def step(self, action):
        done = False
        reward = 0.0

        # ✅ Ensure scalar float
        price = float(self.data.iloc[self.current_step + 1]["close"])
        entry_now = float(self.data.iloc[self.current_step]["close"])

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = entry_now
        elif action == 2 and self.position == 1:
            reward = float(np.round(price - self.entry_price, 6))
            self.position = 0
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = entry_now
        elif action == 1 and self.position == -1:
            reward = float(np.round(self.entry_price - price, 6))
            self.position = 0

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    print("🧪 Columns before renaming:", df.columns.tolist())

    # Rename correctly
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

    # ✅ Clean base DataFrame
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    # ✅ Add technical indicators
    df["rsi"] = compute_rsi(df["close"], period=14)
    df["macd"], df["macd_signal"] = compute_macd(df["close"])
    df["ema_20"] = compute_ema(df["close"], 20)

    # ✅ Fill and clean
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    # ✅ Final clean feature set
    final_df = df[["close", "rsi", "macd", "macd_signal", "ema_20", "volume", "high", "low", "open"]].copy()

    print("✅ Final columns used for env:", final_df.columns.tolist())
    return final_df

def train(symbol="BTCUSDT"):
    print(f"📈 Loading historical data for {symbol}...")

    config = Config(simulate=True)
    shared_state = SharedState()
    market_data = MarketDataFeed(shared_state, config)

    raw = market_data.get_historical_ohlcv_sync(symbol, limit=1000)
    df = pd.DataFrame(raw)
    df = prepare_data(df)
    env = DummyVecEnv([lambda: TradingEnv(df)])

    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"📦 Model already exists. Loading from {MODEL_PATH}.zip...")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print(f"🧠 Training new PPO model for {symbol}...")
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000)

    print(f"💾 Saving model to {MODEL_PATH}.zip...")
    model.save(MODEL_PATH)
    print("✅ Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to train on (e.g., BTCUSDT)")
    args = parser.parse_args()

    train(args.symbol)
