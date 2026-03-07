# 🧪 Binance Testnet Paper Trading Setup

## Step 1: Get Testnet API Credentials

Go to: **https://testnet.binance.vision/**

1. Click "Login" (top right)
2. You can use a GitHub/email login or create a testnet account
3. Once logged in, go to **Account → API Management**
4. Click **"Create New API"**
5. Name it: "Paper Trading Bot"
6. Copy both keys:
   - **API Key** (64 characters)
   - **Secret Key** (64 characters)

---

## Step 2: Update Your .env File

Replace the current API keys with your testnet credentials:

```properties
BINANCE_API_KEY=<your-testnet-api-key>
BINANCE_API_SECRET=<your-testnet-api-secret>
BINANCE_TESTNET=true
TRADING_MODE=paper
PAPER_MODE=True
```

---

## Step 3: Verify Configuration

Make sure these are set:
```properties
PAPER_MODE=True           ✅
BINANCE_TESTNET=true      ✅
TRADING_MODE=paper        ✅
LIVE_MODE=False           ✅
```

---

## Benefits of Testnet

✅ **100% Safe** - No real money involved
✅ **Realistic** - Real market data and order book
✅ **Full Testing** - Test all trading strategies
✅ **Fast** - Instant execution for testing
✅ **Free** - No fees on testnet trades

---

## Ready to proceed?

Once you have testnet credentials from https://testnet.binance.vision/, paste them here and I'll update your .env file.
