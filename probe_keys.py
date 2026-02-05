# probe_keys_legacy.py
import os
import time
import hmac
import hashlib
import requests
from dotenv import load_dotenv
from binance.client import Client

load_dotenv()

LIVE_MODE = os.getenv("LIVE_MODE", "True").lower() == "true"
TESTNET_MODE = os.getenv("TESTNET_MODE", "False").lower() == "true"

if TESTNET_MODE:
    key = os.getenv("BINANCE_TESTNET_API_KEY")
    secret = os.getenv("BINANCE_TESTNET_API_SECRET")
else:
    key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_API_SECRET")

def probe_spot():
    print("\n=== PROBE: SPOT ===")
    try:
        client = Client(key, secret, testnet=TESTNET_MODE)
        info = client.get_account()
        balances = [b for b in info["balances"] if float(b["free"]) > 0 or float(b["locked"]) > 0]
        print("OK: Spot account accessible. Non-zero assets:", [b["asset"] for b in balances])
        return True
    except Exception as e:
        print("Spot ERROR:", e)
        return False

def probe_futures():
    print("\n=== PROBE: FUTURES ===")
    try:
        ts = int(time.time() * 1000)
        params = f"timestamp={ts}"
        sig = hmac.new(secret.encode(), params.encode(), hashlib.sha256).hexdigest()
        headers = {"X-MBX-APIKEY": key}
        base = "https://testnet.binancefuture.com" if TESTNET_MODE else "https://fapi.binance.com"
        url = f"{base}/fapi/v2/balance"
        r = requests.get(url, params={"timestamp": ts, "signature": sig}, headers=headers, timeout=10)
        if r.status_code == 200:
            print("OK: Futures balance accessible. Sample:", r.json()[:1])
            return True
        else:
            print(f"Futures ERROR {r.status_code}: {r.text}")
            return False
    except Exception as e:
        print("Futures ERROR:", e)
        return False

if __name__ == "__main__":
    print("LIVE_MODE:", LIVE_MODE, "| TESTNET_MODE:", TESTNET_MODE)
    spot_ok = probe_spot()
    fut_ok = probe_futures()
    if not spot_ok and not fut_ok:
        print("\nDiagnosis: keys invalid, wrong venue, or IP restriction.")
    elif spot_ok and not fut_ok:
        print("\nDiagnosis: Spot works, but Futures permission is missing or not enabled.")
    elif not spot_ok and fut_ok:
        print("\nDiagnosis: Futures works, but Spot permission is missing or not enabled.")
