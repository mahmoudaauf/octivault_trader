#!/usr/bin/env python3
"""Live balance check from Binance."""
import os, time, hmac, hashlib, requests, sys
from urllib.parse import urlencode
from pathlib import Path

# Manually load .env
env_path = Path(__file__).parent / ".env"
env = {}
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")

K = env.get("BINANCE_API_KEY")
S = env.get("BINANCE_API_SECRET_HMAC")
T = env.get("BINANCE_TESTNET", "false").lower() == "true"
U = "https://testnet.binance.vision" if T else "https://api.binance.com"

print(f"🔍 Binance {'TESTNET' if T else 'LIVE'}: {U}")
ts = int(time.time() * 1000)
q = urlencode({"timestamp": ts, "recvWindow": 5000})
sig = hmac.new(S.encode(), q.encode(), hashlib.sha256).hexdigest()
r = requests.get(f"{U}/api/v3/account?{q}&signature={sig}",
                 headers={"X-MBX-APIKEY": K}, timeout=10)
print(f"HTTP {r.status_code}")
if r.status_code != 200:
    print(r.text[:300]); sys.exit(1)

d = r.json()
bs = [b for b in d["balances"] if float(b["free"]) + float(b["locked"]) > 0]
pr = requests.get(f"{U}/api/v3/ticker/price", timeout=10).json()
P = {p["symbol"]: float(p["price"]) for p in pr}

rows = []
total = 0; fu = 0; pos = 0
for b in bs:
    a = b["asset"]; f = float(b["free"]); l = float(b["locked"]); t = f + l
    if a == "USDT":
        v = t; fu += f
    elif a == "BFUSD":
        v = t; fu += f  # 1:1 USDT-pegged stablecoin
    else:
        sym = f"{a}USDT"
        v = t * P.get(sym, 0)
        pos += v
    total += v
    rows.append((a, f, v))

rows.sort(key=lambda x: -x[2])
print(f"\n💰 ACTUAL ACCOUNT STATE:")
print(f"   Total NAV:        ${total:>9.2f}")
print(f"   Free USDT:        ${fu:>9.2f}  (incl BFUSD)")
print(f"   In Positions:     ${pos:>9.2f}")
print(f"   Position count:   {len([r for r in rows if r[0] not in ['USDT','BFUSD']])}")

print(f"\n📊 ALL POSITIONS BY VALUE:")
for a, f, v in rows:
    tag = "CASH" if a in ["USDT", "BFUSD"] else ("TRADEABLE" if v >= 5 else "DUST")
    print(f"   {a:<12} ${v:>7.2f}  [{tag}]")

trad = [r for r in rows if r[0] not in ["USDT","BFUSD"] and r[2] >= 5.0]
dust = [r for r in rows if r[0] not in ["USDT","BFUSD"] and 0 < r[2] < 5.0]
unp = [r for r in rows if r[0] not in ["USDT","BFUSD"] and r[2] == 0]
print(f"\n📈 CATEGORIZATION:")
print(f"   TRADEABLE (>=$5): {len(trad)} positions = ${sum(r[2] for r in trad):.2f}")
print(f"   DUST (<$5):       {len(dust)} positions = ${sum(r[2] for r in dust):.2f}")
print(f"   UNPRICED:         {len(unp)} positions (no USDT pair)")
if unp:
    print(f"      Assets: {', '.join(r[0] for r in unp)}")
