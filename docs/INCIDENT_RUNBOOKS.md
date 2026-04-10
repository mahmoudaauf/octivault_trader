# Incident Runbooks - Octivault Trader

Critical playbooks for on-call incident response.

**Status**: P9-aligned
**Revision**: 2026-04-10
**Team**: DevOps + Trading Operations

---

## Table of Contents

1. [Database is Down](#database-is-down)
2. [Trading System Hung](#trading-system-hung)
3. [Capital Allocation Failure](#capital-allocation-failure)

---

## Database is Down

**Severity**: CRITICAL 🔴  
**Response Time**: < 5 minutes  
**Owner**: DevOps Lead

### Symptoms

- Error logs show "database connection refused"
- `/ready` endpoint returns `"database": false`
- Dashboard endpoint `GET /dashboard` returns 503
- Application continues running but cannot persist state

### Detection

```bash
# Check if database file exists
ls -lh /app/data/octivault.db

# Check database permissions
stat /app/data/octivault.db

# Try to connect directly
sqlite3 /app/data/octivault.db ".tables"

# Check disk space (database disk full is common)
df -h /app/data/
```

### Root Cause Analysis (2 minutes)

1. **Check disk space first** (most common)
   ```bash
   df -h /
   # If > 90% full: emergency cleanup needed
   ```

2. **Check file permissions**
   ```bash
   ls -la /app/data/octivault.db
   # Should be owned by trader user (uid 1000)
   ```

3. **Check for database corruption**
   ```bash
   sqlite3 /app/data/octivault.db "PRAGMA integrity_check;"
   # Should return: ok
   ```

4. **Check for file locks**
   ```bash
   lsof /app/data/octivault.db
   ```

### Remediation Steps

**Option A: Restart Container (3 minutes)**
```bash
# 1. Check last N lines of logs
docker logs octivault_trader | tail -50

# 2. Stop and remove container
docker-compose stop trader
docker-compose rm trader

# 3. Restart
docker-compose up -d trader

# 4. Verify health
curl http://localhost:8000/ready
```

**Option B: Manual Database Repair (5 minutes)**
```bash
# If integrity check fails:
sqlite3 /app/data/octivault.db "PRAGMA integrity_check;" | grep -v ok

# Backup corrupted database
cp /app/data/octivault.db /app/data/octivault.db.backup-$(date +%s)

# Attempt automatic repair
sqlite3 /app/data/octivault.db ".dump" > /tmp/dump.sql
sqlite3 /app/data/octivault_repaired.db < /tmp/dump.sql

# Replace if successful
mv /app/data/octivault.db /app/data/octivault.db.corrupted
mv /app/data/octivault_repaired.db /app/data/octivault.db
```

**Option C: Restore from Backup (5 minutes)**
```bash
# If backup exists in recovery snapshots
ls -lh ./snapshots/

# Restore latest snapshot
cp ./snapshots/octivault_latest.db /app/data/octivault.db
chown trader:trader /app/data/octivault.db

# Restart
docker-compose restart trader
```

### Verification

```bash
# All should return 200
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/live

# Check logs for errors
docker logs octivault_trader | grep -i error
```

### Post-Incident

- [ ] Document root cause
- [ ] Check if disk space issue is recurring
- [ ] Implement automated backups if not present
- [ ] Review database fragmentation

---

## Trading System Hung

**Severity**: CRITICAL 🔴  
**Response Time**: < 2 minutes  
**Owner**: Backend Lead

### Symptoms

- `/ready` returns 200 but `/live` returns unhealthy
- `time_since_last_trade_sec` > 300 (5 minutes)
- No new trades appearing in trade log
- CPU usage drops to near 0%

### Detection

```bash
# Check live status
curl http://localhost:8000/live

# Check if process is still running
docker ps | grep octivault_trader

# Check CPU/Memory
docker stats octivault_trader

# Check process state
docker exec octivault_trader ps aux | grep main_phased
```

### Root Cause Analysis (1 minute)

1. **Check if process is stuck**
   ```bash
   # Look for zombie processes or high CPU threads
   docker exec octivault_trader ps aux --sort=-%cpu | head -10
   ```

2. **Check event queue depth**
   ```bash
   # Tail logs and look for queue warnings
   docker logs octivault_trader --tail=100 | grep -i queue
   ```

3. **Check market data status**
   ```bash
   # Check if Binance API is reachable
   curl -s https://api.binance.com/api/v3/ping
   # Should return: {}
   ```

4. **Check for deadlocks or async issues**
   ```bash
   docker logs octivault_trader --tail=100 | grep -iE "error|exception|timeout"
   ```

### Remediation Steps

**Option A: Graceful Restart (1 minute)**
```bash
# Send SIGTERM to allow cleanup
docker-compose restart trader

# Wait for /live to become healthy
while ! curl -s http://localhost:8000/live | grep -q '"live": true'; do
  echo "Waiting for trading to resume..."
  sleep 5
done
```

**Option B: Force Restart (30 seconds)**
```bash
# Immediate restart (may lose in-flight state)
docker-compose stop trader
sleep 2
docker-compose up -d trader

# Monitor
docker logs -f octivault_trader
```

**Option C: Kill Hung Process (15 seconds)**
```bash
# Find process ID
PID=$(docker exec octivault_trader pgrep -f main_phased)

# Send SIGKILL
docker exec octivault_trader kill -9 $PID

# Let container restart (restart: unless-stopped)
sleep 5
docker logs octivault_trader | tail -20
```

### Verification

```bash
# Monitor for ~30 seconds
for i in {1..6}; do
  curl -s http://localhost:8000/live | jq '.live'
  sleep 5
done
# Should all be: true
```

### Post-Incident

- [ ] Extract logs for analysis: `docker logs octivault_trader > /tmp/logs.txt`
- [ ] Check for Binance API issues (rate limiting, outage)
- [ ] Review event loop performance
- [ ] Check if market conditions caused hang (extreme volatility)

---

## Capital Allocation Failure

**Severity**: CRITICAL 🔴  
**Response Time**: < 1 minute  
**Owner**: Trading Lead

### Symptoms

- Allocation errors in logs: "Cannot allocate capital"
- Balance in dashboard shows 0 or negative
- New trades being rejected
- Alert: "Capital check failed"

### Detection

```bash
# Check current allocation status
curl -s http://localhost:8000/dashboard -H "Authorization: Bearer $TOKEN" | jq '.balances'

# Check for allocation errors
docker logs octivault_trader | grep -i "allocat"

# Check capital ledger
docker exec octivault_trader sqlite3 /app/data/octivault.db \
  "SELECT * FROM capital_ledger ORDER BY timestamp DESC LIMIT 10;"
```

### Root Cause Analysis (30 seconds)

1. **Check balance sync with Binance**
   ```bash
   # Compare what we think we have vs. Binance
   docker logs octivault_trader | grep -i "balance" | tail -5
   ```

2. **Check for stuck positions**
   ```bash
   # Open positions should total to deployed capital
   curl -s http://localhost:8000/dashboard -H "Authorization: Bearer $TOKEN" | \
     jq '.positions | map(.qty * .entry_price) | add'
   ```

3. **Check for dust balances**
   ```bash
   # Check if small dust amounts are blocking allocation
   curl -s http://localhost:8000/dashboard -H "Authorization: Bearer $TOKEN" | \
     jq '.balances | to_entries[] | select(.value < 0.01)'
   ```

### Remediation Steps

**Option A: Balance Resync (1 minute)**
```bash
# Force balance refresh from Binance
# Add this to config or restart with recovery enabled
docker-compose restart trader

# Monitor for balance recovery
sleep 5
curl -s http://localhost:8000/dashboard -H "Authorization: Bearer $TOKEN" | jq '.balances'
```

**Option B: Clear Dust Balances (2 minutes)**
```bash
# Sweep micro-dust to consolidate
# (Only if dust < 0.01 USDT and not needed for fees)
curl -X POST http://localhost:8000/admin/sweep-dust \
  -H "Authorization: Bearer $TOKEN"
```

**Option C: Manual Position Close (5 minutes)**
```bash
# If stuck position is blocking capital:
# 1. Close smallest position manually
# 2. Check capital recovery
# 3. Restart trading

docker exec octivault_trader python3 << 'EOF'
from core.binance_client import BinanceClient
client = BinanceClient()
# Close position: client.market_order(symbol, "SELL", qty)
EOF
```

**Option D: Emergency Withdrawal (LAST RESORT)**
```bash
# Only if capital is truly stuck and unrecoverable
# Withdraw to wallet and re-deposit fresh

# DO NOT do this in production without capital team approval!
# Contact CTO + CFO first
```

### Verification

```bash
# Balance should match between dashboard and Binance
DASHBOARD_BALANCE=$(curl -s http://localhost:8000/dashboard \
  -H "Authorization: Bearer $TOKEN" | jq '.balances.USDT')

# Check that capital_profile is switching correctly
docker logs octivault_trader | grep -i "capital.profile"

# Check that allocations are resuming
docker logs octivault_trader | grep -i "allocation" | tail -5
```

### Post-Incident

- [ ] Root cause: Was it sync issue, dust, or position tracking bug?
- [ ] Did we lose capital? (compare with Binance statement)
- [ ] Update capital reconciliation logic if needed
- [ ] Add test for this scenario

---

## Quick Reference

| Incident | Health Check | Fix Time | Team |
|----------|--------------|----------|------|
| Database Down | `/ready` = false | 3-5 min | DevOps |
| System Hung | `/live` = false | 1-2 min | Backend |
| Capital Failed | Dashboard = 0 | 1 min | Trading |

## Emergency Contacts

- **On-Call Lead**: Slack #octivault-incidents
- **CTO**: [contact info]
- **DevOps**: [contact info]
- **Trading**: [contact info]

## Escalation

1. **5 minutes no resolution** → Escalate to Tech Lead
2. **10 minutes no resolution** → Page CTO
3. **15 minutes no resolution** → Declare incident, page senior eng
