# Binance WebSocket API v3 Authentication Limitation

## Issue

Code 1008 (Policy Violation) when trying to authenticate with HMAC-only API keys.

## Root Cause

Binance WS API v3 requires **Ed25519 keys** for user-data authentication. HMAC keys are **NOT supported** for WebSocket user-data streams.

### Authentication Methods

| Method | Requires | Works with |
|--------|----------|-----------|
| `session.logon` | Ed25519 private key | Ed25519 keys only |
| `userDataStream.subscribe.signature` | NOT SUPPORTED | ❌ N/A |
| `userDataStream.subscribe` | Active session (requires logon first) | Requires Ed25519 keys |

## Current Situation

- User has HMAC-SHA256 API keys (not Ed25519)
- Binance WS API v3 rejects all authentication attempts with HMAC keys
- Server closes connection with code 1008 (policy violation)

## Solutions

### Option 1: Use WebSocket Streams API (Recommended)
- Use the old REST-based WebSocket streaming API
- Get a listenKey via REST API: `POST /api/v3/userDataStream`
- Connect to: `wss://stream.binance.com:9443/ws/{listenKey}`
- No signature needed, listenKey is temporary credential

### Option 2: Migrate to Ed25519 Keys
- Create new API key with Ed25519 algorithm
- Use `session.logon` method
- All requests signed with Ed25519 private key

### Option 3: Disable WebSocket User-Data
- Only use REST API for account updates
- Disable WS user-data stream startup

## Recommended Implementation

Use WebSocket Streams API with listenKey:

```python
# 1. Create listen key (REST)
async def create_listen_key():
    # POST /api/v3/userDataStream
    # Returns {"listenKey": "..."}
    pass

# 2. Connect WebSocket
ws_url = f"wss://stream.binance.com:9443/ws/{listen_key}"
async with session.ws_connect(ws_url, heartbeat=None, autoping=False) as ws:
    # Listen for events
    async for msg in ws:
        # Process account updates, order fills, etc.
        pass

# 3. Refresh listen key (every 30 minutes)
async def refresh_listen_key(listen_key):
    # PUT /api/v3/userDataStream
    pass
```

## Status

❌ **WS API v3 with HMAC keys: NOT POSSIBLE**
✅ **WebSocket Streams API with listenKey: POSSIBLE**
✅ **WS API v3 with Ed25519 keys: POSSIBLE**
