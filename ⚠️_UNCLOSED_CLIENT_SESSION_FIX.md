# ⚠️ UNCLOSED CLIENT SESSION FIX

## Problem Analysis

**Error in Logs**:
```
asyncio.exceptions.CancelledError
2026-03-07 10:50:01,163 - ERROR - Unclosed client session
client_session: <aiohttp.client.ClientSession object at 0x771d917ddca0>
```

**Root Cause**: 
The `ExchangeClient` is creating aiohttp `ClientSession` objects but they're not being properly closed during exceptional shutdown scenarios.

## Technical Details

### Where Sessions are Created
**File**: `core/exchange_client.py` (lines 2202, 2297)

```python
self.session = aiohttp.ClientSession(timeout=ClientTimeout(total=15), connector=connector)
```

### Where Sessions Should Be Closed
**File**: `core/exchange_client.py` (lines 2244-2257)

```python
async def close(self):
    """Canonical lifecycle exit (AppContext calls this on shutdown)."""
    try:
        with contextlib.suppress(Exception):
            await self.stop_user_data_stream(close_listen_key=True)
        if self.client:
            await self.client.close_connection()
    finally:
        if self.session and not self.session.closed:
            await self.session.close()
    self.client = None
    self.session = None
```

### The Issue
When `asyncio.CancelledError` or other exceptions occur during startup or runtime, the proper shutdown sequence may be interrupted. This leaves sessions unclosed.

## Solutions

### Solution 1: Add Context Manager to ExchangeClient (Recommended)

Add `__aenter__` and `__aexit__` methods so ExchangeClient can be used with `async with`:

```python
async def __aenter__(self):
    """Context manager entry."""
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit - ensure cleanup happens."""
    await self.close()
    return False  # Don't suppress exceptions
```

**Benefits**:
- ✅ Guarantees cleanup even on exceptions
- ✅ Explicit resource management
- ✅ Pythonic async context manager pattern

### Solution 2: Add Timeout to Session Closing

Ensure the close operation doesn't hang:

```python
async def close(self):
    """Canonical lifecycle exit (AppContext calls this on shutdown)."""
    try:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                self.stop_user_data_stream(close_listen_key=True),
                timeout=5.0
            )
        if self.client:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    self.client.close_connection(),
                    timeout=5.0
                )
    finally:
        if self.session and not self.session.closed:
            try:
                await asyncio.wait_for(
                    self.session.close(),
                    timeout=5.0
                )
            except (asyncio.TimeoutError, Exception):
                # Force close if timeout
                pass
    self.client = None
    self.session = None
```

### Solution 3: Better Error Handling in Main

Add more robust error handling:

```python
async def main():
    """Main entry point with proper cleanup."""
    try:
        # Set up logging configuration
        log_path = f"logs/run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
        setup_logging(log_path)

        # Load configuration
        config = Config()

        # Initialize application context and start services
        async with AppContext(config) as app:
            await app.start_all()

            # Application is now running; press Ctrl+C to exit
            logger.info("🏃‍♂️ Application is now running. Press Ctrl+C to exit.")
            # Keep the application alive
            while True:
                await asyncio.sleep(3600)
    
    except asyncio.CancelledError:
        logger.info("✋ Application cancelled, cleaning up...")
        raise
    except KeyboardInterrupt:
        logger.info("✋ Received keyboard interrupt, cleaning up...")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        raise
```

## Recommended Implementation

Implement **Solution 1** (Context Manager) + **Solution 2** (Timeouts) for maximum robustness:

### Step 1: Add Context Manager to ExchangeClient

**Location**: `core/exchange_client.py` (add after the `close()` method)

```python
async def __aenter__(self):
    """Context manager entry."""
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit - ensure cleanup happens."""
    try:
        await asyncio.wait_for(self.close(), timeout=10.0)
    except asyncio.TimeoutError:
        self.logger.error("[EC] Close operation timed out")
    except Exception as e:
        self.logger.error(f"[EC] Error during close: {e}")
    return False
```

### Step 2: Update close() with Timeouts

Replace the current `close()` method with timeout-protected version.

### Step 3: Update AppContext.shutdown()

```python
async def shutdown(self):
    logger.info("Shutting down application context.")
    for task in self.active_tasks.values():
        task.cancel()
    
    # Wait for tasks with timeout
    try:
        await asyncio.wait_for(
            asyncio.gather(*self.active_tasks.values(), return_exceptions=True),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for active tasks to complete")
    
    # Close exchange client with timeout
    if self.exchange_client:
        try:
            await asyncio.wait_for(
                self.exchange_client.close(),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.error("Timeout closing exchange client")
        except Exception as e:
            logger.error(f"Error closing exchange client: {e}")
    
    # Close database with timeout
    if self.database_manager:
        try:
            await asyncio.wait_for(
                self.database_manager.close(),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning("Timeout closing database")
        except Exception as e:
            logger.error(f"Error closing database: {e}")
    
    # Close notification manager
    if self.notification_manager:
        try:
            await self.notification_manager.close()
        except Exception as e:
            logger.error(f"Error closing notification manager: {e}")

    # Release PID lock
    if self.pid_manager and self.pid_manager.is_locked():
        self.pid_manager.remove_pid_file()

    logger.info("AppContext shutdown complete.")
```

## Files to Modify

1. **`core/exchange_client.py`**
   - Add `__aenter__` and `__aexit__` methods
   - Add timeout protection to `close()` method
   - Add timeout protection to session close operation

2. **`main.py`**
   - Add timeout protection to `shutdown()` method
   - Better exception handling in `main()` function
   - Explicit logging for shutdown steps

## Testing

After implementing these fixes, test by:

1. **Normal Shutdown**:
   ```bash
   python main_phased.py
   # Then Ctrl+C after 10 seconds
   ```
   Verify: No "Unclosed client session" warnings

2. **Exception Shutdown**:
   ```bash
   # Modify code to throw exception during startup
   # Run and verify cleanup still happens
   ```
   Verify: Sessions are properly closed despite exception

3. **Timeout Scenario**:
   ```bash
   # Modify code to add delays to close operations
   # Verify timeouts prevent hanging
   ```
   Verify: Application shuts down within reasonable time

## Rollback Plan

If issues arise:
1. Revert changes to `core/exchange_client.py`
2. Revert changes to `main.py`
3. System will work as before (may have unclosed session warnings on shutdown)

---

**Priority**: HIGH  
**Severity**: Medium (Warnings only, no functionality loss)  
**Implementation Time**: ~15 minutes  
**Testing Time**: ~10 minutes  

