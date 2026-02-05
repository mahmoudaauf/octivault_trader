# launch_phase9_supervisor.py
import os, uuid, time, json, re, logging
from contextvars import ContextVar

TRACE = ContextVar("trace_id", default=None)
CYCLE = ContextVar("cycle_no", default=0)

# -------- 1) JSON event overlay on top of existing logs ----------
class JsonEventHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = record.getMessage()
            ev = to_event(msg)
            if ev:
                ev["trace"] = TRACE.get()
                ev["cycle"] = CYCLE.get()
                ev["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                print(json.dumps(ev, separators=(",", ":")))
        except Exception:
            pass

def to_event(msg: str):
    # Map your *existing* log lines -> compact events
    if "OHLCV Poll Summary" in msg:
        m = re.search(r"(\d+)/(\d+) successful", msg)
        if m: return {"ev":"MDF_POLL","ok":int(m.group(1)),"total":int(m.group(2))}
    if "VolatilityRegimeDetector" in msg and "classified as" in msg:
        return {"ev":"REGIME_TICK"}
    if "ModelManager - INFO - ✅ Model loaded" in msg:
        return {"ev":"MODEL_LOADED"}
    if "Final decision for" in msg and "MLForecaster" in msg:
        # e.g. "Final decision for ADAUSDT => Action: BUY, Confidence: 0.66"
        m = re.search(r"for (\w+) => Action: (\w+), Confidence: ([0-9.]+)", msg)
        if m: return {"ev":"SIGNAL","sym":m.group(1),"act":m.group(2),"conf":float(m.group(3))}
    if "StrategyManager - INFO - Total portfolio value" in msg:
        m = re.search(r"value:\s*([0-9.]+)", msg); 
        if m: return {"ev":"PORTFOLIO_VAL","val":float(m.group(1))}
    if "Decision execution failed" in msg and "unexpected keyword argument 'comment'" in msg:
        return {"ev":"EXEC_ERR","why":"EXEC_ARG_MISMATCH(comment)"}
    return None

def install_json_handler():
    root = logging.getLogger()
    root.addHandler(JsonEventHandler())

# -------- 2) Safe patch for ExecutionManager.execute_trade ----------
def patch_execution_manager():
    try:
        from core.execution_manager import ExecutionManager
        orig = ExecutionManager.execute_trade
        def safe_execute(self, order, *args, **kwargs):
            # tolerate foreign kwargs like 'comment' without blowing up
            kwargs.pop("comment", None)
            try:
                return orig(self, order, *args, **kwargs)
            except TypeError as e:
                logging.getLogger(__name__).error("EXEC_ARG_MISMATCH(comment) handled: %s", e)
                # Return a generic error-like dict to keep the loop alive
                return {"status":"ERROR","reason":"EXEC_ARG_MISMATCH(comment)"}
        ExecutionManager.execute_trade = safe_execute
    except Exception as e:
        logging.getLogger(__name__).warning("ExecutionManager patch skipped: %s", e)

# -------- 3) Single-Loop Evaluation (SLEM) using existing public APIs ----------
def run_single_loop():
    # Import lazily so your normal init already happened
    from core.market_data_feed import MarketDataFeed
    from core.position_manager import PositionManager
    from core.strategy_manager import StrategyManager
    from core.pnl_calculator import PnLCalculator
    from core.performance_monitor import PerformanceMonitor

    TRACE.set(str(uuid.uuid4()))
    CYCLE.set(CYCLE.get() + 1)
    print(json.dumps({"ev":"P9_START","trace":TRACE.get(),"cycle":CYCLE.get()}))

    mdf_ok = getattr(MarketDataFeed.instance(), "is_fresh", lambda **_: True)(max_age_ms=2500)
    pos_ok = getattr(PositionManager.instance(), "is_fresh", lambda **_: True)(max_age_ms=5000)
    if not (mdf_ok and pos_ok):
        print(json.dumps({"ev":"SNAPSHOT_FAIL","why":"MDF_STALE_OR_POSYNC"}))
        print(json.dumps({"ev":"P9_END","trace":TRACE.get(),"cycle":CYCLE.get()}))
        return

    # Nudge strategies to produce signals/decisions once
    StrategyManager.instance().run_once()

    # Give the pipeline a moment to flow to execution (reuse your existing loops)
    time.sleep(float(os.getenv("SLEM_SETTLE_SEC","2.5")))

    # Commit metrics with your existing components
    PnLCalculator.instance().recalc()
    PerformanceMonitor.instance().tick()
    print(json.dumps({"ev":"P9_SUMMARY","note":"SLEM complete"}))
    print(json.dumps({"ev":"P9_END","trace":TRACE.get(),"cycle":CYCLE.get()}))

def bootstrap():
    install_json_handler()
    if os.getenv("PATCH_EXEC","1") == "1":
        patch_execution_manager()
    if os.getenv("PHASE9_SINGLE_LOOP","0") == "1":
        run_single_loop()

if __name__ == "__main__":
    bootstrap()
