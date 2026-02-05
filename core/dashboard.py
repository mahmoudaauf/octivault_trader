# dashboard.py

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import os
from collections import defaultdict
from core.shared_state import SharedState
from core.performance_monitor import PerformanceMonitor
from core.capital_allocator import CapitalAllocator
from core.execution_manager import ExecutionManager
from dotenv import load_dotenv

load_dotenv()

# Security Setup
TOKEN = os.getenv("DASHBOARD_TOKEN", "octivault-secret")
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized access")

# Initialize FastAPI App
app = FastAPI(title="Octivault Trader Dashboard", docs_url="/docs")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global module references
shared_state: SharedState = None
performance_monitor: PerformanceMonitor = None
capital_allocator: CapitalAllocator = None
execution_manager: ExecutionManager = None

def initialize_dashboard(state, perf_monitor, cap_alloc, exec_mgr):
    global shared_state, performance_monitor, capital_allocator, execution_manager
    shared_state = state
    performance_monitor = perf_monitor
    capital_allocator = cap_alloc
    execution_manager = exec_mgr

@app.get("/dashboard", dependencies=[Depends(verify_token)])
async def get_dashboard():
    # Per-symbol view
    symbol_metrics = {}
    for symbol, position in shared_state.positions.items():
        entry_price = position.get("entry_price", 0)
        qty = position.get("qty", 0)
        market_price = shared_state.market_data.get(symbol, {}).get("close", 0)
        pnl = (market_price - entry_price) * qty if market_price else 0

        symbol_metrics[symbol] = {
            "qty": qty,
            "entry_price": entry_price,
            "market_price": market_price,
            "unrealized_pnl": round(pnl, 4),
            "balance_USDT": shared_state.balances.get("USDT", 0),
        }

    return {
        "balances": dict(shared_state.balances),
        "positions": shared_state.positions,
        "symbol_metrics": symbol_metrics,
        "active_agents": list(shared_state.agent_signals.keys()),
        "kpi_metrics": shared_state.kpi_metrics,
    }

@app.get("/trades", dependencies=[Depends(verify_token)])
async def get_trade_log(limit: int = 20):
    return shared_state.trade_log[-limit:]

@app.get("/agents", dependencies=[Depends(verify_token)])
async def get_agents(symbol: str = Query(None)):
    result = {}
    for agent, scores in shared_state.agent_scores.items():
        if symbol:
            sym_scores = scores.get("symbols", {}).get(symbol, {})
            result[agent] = {
                "allocated_capital": sym_scores.get("capital", 0),
                "ROI": sym_scores.get("ROI", 0),
                "Sharpe": sym_scores.get("Sharpe", 0),
                "win_rate": sym_scores.get("win_rate", 0),
                "trades_per_hour": sym_scores.get("trades_per_hour", 0)
            }
        else:
            result[agent] = {
                "total_capital": scores.get("capital", 0),
                "ROI": scores.get("ROI", 0),
                "Sharpe": scores.get("Sharpe", 0),
                "win_rate": scores.get("win_rate", 0),
                "trades_per_hour": scores.get("trades_per_hour", 0),
                "by_symbol": scores.get("symbols", {})  # nested data
            }
    return result

@app.get("/performance", dependencies=[Depends(verify_token)])
async def get_performance_summary():
    return {
        "total_profit": performance_monitor.total_profit,
        "profit_by_symbol": performance_monitor.profit_by_symbol,
        "equity_curve": performance_monitor.equity_curve,
        "global_KPIs": shared_state.kpi_metrics
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}
