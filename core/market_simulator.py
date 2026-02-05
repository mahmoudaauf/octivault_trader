import logging
import time
import asyncio
from typing import Dict, Any, List, Optional
from decimal import Decimal

logger = logging.getLogger("MarketSimulator")

class MarketSimulator:
    """
    Simulates a crypto exchange (e.g. Binance) for backtesting.
    Tracks virtual balances and executes orders against historical prices.
    """
    def __init__(self, shared_state: Any, initial_balances: Dict[str, float] = None):
        self.shared_state = shared_state
        self.balances: Dict[str, Dict[str, float]] = initial_balances or {
            "USDT": {"free": 10000.0, "locked": 0.0}
        }
        self.orders: List[Dict[str, Any]] = []
        self.current_time_index = 0
        self.slippage_bps = 5.0 # 0.05% slippage
        self.fee_bps = 10.0      # 0.1% taker fee
        
        # Initial sync should be done by the runner via initialize() or manual await

    async def _sync_balances(self):
        """Push internal virtual balances to SharedState via update_balances."""
        if hasattr(self.shared_state, "update_balances"):
            await self.shared_state.update_balances(self.balances)
        elif hasattr(self.shared_state, "balances"):
            self.shared_state.balances = self.balances.copy()
            self.shared_state.metrics["balances_updated_at"] = time.time()
            self.shared_state.balances_ready_event.set()

    def set_time_index(self, index: int):
        self.current_time_index = index

    async def get_account_balances(self) -> Dict[str, Dict[str, float]]:
        return self.balances

    async def get_account_balance(self, asset: str) -> Dict[str, float]:
        return self.balances.get(asset.upper(), {"free": 0.0, "locked": 0.0})

    async def execute_market_order(self, symbol: str, side: str, 
                                  quantity: Optional[float] = None, 
                                  quote_order_qty: Optional[float] = None) -> Dict[str, Any]:
        """
        Simulates a market order execution.
        """
        side = side.upper()
        price = self.shared_state.latest_prices.get(symbol)
        
        if not price:
            return {"status": "FAILED", "reason": f"No price for {symbol}"}

        # Apply slippage
        exec_price = price * (1 + self.slippage_bps / 10000.0) if side == "BUY" else price * (1 - self.slippage_bps / 10000.0)
        
        base_asset = symbol.replace("USDT", "") # Simplification
        quote_asset = "USDT"

        if side == "BUY":
            if quote_order_qty:
                total_quote = quote_order_qty
                exec_qty = total_quote / exec_price
            else:
                exec_qty = quantity
                total_quote = exec_qty * exec_price
            
            fee = total_quote * (self.fee_bps / 10000.0)
            total_cost = total_quote + fee
            
            if self.balances.get(quote_asset, {}).get("free", 0) < total_cost:
                return {"status": "FAILED", "reason": "INSUFFICIENT_FUNDS"}
            
            self.balances[quote_asset]["free"] -= total_cost
            self.balances.setdefault(base_asset, {"free": 0.0, "locked": 0.0})
            self.balances[base_asset]["free"] += exec_qty
            
        else: # SELL
            exec_qty = quantity
            total_quote = exec_qty * exec_price
            fee = total_quote * (self.fee_bps / 10000.0)
            net_proceeds = total_quote - fee
            
            if self.balances.get(base_asset, {}).get("free", 0) < exec_qty:
                return {"status": "FAILED", "reason": "INSUFFICIENT_FUNDS"}
            
            self.balances[base_asset]["free"] -= exec_qty
            self.balances[quote_asset]["free"] += net_proceeds

        await self._sync_balances()
        
        result = {
            "symbol": symbol,
            "orderId": int(time.time() * 1000),
            "side": side,
            "status": "FILLED",
            "executedQty": exec_qty,
            "cummulativeQuoteQty": total_quote,
            "price": exec_price,
            "ts": time.time()
        }
        self.orders.append(result)
        logger.info(f"SIM: Executed {side} {symbol} at {exec_price:.4f} | Qty: {exec_qty:.4f}")
        return result

    # Shims for compatibility with ExecutionManager & other components
    async def create_order(self, **kwargs):
        symbol = kwargs.get("symbol")
        side = kwargs.get("side")
        quantity = kwargs.get("quantity")
        quote_order_qty = kwargs.get("quoteOrderQty")
        return await self.execute_market_order(symbol, side, quantity, quote_order_qty)

    async def place_market_order(self, symbol: str, side: str, 
                                 quantity: Optional[float] = None, 
                                 quote_order_qty: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """Canonical marketplace path expected by ExecutionManager."""
        return await self.execute_market_order(symbol, side, quantity, quote_order_qty)

    async def get_exchange_info(self) -> Dict[str, Any]:
        """Mock exchange info with standard filters for symbols in simulation."""
        symbols_info = []
        for sym in self.shared_state.accepted_symbols:
            symbols_info.append({
                "symbol": sym,
                "status": "TRADING",
                "baseAsset": sym.replace("USDT", ""),
                "quoteAsset": "USDT",
                "filters": [
                    {"filterType": "LOT_SIZE", "minQty": "0.00001000", "maxQty": "9000.00000000", "stepSize": "0.00001000"},
                    {"filterType": "NOTIONAL", "minNotional": "5.00000000", "applyToMarket": True, "avgPriceMins": 5}
                ]
            })
        return {"symbols": symbols_info}

    async def ensure_symbol_filters_ready(self, symbol: str) -> Dict[str, Any]:
        """Push mock filters to SharedState and return them."""
        if symbol not in self.shared_state.symbol_filters:
            self.shared_state.symbol_filters[symbol] = {
                "LOT_SIZE": {"minQty": "0.00001000", "maxQty": "9000.00000000", "stepSize": "0.00001000"},
                "MIN_NOTIONAL": {"minNotional": "5.10", "applyToMarket": True},
                "NOTIONAL": {"minNotional": "5.10", "applyToMarket": True}
            }
        return self.shared_state.symbol_filters[symbol]

    async def get_current_price(self, symbol: str) -> float:
        """Return the latest price from SharedState for the symbol."""
        return self.shared_state.latest_prices.get(symbol, 0.0)

    async def get_price(self, symbol: str) -> float:
        """Alias for get_current_price."""
        return await self.get_current_price(symbol)

    async def _ensure_started_public(self):
        pass

    async def refresh_balances(self):
        await self._sync_balances()
