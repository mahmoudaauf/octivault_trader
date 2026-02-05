# utils/trade_executor.py

import logging
import time

logger = logging.getLogger("TradeExecutor")
logger.setLevel(logging.INFO)


class TradeExecutor:
    def __init__(self, shared_state, execution_manager, tp_sl_engine):
        self.shared_state = shared_state
        self.execution_manager = execution_manager
        self.tp_sl_engine = tp_sl_engine

    async def execute(self, symbol, price, action, agent_name, confidence,
                      reason="", min_usdt=10.0, allocation_fraction=0.1):
        try:
            qty = await self.shared_state.get_trade_quantity(symbol, price)
            if qty <= 0:
                logger.warning(f"⚠️ [{agent_name}] Quantity invalid for {symbol}, skipping trade.")
                return {"action": "hold", "confidence": 0.0, "reason": "Quantity zero"}

            tp, sl = self.tp_sl_engine.calculate_tp_sl(symbol, price)

            signal = {
                "action": action,
                "confidence": confidence,
                "reason": reason or f"{agent_name} signal",
                "timestamp": time.time()
            }

            await self.shared_state.inject_agent_signal(agent_name, symbol, signal)

            trade_result = await self.execution_manager.execute_trade(
                symbol=symbol,
                action=action,
                confidence=confidence,
                agent=agent_name,
                reason=signal["reason"],
                take_profit=tp,
                stop_loss=sl,
                quantity=qty,
                price=price
            )

            logger.info(f"✅ [{agent_name}] Trade executed for {symbol}: {trade_result}")

            if hasattr(self.shared_state, "log_trade"):
                self.shared_state.log_trade(symbol, agent_name, {
                    "price": price,
                    "qty": qty,
                    "reason": signal['reason'],
                    "tp": tp,
                    "sl": sl
                })

            if hasattr(self.shared_state, "update_agent_roi"):
                self.shared_state.update_agent_roi(agent_name, symbol, qty, price)

            return signal

        except Exception as e:
            logger.error(f"❌ [{agent_name}] Error during trade execution for {symbol}: {str(e)}", exc_info=True)
            return {
                "action": "hold",
                "confidence": 0.0,
                "reason": f"Execution error: {str(e)}",
                "timestamp": time.time()
            }
