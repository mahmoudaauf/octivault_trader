import asyncio
from core.config import Config
from core.database_manager import DatabaseManager
from core.shared_state import SharedState
from core.exchange_client import ExchangeClient
from core.symbol_manager import SymbolManager

from agents.wallet_scanner_agent import WalletScannerAgent
from agents.ipo_chaser import IPOChaser
from core.tp_sl_engine import TPSLEngine  # Needed for IPOChaser init

async def main():
    cfg = Config()
    db = DatabaseManager(cfg)
    await db.connect()

    shared_state = SharedState(cfg, db)
    db.shared_state = shared_state

    exchange = ExchangeClient(cfg, shared_state)
    await exchange.initialize()

    # ✅ Correct: construct symbol_mgr with proper argument order
    symbol_mgr = SymbolManager(cfg, exchange, shared_state, db)
    shared_state.symbol_manager = symbol_mgr

    # ✅ Initialize discovery agents
    wallet_agent = WalletScannerAgent(shared_state, cfg, exchange, symbol_mgr)
    ipo_agent = IPOChaser(
        shared_state,
        cfg,
        exchange,
        symbol_mgr,
        None,  # execution_manager
        TPSLEngine(cfg, shared_state, None)
    )

    # ✅ Run discovery
    await wallet_agent.run_discovery()
    await ipo_agent.run_discovery()

    # ✅ Run symbol filter pipeline
    await symbol_mgr.run_symbol_filter_pipeline()

    # ✅ Display final results
    accepted = shared_state.accepted_symbols
    print("\n✅ Accepted Symbols:")
    for sym, meta in accepted.items():
        print(f" - {sym} | Volume: {meta.get('24h_volume')} | Source: {meta.get('source')}")

    # ✅ Cleanup
    await exchange.close()
    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
