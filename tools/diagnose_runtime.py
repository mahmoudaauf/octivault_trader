# tools/diagnose_runtime.py
import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

# ==== imports من مشروعك (عدّل المسارات لو تختلف) ====
from src.l0_core.config import Config
from src.l0_core.logger_utils import setup_structured_logging
from src.l0_core.shared_state import SharedState
from src.l1_exchange.exchange_client import ExchangeClient
from src.l2_marketdata.heartbeat import Heartbeat
from src.l2_marketdata.market_data_feed import MarketDataFeed
from src.l4_execution.execution_manager import ExecutionManager
from src.l4_execution.tp_sl_engine import TP_SLEngine
from src.l5_strategy.agent_manager import AgentManager
from src.l6_governance.risk_manager import RiskManager
from src.l7_observability.diagnostics.system_summary import system_summary_logger
from src.l8_lifecycle.meta_controller import MetaController
from src.l8_lifecycle.watchdog import Watchdog

# ========== إعدادات تشخيص ==========
LIVE_MODE = True  # شغّل على Binance Live
DRY_RUN = False  # لو True يمنع إصدار أوامر حقيقية عبر ExecutionManager (لو عندكم دعم المحاكاة)
TEST_TIMEFRAME = "1m"
READY_TIMEOUT_S = 90  # انتظر جاهزية MarketDataFeed
SMOKE_RUNTIME_S = 180  # مدة التشغيل التشخيصي قبل الإنهاء المنظم
MIN_SYMBOLS = 3  # أقل عدد رموز مقبولة قبل بدء التشخيص
MIN_BALANCE_CHECK = ["USDT", "FDUSD"]  # عملات نتحقق من رصيدها

# (اختياري) تثبيت Universe للتشخيص. لو تتركه None، يستخدم الاكتشاف الديناميكي عندكم.
TEST_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]


# ========== Helpers ==========
def wrap_task(name: str, coro_fn: Callable[[], Awaitable[Any]]):
    logger = logging.getLogger(f"DiagTask:{name}")

    async def _wrapper():
        logger.info(f"▶️ بدء مهمة: {name}")
        try:
            await coro_fn()
            logger.info(f"✅ المهمة اكتملت: {name}")
        except asyncio.CancelledError:
            logger.warning(f"⚠️ المهمة أُلغيَت: {name}")
            raise
        except Exception as e:
            logger.exception(f"🔥 المهمة تعطلت: {name} | {e}")

    return _wrapper


async def smoke_checks(shared: SharedState, exch: ExchangeClient, mdf: MarketDataFeed):
    log = logging.getLogger("SmokeChecks")

    # 1) فحص رموز التداول
    accepted = shared.get_accepted_symbols() or []
    log.info(f"📊 Accepted symbols: {len(accepted)} -> {accepted[:10]}")
    if TEST_SYMBOLS and not accepted:
        # لو نظامكم يقبل فرض الرموز:
        try:
            for s in TEST_SYMBOLS:
                shared.accept_symbol(s, metadata={"source": "diagnostic"})
            accepted = TEST_SYMBOLS[:]
            log.info("🧪 فرضنا TEST_SYMBOLS في SharedState لهذا التشخيص.")
        except Exception:
            log.info("ℹ️ لا يمكن فرض TEST_SYMBOLS؛ نكمل بالمتاح.")

    if len(accepted) < MIN_SYMBOLS:
        raise RuntimeError(
            f"عدد الرموز المقبولة قليل ({len(accepted)})، نحتاج ≥ {MIN_SYMBOLS} قبل الاختبار."
        )

    # 2) فحص معلومات الرمز/الفلاتر من Binance
    for s in accepted[:MIN_SYMBOLS]:
        info = await exch.get_symbol_info(s)
        price = await exch.get_current_price(s)
        log.info(f"🔎 {s}: price={price} info_ok={bool(info)}")

    # 3) فحص الأرصدة
    balances = await exch.get_balances()
    log.info(f"💰 Balances snapshot: { {k: dict(v) for k,v in list(balances.items())[:5]} }")
    for ccy in MIN_BALANCE_CHECK:
        if ccy in balances:
            log.info(f"✅ Balance {ccy}: {balances[ccy].get('free', 0)} free")
        else:
            log.info(f"⚠️ لا يوجد رصيد ظاهر لـ {ccy} (مو لازم خطأ، مجرد معلومة)")

    # 4) فحص بيانات السوق (OHLCV)
    for s in accepted[:MIN_SYMBOLS]:
        candles = mdf.get_recent_ohlcv(s, TEST_TIMEFRAME) or []
        log.info(f"🕯️ OHLCV {s}-{TEST_TIMEFRAME}: {len(candles)} شموع")
        if not candles:
            raise RuntimeError(
                f"لا توجد شموع لـ {s}-{TEST_TIMEFRAME}. تأكد أن MarketDataFeed يعمل."
            )

    log.info("✅ Smoke checks passed.")


async def main():
    setup_structured_logging(level=logging.INFO)
    root = logging.getLogger("DiagnoseRuntime")
    root.info("🚀 بدء جلسة تشخيص Live متكاملة…")

    # 1) تهيئة Config + SharedState + Exchange
    config = Config()
    config.set("LIVE_TRADING", LIVE_MODE)
    config.set("DRY_RUN", DRY_RUN)
    config.set("MIN_NOTIONAL_MARGIN", 1.02)
    config.set("USE_QUANTITY_MARKET_ORDERS", True)

    shared_state = SharedState(config=config)
    exchange = ExchangeClient(config=config, shared_state=shared_state)

    # 2) Execution + Risk + TP/SL
    execution = ExecutionManager(shared_state=shared_state, exchange_client=exchange, config=config)
    risk = RiskManager(shared_state=shared_state, config=config)
    tp_sl = TP_SLEngine(shared_state=shared_state, execution_manager=execution, config=config)

    root.info("✅ ExecutionManager initialized.")

    # 3) MarketDataFeed (لازم قبل AgentManager)
    mdf = MarketDataFeed(shared_state=shared_state, exchange_client=exchange, config=config)

    # 4) AgentManager + MetaController
    agent_mgr = AgentManager(
        shared_state=shared_state,
        execution_manager=execution,
        config=config,
        market_data=mdf,  # ← هذا كان ناقص
        tp_sl_engine=tp_sl,
        risk_manager=risk,
    )
    meta = MetaController(
        shared_state=shared_state,
        execution_manager=execution,
        agent_manager=agent_mgr,
        config=config,
    )

    # 5) Watchdog + Heartbeat + System summary
    watchdog = Watchdog(shared_state=shared_state, interval_seconds=10, tolerance_seconds=30)
    heartbeat = Heartbeat(shared_state=shared_state, interval_seconds=30)

    # 6) تشغيل المكوّنات كمهمات async
    tasks: list[asyncio.Task] = []
    loop = asyncio.get_running_loop()

    async def run_market_data():
        await mdf.start()  # لو عندكم start() تشغّل polling

    async def run_agents():
        await agent_mgr.run_loop()

    async def run_meta():
        await meta.run_loop()

    async def run_watchdog():
        await watchdog.run()

    async def run_heartbeat():
        await heartbeat.run()

    async def run_system_summary():
        while True:
            await system_summary_logger(shared_state, exchange, agent_mgr, execution)
            await asyncio.sleep(20)

    # جدولة المهام
    tasks.append(loop.create_task(wrap_task("MarketDataFeed")(run_market_data())))
    tasks.append(loop.create_task(wrap_task("AgentManager")(run_agents())))
    tasks.append(loop.create_task(wrap_task("MetaController")(run_meta())))
    tasks.append(loop.create_task(wrap_task("Watchdog")(run_watchdog())))
    tasks.append(loop.create_task(wrap_task("Heartbeat")(run_heartbeat())))
    tasks.append(loop.create_task(wrap_task("SystemSummary")(run_system_summary())))

    # 7) انتظار جاهزية MarketDataFeed
    if hasattr(mdf, "get_ready_event"):
        ready_event = mdf.get_ready_event()
        try:
            await asyncio.wait_for(ready_event.wait(), timeout=READY_TIMEOUT_S)
            root.info("✅ MarketDataFeed جاهز.")
        except asyncio.TimeoutError:
            root.error(f"⏰ MarketDataFeed لم يصبح جاهزاً خلال {READY_TIMEOUT_S}s")
            # نكمل لكن سنفشل في smoke_checks غالباً

    # 8) Smoke checks (تُفشل سريعاً لو في مشكلة تكامل)
    await smoke_checks(shared_state, exchange, mdf)

    # 9) تشغيل لفترة محدودة للتشخيص
    root.info(f"⏳ تشغيل تشخيصي لمدة ~{SMOKE_RUNTIME_S}s… راقب السجلات لأي أخطاء.")
    try:
        await asyncio.sleep(SMOKE_RUNTIME_S)
    finally:
        root.info("🛑 إيقاف منظم للمهام…")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        root.info("✅ انتهى التشخيص.")


if __name__ == "__main__":
    asyncio.run(main())
