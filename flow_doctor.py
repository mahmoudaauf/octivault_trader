# flow_doctor.py
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Iterable, Tuple

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

SEV_INFO = "INFO"
SEV_WARN = "WARN"
SEV_ERR  = "ERROR"

@dataclass
class Finding:
    severity: str
    code: str
    message: str
    hint: Optional[str] = None
    meta: Dict = field(default_factory=dict)

class FlowDoctor:
    """
    Validates: startup phase flow, agent/MC loops, MDF readiness, signals, and execution path.
    Input sources: log file + env (or imported Config).
    """
    # --- regexes for plain log lines ---
    RX_META_LOOP    = re.compile(r"MetaController loop started", re.I)
    RX_AGENTM_WARM  = re.compile(r"All agents warmed up", re.I)
    RX_STRAT_LOOP   = re.compile(r"StrategyManager .*started|StrategyManagerLoop started", re.I)
    RX_EXEC_LOOP    = re.compile(r"ExecutionManager(Loop)? .*started|process_queue_once", re.I)
    RX_TPSL_LOOP    = re.compile(r"TP[_\-]?SLEngineLoop started|TP[_\-]?SLEngine .*started", re.I)
    RX_POSM_LOOP    = re.compile(r"PositionManager .*started|PositionManagerLoop started", re.I)
    RX_SYMBOLS_SET  = re.compile(r"Accepted symbols set \((\d+)\)")
    RX_PHASE_ENTER  = re.compile(r"START Phase (\d+)")
    RX_PHASE_END    = re.compile(r"END Phase (\d+)")
    RX_PREFETCH_OK  = re.compile(r"OHLCV prefetch completed", re.I)
    RX_MARKET_READY = re.compile(r"Emitted MarketDataReady|MarketDataFeed.*ready", re.I)
    RX_AGENT_REGISTER= re.compile(r"agents? registered|auto_register_agents", re.I)
    RX_META_STARTED  = re.compile(r"MetaController .* started", re.I)

    # --- JSON overlay events (from your _JsonEventHandler) ---
    # If ENABLE_P9_JSON_EVENTS=True, logs will contain minified JSON lines like:
    # {"ev":"STRAT_LOOP"}   {"ev":"EXEC_LOOP"}   {"ev":"SIGNAL","sym":"BTCUSDT","act":"BUY","conf":0.71}
    def parse_event(self, line: str) -> Optional[dict]:
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            return None
        try:
            ev = json.loads(line)
            # We only consider tiny events your overlay is known to emit
            if ev.get("ev") in {"MDF_POLL","REGIME_CLASSIFIED","SIGNAL","EXEC_ERR","STRAT_LOOP","EXEC_LOOP"}:
                return ev
        except Exception:
            return None
        return None

    def __init__(self, log_path: Optional[str], env_path: Optional[str], import_config: bool):
        self.log_path = log_path
        self.env_path = env_path
        self.import_config = import_config
        self.findings: List[Finding] = []
        self.events: List[dict] = []      # parsed JSON overlay events
        self.lines: List[str] = []        # raw log lines
        self.flags_seen: Dict[str, bool] = {
            "meta_loop": False,
            "agent_warm": False,
            "strat_loop": False,
            "exec_loop": False,
            "tpsl_loop": False,
            "posm_loop": False,
            "prefetch_ok": False,
            "market_ready": False,
            "agent_register": False,
            "meta_started": False
        }
        self.symbol_counts: List[int] = []
        self.phases: Dict[str, bool] = {f"p{i}": False for i in range(1, 10)}
        self.cfg: Dict[str, str] = {}

    # ---------------- IO ----------------
    def load_env(self):
        if self.env_path and os.path.exists(self.env_path):
            if load_dotenv:
                load_dotenv(self.env_path)
        # collect relevant env into self.cfg (strings; cast later as needed)
        keys = [
            "LIVE_MODE","SIMULATION_MODE","PAPER_MODE","TESTNET_MODE",
            "ENABLE_STRATEGY_MANAGER","ENABLE_LIQUIDATION_AGENT",
            "ALERT_SYSTEM_ENABLED",
            "MIN_EXECUTION_CONFIDENCE","MIN_ORDER_USDT","TRADE_AMOUNT_USDT",
            "MIN_SIGNAL_CONF","MAX_SIGNAL_AGE_SEC","META_DECISION_COOLDOWN_SEC",
            "DEFAULT_PLANNED_QUOTE","MIN_ENTRY_QUOTE_USDT",
            "PORTFOLIO_MIN_ENTRY_QUOTE_USDT",  # used by balancer
            "PHASE9_RUN_ONCE","PATCH_EXEC"
        ]
        for k in keys:
            v = os.getenv(k)
            if v is not None:
                self.cfg[k] = v

    def maybe_import_config(self):
        if not self.import_config:
            return
        try:
            from core.config import Config
            cfg = Config()
            # pull a selection of attributes that impact flow/execution
            def _get(name, default=None):
                return getattr(cfg, name, os.getenv(name, default))
            pairs = {
                "LIVE_MODE": _get("LIVE_MODE"),
                "SIMULATION_MODE": _get("SIMULATION_MODE"),
                "PAPER_MODE": _get("PAPER_MODE"),
                "ENABLE_STRATEGY_MANAGER": _get("ENABLE_STRATEGY_MANAGER"),
                "ENABLE_LIQUIDATION_AGENT": _get("ENABLE_LIQUIDATION_AGENT"),
                "ALERT_SYSTEM_ENABLED": _get("ALERT_SYSTEM_ENABLED"),
                "MIN_EXECUTION_CONFIDENCE": _get("MIN_EXECUTION_CONFIDENCE"),
                "MIN_ORDER_USDT": _get("MIN_ORDER_USDT"),
                "TRADE_AMOUNT_USDT": _get("TRADE_AMOUNT_USDT"),
                "MIN_SIGNAL_CONF": _get("MIN_SIGNAL_CONF"),
                "MAX_SIGNAL_AGE_SEC": _get("MAX_SIGNAL_AGE_SECONDS", _get("MAX_SIGNAL_AGE_SEC")),
                "META_DECISION_COOLDOWN_SEC": _get("META_DECISION_COOLDOWN_SEC"),
                "DEFAULT_PLANNED_QUOTE": _get("DEFAULT_PLANNED_QUOTE"),
                "MIN_ENTRY_QUOTE_USDT": _get("MIN_ENTRY_QUOTE_USDT"),
                "PHASE9_RUN_ONCE": _get("PHASE9_RUN_ONCE", False),
                "PATCH_EXEC": os.getenv("PATCH_EXEC", "True")
            }
            for k, v in pairs.items():
                if v is not None:
                    self.cfg[k] = str(v)
        except Exception as e:
            self.findings.append(Finding(
                SEV_WARN, "CFG_IMPORT_FAIL",
                f"Could not import Config() ({e}). Falling back to env-only."
            ))

    def load_log(self):
        if not self.log_path or not os.path.exists(self.log_path):
            self.findings.append(Finding(
                SEV_WARN, "NO_LOG",
                "Log file not found or not provided. Some checks will be skipped.",
                hint="Pass --log path/to/log"
            ))
            return
        with open(self.log_path, "r", encoding="utf-8", errors="ignore") as f:
            self.lines = f.readlines()

    # ---------------- Parsers ----------------
    def scan_lines(self):
        for ln in self.lines:
            ev = self.parse_event(ln)
            if ev:
                self.events.append(ev)
                if ev["ev"] == "STRAT_LOOP":
                    self.flags_seen["strat_loop"] = True
                elif ev["ev"] == "EXEC_LOOP":
                    self.flags_seen["exec_loop"] = True
                elif ev["ev"] == "SIGNAL":
                    # Keep all signals for later stats
                    pass
                elif ev["ev"] == "EXEC_ERR":
                    self.findings.append(Finding(
                        SEV_ERR, "EXEC_ERR",
                        "Execution error from log overlay.",
                        hint=str(ev)
                    ))
                continue

            if self.RX_META_LOOP.search(ln):
                self.flags_seen["meta_loop"] = True
            if self.RX_AGENTM_WARM.search(ln):
                self.flags_seen["agent_warm"] = True
            if self.RX_STRAT_LOOP.search(ln):
                self.flags_seen["strat_loop"] = True
            if self.RX_EXEC_LOOP.search(ln):
                self.flags_seen["exec_loop"] = True
            if self.RX_TPSL_LOOP.search(ln):
                self.flags_seen["tpsl_loop"] = True
            if self.RX_POSM_LOOP.search(ln):
                self.flags_seen["posm_loop"] = True
            if self.RX_PREFETCH_OK.search(ln):
                self.flags_seen["prefetch_ok"] = True
            if self.RX_MARKET_READY.search(ln):
                self.flags_seen["market_ready"] = True
            if self.RX_AGENT_REGISTER.search(ln):
                self.flags_seen["agent_register"] = True
            if self.RX_META_STARTED.search(ln):
                self.flags_seen["meta_started"] = True

            m = self.RX_SYMBOLS_SET.search(ln)
            if m:
                try:
                    self.symbol_counts.append(int(m.group(1)))
                except Exception:
                    pass

            pe = self.RX_PHASE_ENTER.search(ln)
            if pe:
                self.phases.get(f"p{pe.group(1)}")  # touch
            pd = self.RX_PHASE_END.search(ln)
            if pd:
                self.phases[f"p{pd.group(1)}"] = True

    # ---------------- Validators ----------------
    def _bool(self, key: str, default: bool = False) -> bool:
        v = self.cfg.get(key)
        if v is None:
            return default
        return str(v).strip().lower() in ("1","true","t","yes","y","on")

    def _float(self, key: str, default: float) -> float:
        try:
            return float(self.cfg.get(key, default))
        except Exception:
            return default

    def validate_flow(self):
        # Phase coverage
        for i in range(1, 10):
            if not self.phases.get(f"p{i}", False):
                self.findings.append(Finding(
                    SEV_WARN, f"PHASE{i}_MISSING_END",
                    f"Phase {i} may not have completed (no END marker)."
                ))

        # MDF readiness gates
        if not self.flags_seen["prefetch_ok"]:
            self.findings.append(Finding(
                SEV_WARN, "MDF_PREFETCH_MISSING",
                "No 'OHLCV prefetch completed' found. Market data may be cold.",
                hint="Check Phase 5 prefetch and MDF readiness."
            ))
        if not self.flags_seen["market_ready"]:
            self.findings.append(Finding(
                SEV_WARN, "MDF_NOT_READY",
                "No MarketDataReady emission observed.",
                hint="Ensure Phase 5 emitted MarketDataReady and accepted symbols are set."
            ))

        # Symbols
        if self.symbol_counts:
            if max(self.symbol_counts) == 0:
                self.findings.append(Finding(
                    SEV_ERR, "NO_SYMBOLS",
                    "Accepted symbols were set to 0.",
                    hint="Discovery/DB snapshot may be empty; check Phases 3–4."
                ))
        else:
            self.findings.append(Finding(
                SEV_WARN, "NO_SYMBOL_LOG",
                "Did not see 'Accepted symbols set (N)' log lines.",
                hint="Phase 4/5 may not have set symbols; or log level filtered."
            ))

        # Agent & Meta loops
        if not self.flags_seen["agent_warm"]:
            self.findings.append(Finding(
                SEV_WARN, "AGENTS_NOT_WARMED",
                "No evidence that agents warmed up.",
                hint="Check Phase 6 warmup and AgentManager.warmup_all()."
            ))

        if not self.flags_seen["meta_loop"] and not self.flags_seen["meta_started"]:
            self.findings.append(Finding(
                SEV_ERR, "MC_LOOP_NOT_RUNNING",
                "MetaController loop not observed starting.",
                hint="Phase 6 should start MC; Phase 9 should wire it. Look for 'MetaController loop started'."
            ))

        if not self.flags_seen["exec_loop"]:
            self.findings.append(Finding(
                SEV_WARN, "EXEC_LOOP_NOT_RUNNING",
                "ExecutionManager loop not observed.",
                hint="Phase 9 should schedule an ExecutionManagerLoop (or process_queue_once poller)."
            ))

        if not self.flags_seen["tpsl_loop"]:
            self.findings.append(Finding(
                SEV_INFO, "TPSL_LOOP_NOT_SEEN",
                "TPSL loop not observed in logs (may be disabled or quiet)."
            ))

        if not self.flags_seen["posm_loop"]:
            self.findings.append(Finding(
                SEV_INFO, "POSM_LOOP_NOT_SEEN",
                "PositionManager loop not observed; verify it's scheduled in Phase 9."
            ))

        # Signals → Trades path sanity
        n_signals = sum(1 for ev in self.events if ev.get("ev") == "SIGNAL")
        n_exec_err = sum(1 for ev in self.events if ev.get("ev") == "EXEC_ERR")
        if n_signals == 0:
            self.findings.append(Finding(
                SEV_WARN, "NO_SIGNALS",
                "No SIGNAL events found. Agents/Meta may not be evaluating.",
                hint="Check AgentManager loop and MetaController.evaluate_once/evaluate loop."
            ))
        if n_exec_err > 0:
            self.findings.append(Finding(
                SEV_ERR, "EXEC_ERRORS",
                f"Detected {n_exec_err} execution error overlay events.",
                hint="Search logs for 'Decision execution failed' or 'EXEC_ARG_MISMATCH'."
            ))

    def validate_config(self):
        # Execution-blockers
        min_conf = self._float("MIN_EXECUTION_CONFIDENCE", 0.7)
        min_sig  = self._float("MIN_SIGNAL_CONF", 0.55)
        if min_conf > 0.9:
            self.findings.append(Finding(
                SEV_WARN, "CONFIDENCE_TOO_HIGH",
                f"MIN_EXECUTION_CONFIDENCE={min_conf} may be too strict.",
                hint="Try 0.65–0.75 to allow some trades while testing."
            ))

        # Notional thresholds
        min_order = self._float("MIN_ORDER_USDT", 5.0)
        spend     = self._float("TRADE_AMOUNT_USDT", 10.0)
        min_entry = self._float("MIN_ENTRY_QUOTE_USDT", 10.0)
        if spend < min_order or spend < min_entry:
            self.findings.append(Finding(
                SEV_WARN, "NOTIONAL_TOO_LOW",
                f"TRADE_AMOUNT_USDT={spend} is below exchange/min-entry thresholds.",
                hint=f"Raise TRADE_AMOUNT_USDT to ≥ max(MIN_ORDER_USDT={min_order}, MIN_ENTRY_QUOTE_USDT={min_entry})."
            ))

        # Mode expectations
        sim = self._bool("SIMULATION_MODE", False)
        paper = self._bool("PAPER_MODE", True)
        live = self._bool("LIVE_MODE", False)
        if live and sim:
            self.findings.append(Finding(
                SEV_WARN, "LIVE_PLUS_SIM",
                "LIVE_MODE=True together with SIMULATION_MODE=True.",
                hint="Choose one. Mixed modes can short‑circuit real execution."
            ))
        if not (sim or paper or live):
            self.findings.append(Finding(
                SEV_WARN, "ALL_MODES_OFF",
                "SIMULATION_MODE, PAPER_MODE, LIVE_MODE all false?",
                hint="At least one mode should be on; PAPER is safest to start."
            ))

        # Optional: execution patch guard
        if not self._bool("PATCH_EXEC", True):
            self.findings.append(Finding(
                SEV_INFO, "PATCH_EXEC_OFF",
                "PATCH_EXEC=False — if you see EXEC_ARG_MISMATCH('comment'), turn it on."
            ))

    # ---------------- Reporting ----------------
    def summarize(self):
        # Brief counts
        n_sig = sum(1 for ev in self.events if ev.get("ev") == "SIGNAL")
        sig_preview = [
            {"sym": ev.get("sym"), "act": ev.get("act"), "conf": ev.get("conf")}
            for ev in self.events if ev.get("ev") == "SIGNAL"
        ][:10]
        print("\n=== Flow Doctor Summary ===")
        print(f"Signals seen: {n_sig} {sig_preview}")
        print(f"Meta loop: {self.flags_seen['meta_loop'] or self.flags_seen['meta_started']}")
        print(f"Exec loop: {self.flags_seen['exec_loop']}")
        print(f"MDF ready: {self.flags_seen['market_ready']}  Prefetch: {self.flags_seen['prefetch_ok']}")
        if self.symbol_counts:
            print(f"Accepted symbols (max): {max(self.symbol_counts)}")
        print("\nFindings:")
        if not self.findings:
            print("  ✓ No obvious blockers detected.")
        else:
            for f in self.findings:
                line = f"- [{f.severity}] {f.code}: {f.message}"
                if f.hint:
                    line += f"  (hint: {f.hint})"
                print(line)

    def run(self):
        self.load_env()
        self.maybe_import_config()
        self.load_log()
        if self.lines:
            self.scan_lines()
        self.validate_flow()
        self.validate_config()
        self.summarize()


def main():
    ap = argparse.ArgumentParser(description="Octivault Flow Doctor")
    ap.add_argument("--log", help="Path to combined app log", default=os.getenv("FLOW_DOCTOR_LOG"))
    ap.add_argument("--env", help="Path to .env to read", default=None)
    ap.add_argument("--no-config-import", action="store_true",
                    help="Do not import core.config.Config() (use env only)")
    args = ap.parse_args()

    doctor = FlowDoctor(
        log_path=args.log,
        env_path=args.env,
        import_config=(not args.no_config_import)
    )
    doctor.run()

if __name__ == "__main__":
    main()
