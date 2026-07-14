# Strategy Contribution Assessment — 5-Net-Profitable-Trades/Day Target

Scope: for each candidate strategy, is it running in the supervised native runtime today,
what real trade/signal volume exists in on-disk journals, what its proven edge quality is,
and whether it could structurally reach 5 net-profitable trades/day unmodified.

Evidence base: `trade_ledger.jsonl`, `trade_progress.json`, `runtime_state_snapshot.json`,
`logs/carry_ledger.jsonl`, `logs/carry_state.json`, `logs/run_latest.log` (current live
session, 2026-07-14), `logs/supervisor.log`, `ps aux` (live process check), `crontab -l`,
and both cited memory files in full (contents reproduced/quoted below, not paraphrased
from second-hand summary).

---

## (a) MLForecaster / legacy signal path

**Wiring status: ACTIVELY RUNNING, the only signal source wired into the live native
decision path.** Confirmed independently of the prior audit by a fresh live-process check
during this pass:

- `ps aux` shows `main.py` currently running (pid 54172, started 2:04 AM, `RN` state,
  22m10s CPU time at time of check) — this is a live, in-progress supervised session, not
  a historical artifact.
- `logs/run_latest.log` (current session) shows `sigs=15` on nearly every cycle
  (`octivault.main — cycle 00214 │ 571.0ms │ nav= 57.85 │ sigs=15 │ dec= 9 │ exe= 0`),
  i.e. MLForecaster is actively generating 15 signals/cycle, every cycle, right now.
- Decisions *do* reach the arbitration/decision stage this session (unlike the prior
  audit's dry-run, where PERSIST_GATE blocked everything upstream) — `dec=` ranges from
  0 to 13 across cycles, and 56 individual `allowed=True` BUY/SELL decision log lines
  exist in the current session log.
- Gating: of ~1,750 BUY-decision log lines, 1,694 are `allowed=False`; the dominant
  blocked reasons are `gate_2_confidence,gate_3_regime,gate_11_symbol_downtrend` (205
  occurrences) — i.e. confidence floor + regime + downtrend veto still doing most of the
  blocking, consistent with the prior audit's finding that the confidence floor is a
  defensive response to the no-edge result below.

**Opportunities/signals/trades attributable to it, from real journals:**

- `trade_ledger.jsonl`: 9 entries total, all dated 2026-06-13 (a single day, over a month
  before today's date 2026-07-14). 3 wins / 6 losses gross. No entries since. Confirmed
  these were the ONLY real trade records in the ledger — nothing has been appended since.
- `trade_progress.json` (`updated: 2026-06-15`, also stale — 1 month old): `total_trades:
  9`, `wins: 3`, `losses: 6`, `net_wins: 1`, `net_losses: 8`, `net_win_rate: 0.1111`,
  `cum_net_pnl_usdt: -0.5377`. Net of the 0.38%-round-trip fee assumption, only 1 of 9
  trades was actually profitable.
- `runtime_state_snapshot.json`: `realized_pnl: -57.67 USDT` cumulative, `positions_count:
  5` still open, matching the prior audit's "5 real open positions, 1 profitable / 4
  losing" finding — i.e. the account is still sitting on the same unresolved loss-making
  position set the prior audit flagged, not fresh new activity.
- **Today's live session (2026-07-14) produced zero real executions.** Every
  `execution_result` in the current run log is `NONE` (278 occurrences) except 4
  `FILLED` lines — inspection confirms those 4 are a hardcoded startup self-test/smoke-test
  block (`symbol: TESTUSDT`/`BTCUSDT`, `nav_usdt: 100.0` fixed, `timestamp: 1.0`,
  `loop_duration_ms: 12.5` identical every time), not real fills. Net real trades in
  today's live session: **0**.

**Known edge quality — quoted directly from
`memory/edge-verdict-no-edge.md`:**

> "The OctiVault bot has no demonstrated edge — this is proven, not opinion (2026-06-13)."
> Live: 69 round-trips → expectancy −0.033 USDT/trade; highest-confidence band (1.0–1.1,
> 51 trades) was 31% win / negative — confidence uncorrelated with profit. Backtest: 3,140
> samples / 38 models → net expectancy −0.27%/trade, win-rate 37%, every confidence bucket
> net-negative, and the high-confidence half (−0.287%) is *worse* than the low-confidence
> half (−0.262%) — confidence is mildly ANTI-predictive. Retested at 1h: still no edge
> (−0.36%/trade, 1,106 samples). "No execution/gate/fee/model-selection change can create
> edge... the signal itself is the problem." "Do NOT deploy capital on the current model."

The live 9-trade sample (net win-rate 11.1%, cum net −0.54 USDT) and the current session's
zero net trades are directionally consistent with — not contradictory to — this proven
no-edge verdict.

**Could this structurally produce 5 net-profitable trades/day, run continuously,
unmodified?** No. This is not a volume/wiring problem — it is running continuously right
now, generating 15 signals/cycle, and still producing net-negative-to-zero real trades.
A signal source with negative net expectancy (−0.27%/trade to −0.36%/trade across every
timeframe and confidence bucket tested) cannot be scaled into 5 net-*profitable*
trades/day by running it more; running it more only accelerates the existing loss rate,
which is presumably exactly why gate_2/gate_3/gate_11/the 0.95 confidence floor were added
— they are suppressing volume as a capital-preservation measure, and that suppression is
correctly diagnosed by the prior audit as a symptom of the no-edge finding, not a bug to
fix.

**What would need to change:** per the memory file's own conclusion, "the THESIS must
change" — this is edge-fix work, not wiring work. No amount of gate-tuning, timeframe
change (5m and 1h both tested, both negative), or execution-quality work will produce a
positive-expectancy signal from this model family. The path forward is a different alpha
thesis entirely (the memory file names funding-rate arb, stat-arb, cross-exchange,
event-driven as candidates — (b) and (c) below are exactly the first two of these,
already built).

---

## (b) Funding-carry (`funding_carry_backtest.py` / `carry_paper_trader.py`)

**Wiring status: NOT wired into the native supervised runtime, and currently NOT RUNNING
at all** (upgraded finding vs. the prior audit's "standalone, not wired" — this pass found
it isn't even running standalone right now):

- `ps aux` shows no `carry_paper_trader.py` or `carry_supervisor.sh` process active.
- `logs/carry_ledger.jsonl` — the paper-trading forward-proof journal — has exactly 2
  closed trades, both dated 2026-06-30 (SYNUSDT, held 2.5h and 1.5h). Nothing since.
  Running `python3 carry_paper_trader.py report` confirms: *"CARRY PAPER — FORWARD TRACK
  RECORD (2 closed trades) ... Avg net/trade: −0.2209% win-rate: 0% cum: −0.44% ...
  VERDICT: ⏳ INCONCLUSIVE — 2/30 trades. Keep running."*
- `logs/carry_state.json` (`{"open": {}}`, mtime 2026-07-05) shows no open positions and
  no activity for the last ~9 days relative to today (2026-07-14) — the keep-alive
  supervisor (`carry_supervisor.sh`, designed to auto-restart the daemon "so the forward
  proof keeps accumulating unattended") is itself not running, so the daemon silently
  stalled at 2 trades instead of the ≥30 needed for a verdict.
- No crontab entry, no supervisor.sh integration, no reference anywhere in
  `core_engine/` or `main.py` — confirmed via grep, matching the prior audit.
- Note: the 2 real paper trades that *did* execute were both net-negative (−0.22% avg,
  0% win), but n=2 is statistically meaningless and does not update the backtest-based
  edge assessment below either way.

**Known edge quality — quoted directly from
`memory/funding-carry-edge-candidate.md`:**

> "Funding-rate carry (delta-neutral) is the first strategy with a positive edge
> signature." Backtest across 361 spot-hedgeable perps (the realistic universe, since
> delta-neutral needs a spot leg): **+1.22%/trade, 80% win, ~+3.2%/yr/symbol.** "Edge
> SURVIVED the hedgeability filter (even improved)." Liquidity-haircut version (liquid,
> $50M/24h, spot-hedgeable, 0.24% cost): +0.90%/trade, 73% win, but only 26 trades
> ("borderline... THIN and sparse"). Execution is fully built and testnet-validated
> (real two-leg delta-neutral cycle, open and close, on Binance testnets). The **only**
> remaining gate before live capital is forward paper proof (≥30 trades, net+) — which,
> per the ledger above, is stalled at 2/30 because the keep-alive daemon isn't running.
> Caveats explicitly flagged as still unmodeled: survivorship bias, slippage on illiquid
> alts (the biggest winners are the least liquid), and liquidation/basis risk on the
> short-perp leg during violent funding spikes. "Edge is THIN (~1-3%/yr). Do NOT deploy
> capital until forward proof ✅ AND testnet-validated."

**Could this structurally produce 5 net-profitable trades/day, run continuously,
unmodified?** No, and not for a wiring reason — for a trade-frequency reason inherent to
the strategy. Even at the most optimistic backtest rate (361 symbols, all showing
qualifying funding events over ~333 days), this is an opportunistic, event-driven strategy
gated on extreme funding-rate divergence — the memory file itself notes majors-only
funding events run "~2 trades/yr/symbol" and even the full universe's richer version is
inherently sparse and lumpy (concentrated in "extreme-funding events," not steady daily
flow). 5 *net-profitable* trades/day, every day, is not consistent with a strategy whose
own author describes it as "THIN and sparse" at ~26-361 trades across an entire year
across the full scanned universe. It could contribute *some* trades on days with funding
dislocations, but not a reliable 5/day floor, unmodified.

**What would need to change:** this is wiring + patience work, not edge-fix work — the
opposite profile from (a). Concretely: (1) get `carry_supervisor.sh` running again so the
forward paper proof resumes accumulating past 2/30 trades; (2) reach the ≥30-trade,
net-positive gate; (3) integrate execution into the supervised runtime (it currently runs
as a fully separate process/script with its own env vars, arm file, and kill-switch, not
through `bootstrap.py`/`NativeOrchestrator`); (4) even fully wired and proven, expect it to
supply an intermittent trickle of trades on high-funding-dislocation days, not a dependable
daily 5-trade quota by itself — it would need to run alongside another volume source to
hit a 5/day target, not replace one.

---

## (c) `statarb_discover.py` (stat-arb / pairs mean-reversion)

**Wiring status: NOT wired into the native runtime, not running at all, and has no live
or paper track record of any kind** — this is a discovery/backtest script only, one level
further from production than (b):

- No process running (`ps aux` clean), no crontab entry, no journal file exists anywhere
  in the repo (`find . -iname "*statarb*"` returns only the script itself — no ledger, no
  state file, no output artifact of any run). Confirmed via grep that `core_engine/` and
  `main.py` contain zero references to it, same as (b).
- The script is explicitly a discovery/falsification tool, not a trading daemon: it
  screens `SYMBOLS` (30 majors) for cointegrated pairs, selects pairs on the first half of
  history, and out-of-sample-tests on the second half, printing a verdict — there is no
  `carry_paper_trader.py`-equivalent forward-proof daemon for stat-arb at all. Even the
  "live-readiness" infrastructure that exists for funding-carry (execution modes,
  kill-switch, arm file, testnet validation) has no stat-arb counterpart in this repo.

**Known edge quality — quoted directly from
`memory/funding-carry-edge-candidate.md`** (which is also the authoritative source for
this strategy's status, since it has no dedicated memory file of its own):

> "Other strategies tested & DEAD: directional ML, stat-arb (see [[edge-verdict-no-edge]])."

The `edge-verdict-no-edge.md` memory file itself does not contain stat-arb-specific
numbers (its content is entirely about the directional GRU/OHLCV model); the only written
record of a stat-arb verdict is the one-line "tested & DEAD" cross-reference in the
funding-carry file above. This pass could not locate any stat-arb backtest output file,
report, or ledger in the repository to independently verify the sample size or magnitude
behind that verdict — it is asserted in memory but not backed by an artifact on disk the
way (a) and (b) are (`backtest_edge.py`/`edge_report.py` outputs and
`funding_carry_backtest.py`/`carry_ledger.jsonl` respectively). This is flagged as a
documentation gap: the "DEAD" verdict for stat-arb should be treated as directionally
credible (it came from the same investigation session as the other two, which are
well-evidenced) but is not independently re-verifiable from artifacts in this repo as it
stands.

**Could this structurally produce 5 net-profitable trades/day, run continuously,
unmodified?** No — per the only evidence available (the "tested & DEAD" note), this
strategy was tried and abandoned for lack of edge, same disposition as (a) but with less
on-disk documentation of why. Zero live or paper trades exist to even estimate a
structural trade-frequency ceiling the way (b)'s backtest allows.

**What would need to change:** if this is to be reconsidered, it needs the same treatment
funding-carry got — a written backtest report with sample size, edge magnitude, and
caveats — before deciding whether it's an edge-fix dead end (like (a)) or a wiring-only
gap (like (b)). Right now it is undifferentiable from "abandoned, no edge" without
re-running `statarb_discover.py` and capturing output; this audit did not re-run it, since
doing so would be new backtest work outside this pass's scope (assessing existing
contribution, not generating new edge evidence).

---

## Summary table

| Strategy | Running now? | Real trades in journals | Proven edge | Could hit 5 net-profitable trades/day unmodified? | What's needed |
|---|---|---|---|---|---|
| (a) MLForecaster (live native path) | **Yes** — actively generating 15 sig/cycle right now | 9 trades total, all 2026-06-13 (1 month stale); 0 real fills in today's live session | **Proven negative** (−0.27%/trade backtest 3,140 samples, −0.033 USDT/trade live 69 round-trips, confidence anti-predictive, 5m and 1h both tested) | **No** — running continuously would scale losses, not profit | Edge-fix (new thesis), not wiring |
| (b) Funding-carry | **No** — daemon and its keep-alive supervisor both currently stopped; last trade 2026-06-30 | 2 paper trades ever (stalled at 2/30 gate), both net-negative (n too small to mean anything) | **Proven positive** in backtest (+1.22%/trade, 80% win, 361 spot-hedgeable perps), but forward-proof gate not yet met and strategy is inherently sparse/event-driven | **No** — even if proven and wired, structurally an intermittent-opportunity strategy, not a daily-volume one | Wiring + resume paper-proof daemon + integrate into supervised runtime |
| (c) statarb_discover.py | **No** — no process, no journal, no output artifact anywhere in repo | 0 — never run to produce a saved result in this repo | Asserted "DEAD" in memory (cross-reference only, no on-disk backtest artifact found this pass) | **No** — no evidence exists it ever had edge | Needs its own documented backtest before any wiring decision is meaningful |

## Bottom-line, stated plainly

The only strategy wired into the live, currently-running native decision path
(MLForecaster) has a proven negative edge and is producing zero net-profitable real
trades even while actively running. The one strategy in this repo with a proven positive
edge (funding-carry) is not wired into the native runtime, and its own forward-proof
daemon — the last gate before it would even be considered for real capital — is not
currently running either, stalled at 2 of the required 30 trades. Neither path, as it
stands today, can produce 5 net-profitable trades/day: (a) because the signal has no
edge to scale, and (b)/(c) because they are either not running, not proven forward, or
structurally too sparse to supply daily volume even if fully wired and validated. Closing
this gap requires two different kinds of work in parallel — a new/fixed signal thesis for
volume-capable daily trading, and wiring + forward-proof completion for the
already-positive-edge but low-frequency carry strategy — not a single fix to either.
