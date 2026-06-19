# Funding-Carry — Go-Live Checklist

**Rule #1: do not skip a gate. Each section must pass before the next.**
Real money only after EVERY box in Phases 0–2 is checked.

---

## Phase 0 — Proof gates (must ALL be ✅ before any real money)
- [ ] **Forward edge proof:** `python3 carry_paper_trader.py report` shows **✅** (≥30 closed trades, positive avg net/trade)
- [ ] **Testnet execution:** full two-leg cycle validated (`testnet_validate_full.py` → ✅) — *already done*
- [ ] You have **read** and accept: this is a **thin edge (~1–3%/yr)**, not a printer

> If the proof shows ❌ → **STOP.** Carry doesn't hold forward. Do not deploy. (It just saved your capital.)

---

## Phase 1 — Capital + key setup (security is non-negotiable)
- [ ] Decide **max risk capital** = money you can lose *entirely* (start: **$200–500**)
- [ ] Create **REAL** Binance API keys with:
  - [ ] **Spot + Futures trading ENABLED**
  - [ ] **Withdrawals DISABLED** ← critical: a leaked key then can't steal funds
  - [ ] **IP whitelist** = this machine's IP
- [ ] Put keys in `.env` as `BINANCE_API_KEY` / `BINANCE_API_SECRET` (real, not testnet)
- [ ] Fund the account with ONLY the tiny risk capital. Keep everything else off-exchange.
- [ ] Move USDT into BOTH wallets (spot + USDⓈ-M futures) so both legs can trade.

---

## Phase 2 — First live (tiny size, supervised)
Set conservative limits in `.env`:
```
CARRY_MODE=live
CARRY_NOTIONAL=50            # $ per leg — tiny
CARRY_MAX_POSITIONS=3
CARRY_MAX_TOTAL_USD=300
CARRY_LEVERAGE=2
CARRY_MAX_DD_PCT=5          # auto-halt at 5% drawdown
CARRY_LIQ_BUFFER_PCT=15     # close if within 15% of liquidation
```
- [ ] **Arm it** (2nd gate): `touch logs/carry_live_armed`
- [ ] Start it and **watch the first trade live** (don't walk away):
  - [ ] First open: both legs fill, position is delta-neutral
  - [ ] Funding accrues over an 8h window as expected
  - [ ] First close: both legs unwind cleanly
- [ ] Run **2–4 weeks** at this size. Goal = confirm *live matches paper*, NOT profit.

**Emergency stop at any time:** `touch logs/carry.stop` → halts new entries + closes all.

---

## Phase 3 — Scale-up (only if Phase 2 held)
- [ ] Live net/trade tracks the paper proof (edge is real at small size)
- [ ] No execution surprises (fills, funding, margin all as modeled)
- [ ] **Double capital in steps** — `CARRY_NOTIONAL` / `CARRY_MAX_TOTAL_USD` — one level at a time
- [ ] At each level, watch for **capacity decay** (edge thins as size grows into small-cap funding spikes)
- [ ] Keep `report` running — **carry compresses over time**; size down / stop when it decays

---

## Always-on safety (built in)
| Control | How |
|---------|-----|
| Kill-switch | `touch logs/carry.stop` |
| Drawdown auto-halt | automatic at `CARRY_MAX_DD_PCT` |
| Liquidation guard | auto-closes legs near liquidation (live) |
| Re-arm required after kill | delete `logs/carry.stop`, recreate `logs/carry_live_armed` |
| Disarm live instantly | delete `logs/carry_live_armed` (orders stop, becomes blocked) |

## The non-negotiables
1. **Withdrawals OFF on the API key.** Always.
2. **Start so small that total loss is irrelevant.** $300 teaches the same lessons as $30k.
3. **Prove before deploy, start tiny, scale slow, kill fast.**
4. **It's a risk-managed operation, not a printer.** Respect your max-loss line.
