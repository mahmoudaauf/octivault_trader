# 🎨 UI Design Review — AI Trading Command Center
> **Designer Review** · Branch: `phase-3/wiring` · June 1, 2026

---

## 📋 Executive Design Summary

The current dashboard is **functionally complete but visually immature**. It reads as a developer-built interface — the dark theme is solid, the data model is rich, but the visual hierarchy is flat, the layout has tension points, and the interaction design is inconsistent. This review audits every component and provides a concrete redesign direction.

**Verdict**: The foundations are correct. The system needs a design pass — not a rebuild.

---

## 🗺️ Current Layout Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│  HEADER  ·  🚀 AI Trading Command Center          ● Connected / ✗     │
├─────────────────────────────────────────────────────────────────────────┤
│  SYSTEM STATE BAR  (8-column grid)                                      │
│  NAV · 24h% · Free · Positions · Mode · Regime · Health · Throttle     │
├──────────────────────────────┬──────────────────────────────────────────┤
│  AI BRAIN PANEL              │  CAPITAL HEALTH PANEL                   │
│  Symbol · Action badge       │  3 SVG rings (Free/Active/Reserve)      │
│  Signals list                │  3 progress bars                        │
│  Gate checks                 │  Warnings                               │
│  Playbook · Confidence       │                                         │
├──────────────────────────────┴──────────────────────────────────────────┤
│  PORTFOLIO CARDS (2/3 width)         │  ACTIVITY TIMELINE (1/3 width)  │
│  Summary stats row                   │  max-h-96 scroll                │
│  md:grid-cols-2 position cards       │  20 events, newest first        │
├──────────────────────────────────────┴─────────────────────────────────┤
│  GOVERNANCE PANEL  (full-width)                                         │
│  Trading Controls · Emergency · Recovery                                │
├─────────────────────────────────────────────────────────────────────────┤
│  SYSTEM HEALTH  (conditional, 4-col grid)                               │
├─────────────────────────────────────────────────────────────────────────┤
│  FOOTER                                                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Component-by-Component Audit

---

### 1. Header
**File**: `pages/index.tsx` · Lines 65–80

#### ✅ What works
- Clean minimal header
- Connection status with animated pulse dot is a good pattern

#### ❌ Issues
| # | Issue | Severity |
|---|-------|----------|
| 1 | No last-refresh timestamp shown | Medium |
| 2 | Emoji `🚀` feels too casual for a professional trading tool | Low |
| 3 | No visual distinction between paper/live mode in header | **High** |
| 4 | Disconnected state shows `bg-red-500` dot only — no banner, no recovery button | Medium |
| 5 | `text-gray-500` subtitle has very low contrast (< 4.5:1 ratio) | Low |

#### 🎨 Design Direction
```
┌──────────────────────────────────────────────────────────────────────────┐
│  ◈ OCTIVAULT TRADER                    MODE: LIVE ●   ● 2s ago    [⚙]  │
│    Autonomous Trading System                                             │
└──────────────────────────────────────────────────────────────────────────┘
```
- Replace emoji with geometric symbol `◈` or wordmark
- Add `MODE: LIVE / PAPER` badge with red/blue distinction
- Show `● Xs ago` last-data timestamp instead of just connected/disconnected
- Add subtle settings gear

---

### 2. SystemStateBar
**File**: `components/SystemStateBar.tsx` · 139 lines

#### ✅ What works
- 8 key metrics in one scannable row — right idea
- Color-coded values per status type
- Tooltips on each metric
- `animate-fadeIn` on value changes

#### ❌ Issues
| # | Issue | Severity |
|---|-------|----------|
| 1 | `grid-cols-4` on mobile collapses into cramped 2×4 — emoji icons overlap with labels | **High** |
| 2 | Metric sizes are inconsistent — NAV/Growth have `text-2xl`, Mode/Regime have `text-sm` — creates visual imbalance | **High** |
| 3 | 8 columns in one row creates extreme horizontal density — no breathing room | Medium |
| 4 | Emoji icons are decorative noise — `💵 📈 💧` don't add meaning at glance speed | Low |
| 5 | `bg-gray-900 border-b border-gray-700` — the bar blends into the page background | Medium |
| 6 | No visual grouping — financial metrics (NAV, growth, capital) mixed with system state metrics (health, throttle) | **High** |
| 7 | Throttle timer countdown not rendered when `throttle_until_ts` is set | Medium |

#### 🎨 Design Direction
Split into two logical rows or use a **pill-style segmented bar**:

```
┌────────────────────────────────────────────────────────────────────────┐
│  PORTFOLIO                          SYSTEM STATE                       │
│  $12,450.23  +2.41%  $340 free  ·  LIVE · TRENDING · HEALTHY · CLEAR  │
│  ──────────────────  ─────────────────────────────────────────────     │
│   NAV         24h    Free            Mode   Regime   Health  Throttle  │
└────────────────────────────────────────────────────────────────────────┘
```
- Left group: **Portfolio** (NAV, growth, free capital, positions)
- Right group: **System State** (mode, regime, health, throttle) — as compact status pills
- Large `text-3xl` for NAV and 24h% — these are the hero numbers
- Replace emojis with minimal SVG icons or remove entirely

---

### 3. AIBrainPanel
**File**: `components/AIBrainPanel.tsx` · 139 lines

#### ✅ What works
- Action badge with status color is the right design pattern
- `animate-pulse` on active BUY/SELL badge draws attention correctly
- `blocked_reason` red banner is well-placed and properly alarming
- Gate checks pass/fail list is clean
- Playbook + Confidence footer section has good information architecture

#### ❌ Issues
| # | Issue | Severity |
|---|-------|----------|
| 1 | Symbol shown as `text-2xl` but action badge is also large — they compete visually | Medium |
| 2 | Signals list has duplicate confidence display: once inline `(65%)` and again right-aligned `65%` | Medium |
| 3 | Gate checks show `✓ PASS` and `✗ FAIL` in `animate-pulse` — passed gates should NOT pulse | Medium |
| 4 | No timestamp shown — "when was this last decision?" is unanswered | **High** |
| 5 | No "NO DECISION" empty state when `decision.action === "NONE"` — falls through to generic badge | Low |
| 6 | Panel title "AI Brain" is generic — consider "Last Decision" or "Signal Interpreter" | Low |
| 7 | When `signals.length === 0` the panel is nearly empty — needs an idle state | Medium |

#### 🎨 Design Direction
```
┌──────────────────────────────────────────────────────────────────┐
│  SIGNAL INTERPRETER                              2 mins ago      │
│  ─────────────────────────────────────────────────────────────   │
│  BTCUSDT                                   ┌──────────────┐      │
│                                            │  ▲  BUY      │      │
│                                            └──────────────┘      │
│  ─────────────────────────────────────────────────────────────   │
│  SIGNALS                          GATES                          │
│  RSI   ▲ UP   87%                 ✓ CAPITAL_GUARD               │
│  MAHAL ▲ UP   72%                 ✓ THROTTLE_GUARD              │
│  VOL   ─ N/A  —                   ✗ RISK_GUARD                  │
│  ─────────────────────────────────────────────────────────────   │
│  Playbook: MOMENTUM_BREAKOUT              Confidence: 79%        │
└──────────────────────────────────────────────────────────────────┘
```
- Side-by-side Signals and Gates columns (better use of horizontal space)
- Remove duplicate confidence value
- Only pulse the action badge, not gate results
- Add `formatDistanceToNow(decision.timestamp)` to top-right

---

### 4. CapitalHealthPanel
**File**: `components/CapitalHealthPanel.tsx` · 216 lines

#### ✅ What works
- SVG circular rings are the best visual element in the dashboard — unique and informative
- Three-ring approach (Free/Active/Reserve) clearly communicates allocation
- Progress bars for risk metrics are appropriate
- Warnings list with `⚠️` is properly alarming

#### ❌ Issues
| # | Issue | Severity |
|---|-------|----------|
| 1 | `w-20 h-20` rings are too small — the percentage text inside is barely readable | **High** |
| 2 | Three identical rings with different colors but no combined/donut view — harder to compare | Medium |
| 3 | `animate-pulse` on state badge (CAUTION/WARNING) — this pulses even when state is HEALTHY | Medium |
| 4 | `dust_ratio` progress bar is shown but `dust_ratio` is not in the SVG rings — inconsistent | Low |
| 5 | No dollar amounts shown alongside percentages — `32%` means nothing without `$4,120` | **High** |
| 6 | Warnings section at bottom is easily missed — no count badge on panel header | Medium |
| 7 | Ring labels "Free / Active / Reserve" are `text-xs text-gray-500` — barely visible | Low |

#### 🎨 Design Direction
```
┌──────────────────────────────────────────────────────────────────┐
│  CAPITAL ALLOCATION               ●  HEALTHY    ⚠️ 1 warning    │
│  ──────────────────────────────────────────────────────────────  │
│                                                                  │
│   ╔══════════════════════╗    Free       $4,120   32%  ████░░   │
│   ║  ░░░░░░░░░░░░░  32% ║    Active     $7,800   61%  ███████  │
│   ║  ░░░  FREE   ░░░    ║    Reserve    $530      4%  ░░░░░░░  │
│   ║  ░░░░░░░░░░░░░       ║    Dust       $200      2%  ░░░░░░░  │
│   ╚══════════════════════╝                                       │
│   (single stacked donut)                                         │
│  ──────────────────────────────────────────────────────────────  │
│  Exposure: 68%  ████████░░    Largest Position: 22%  ████░░░░  │
└──────────────────────────────────────────────────────────────────┘
```
- Replace 3 separate rings with **one stacked donut chart** — more information, less space
- Show dollar amounts alongside percentages everywhere
- Pulse only WARNING and CRITICAL states, not HEALTHY
- Warning count badge on panel header

---

### 5. PortfolioCards
**File**: `components/PortfolioCards.tsx` · 188 lines

#### ✅ What works
- `md:grid-cols-2` card layout is correct — good density
- Status badge + AI action tag is the right UX pattern
- Summary stats row (Total/PnL/PnL%/NAV) at top is good
- `hover:scale-[1.02]` is a nice subtle interaction
- CRITICAL `animate-pulse border-2 border-red-500` critical alert is well designed

#### ❌ Issues
| # | Issue | Severity |
|---|-------|----------|
| 1 | Each card has 6 stacked data sections — very tall cards, requiring excessive scrolling | **High** |
| 2 | `text-current/70` for labels — `text-current` with opacity is unreliable across status backgrounds | Medium |
| 3 | Entry price and current price are `text-sm font-mono` — should be equally prominent as quantity | Low |
| 4 | `AI Action → TAKE_PROFIT` shown at bottom of card — the most actionable info buried last | **High** |
| 5 | PnL% is `text-lg font-bold` but PnL absolute value is `text-sm` — opposite priority | Medium |
| 6 | No sort/filter controls — 10+ positions with no way to sort by PnL, status, etc. | Medium |
| 7 | No visual trend indicator (up/down arrow with context on how position is moving) | Low |
| 8 | Summary stats use `bg-gray-900 px-3 py-1 rounded` pill style — inconsistent with card styles | Low |

#### 🎨 Design Direction
Redesign cards to be **wider, shorter** with a horizontal layout:

```
┌───────────────────────────────────────────────────────────────────────┐
│  BTCUSDT              LEADER        ↑ TAKE PROFIT @ $68,400          │
│  0.2340 BTC · $63,248             Entry: $61,100  · +$494  (+3.52%)  │
│  ══════════════════════════════════════════════════════════════════    │
│  [Progress bar showing current vs entry vs TP target]                 │
└───────────────────────────────────────────────────────────────────────┘
```
- **Top row**: Symbol + Status badge (left) · AI Action (right, always visible)
- **Middle row**: Quantity · Current Price · Entry Price · PnL (all on one line)
- **Bottom**: Visual price progress bar — current relative to entry/TP
- Add sort tabs: `All · Leaders · Recovering · Dust`

---

### 6. ActivityTimeline
**File**: `components/ActivityTimeline.tsx` · 151 lines

#### ✅ What works
- Emoji-based event type icons are immediately scannable
- `formatDistanceToNow` relative timestamps are user-friendly
- `max-h-96 overflow-y-auto` scroll constraint prevents layout overflow
- `hover:bg-gray-700/40` hover state is tasteful

#### ❌ Issues
| # | Issue | Severity |
|---|-------|----------|
| 1 | `animate-pulse` on event type color — every event pulses forever, making it visually noisy | **High** |
| 2 | Events use `gap-4 pb-4 border-b border-gray-700` — neutral visual weight, no way to see recency at a glance | Medium |
| 3 | `w-12` left column for icon+time is too narrow — timestamp text truncates | Medium |
| 4 | No visual timeline line connecting events (vertical connector line) | Low |
| 5 | PnL on events (`event.pnl`) not rendered anywhere in the component | Medium |
| 6 | No event count badge on panel header | Low |
| 7 | `recentEvents.slice(-20)` — arbitrary 20-event limit with no "load more" | Low |

#### 🎨 Design Direction
```
┌──────────────────────────────────────────────┐
│  ACTIVITY LOG                  20 events  ↓  │
│  ──────────────────────────────────────────  │
│  │                                           │
│  ●──── 🧠 DECISION  BTCUSDT  BUY            │
│  │     "RSI breakout + momentum" · 2m ago   │
│  │     Confidence: 79%                       │
│  │                                           │
│  ●──── ⚡ EXECUTION  BTCUSDT                │
│  │     Order placed · 2m ago                │
│  │                                           │
│  ●──── ✓ FILL  BTCUSDT  +$12.40            │
│  │     Filled at $63,248 · 3m ago           │
│  │                                           │
│  ●──── 🔒 THROTTLE                          │
│        API limit hit · 5m ago               │
└──────────────────────────────────────────────┘
```
- Vertical timeline line connecting events
- Only the **most recent** event pulses (not all)
- Show PnL on FILL events
- Event count badge on header
- Fixed-width timestamp column

---

### 7. GovernancePanel
**File**: `components/GovernancePanel.tsx` · 241 lines

#### ✅ What works
- Three-section layout (Trading / Emergency / Recovery) is correct information architecture
- Confirmation modal pattern for critical actions is exactly right
- Button disable during loading with `Processing...` state is correct
- `animate-pulse` on confirm button as a "danger" signal is appropriate
- Footer safety notice is a good UX detail

#### ❌ Issues
| # | Issue | Severity |
|---|-------|----------|
| 1 | `⚠️ HALT ALL` button is `text-lg font-bold` — draws too much attention at rest | Medium |
| 2 | Emergency section has only 2 buttons in `grid-cols-2` — "Cancel Orders" and "HALT ALL" look like equal severity | **High** |
| 3 | Recovery section has 1 button in `grid-cols-2 md:grid-cols-3` — lonely and orphaned | Medium |
| 4 | Confirmation modal body text is hardcoded per action with if-chain — brittle | Low |
| 5 | No last-action log shown — "What was the last governance action?" is unanswered | Medium |
| 6 | All sections at equal visual weight — emergency doesn't feel visually distinct | **High** |
| 7 | `fixed inset-0 bg-black/70` modal has no keyboard trap — Tab can escape the modal | Medium |

#### 🎨 Design Direction
```
┌──────────────────────────────────────────────────────────────────────┐
│  GOVERNANCE CONTROLS  🛡️                                            │
│                                                                      │
│  TRADING MODE                                                        │
│  [Pause Buying]  [Resume Buying]  [Safe Mode]  [Resume Normal]      │
│                                                                      │
│  ╔══════════════════════════════════════════════════════════════╗    │
│  ║  EMERGENCY ZONE                                              ║    │
│  ║  [  Cancel All Orders  ]    [  ⚠️  HALT ALL TRADING  ]     ║    │
│  ╚══════════════════════════════════════════════════════════════╝    │
│                                                                      │
│  RECOVERY                                                            │
│  [Resume Trading]                                                    │
│                                                                      │
│  Last action: PAUSE_BUYING · 14 mins ago · ✓ Success               │
└──────────────────────────────────────────────────────────────────────┘
```
- Emergency zone with **distinct visual container** (red border/background)
- Make HALT ALL larger than Cancel Orders — they're not equal
- Show last action + result + timestamp
- Keyboard trap on modal

---

### 8. System Health Panel
**File**: Inline in `pages/index.tsx` · Conditional render

#### ✅ What works
- Conditional render (only shows when data exists)
- Green/yellow/red border coding is clear

#### ❌ Issues
| # | Issue | Severity |
|---|-------|----------|
| 1 | Entire panel is anonymous inline JSX — should be a `SystemHealthPanel` component | Low |
| 2 | No `error_count` or `last_error` displayed — the most actionable fields on `ComponentHealth` are unused | **High** |
| 3 | `last_check_ts` not shown — "how stale is this health data?" is unanswered | Medium |
| 4 | Static 4-column grid — breaks on narrow screens with long component names | Low |

#### 🎨 Design Direction
Extract to component. Show:
- Component name + status pill (top)
- Error count (if > 0, show red badge)
- Last error message (truncated, expandable)
- Last check relative timestamp

---

## 🎨 Design System Critique

### Typography
| Issue | Current | Recommended |
|-------|---------|-------------|
| Scale inconsistency | NAV `text-2xl`, Mode `text-sm`, PnL% `text-lg` mixed arbitrarily | Define 3 tiers: Hero (3xl), Feature (xl), Data (base) |
| Font stack | Long monospace fallback chain | Keep — this is correct for a trading terminal |
| Labels | `text-xs text-gray-500 uppercase` — all labels look identical | Distinguish panel headers from field labels |

### Color System
| Issue | Current | Recommended |
|-------|---------|-------------|
| Green/Red overuse | Both P&L and system health use same green/red | Use distinct hues — emerald for gains, red for losses, green for healthy system |
| Gray monotony | `gray-700/gray-800/gray-900/gray-950` — panels blend together | Add 1px `border-gray-700` and subtle `bg-opacity` differences per panel depth |
| Status semantics | `animate-pulse` on EVERYTHING — HEALTHY, PASS, active values all pulse | **Pulse should mean: live-changing data only.** Remove from static pass/fail states |

### Spacing & Density
| Area | Issue |
|------|-------|
| SystemStateBar | 8 columns with `gap-8` — extremely dense, no visual grouping |
| PortfolioCards | Cards are very tall for the amount of data |
| GovernancePanel | Sections use `mb-8 pb-8` — excessive vertical padding |
| Page Layout | `p-6 space-y-6` is consistent and correct ✅ |

### Animation
**Current**: `animate-pulse` appears on ~15 distinct elements simultaneously — creates visual noise

**Rule**: Pulse = **live, actively changing data** only.

| Element | Current | Correct |
|---------|---------|---------|
| BUY/SELL badge | Pulse ✅ | Keep |
| HEALTHY status text | Pulse ❌ | Remove |
| Gate PASS result | Pulse ❌ | Remove |
| blocked_reason banner | Pulse ✅ | Keep |
| Confirm button | Pulse ✅ | Keep |
| Activity event colors | Pulse ❌ | Only most recent event |
| Capital state badge (HEALTHY) | Pulse ❌ | Remove |
| PnL value | Pulse (via `animate-fadeIn`) ✅ | Keep only on change |

---

## 🔌 Data / UX Gaps

These are data fields available in `lib/types.ts` that are **not rendered anywhere** in the UI:

| Field | Type | Located In | Currently Rendered? |
|-------|------|-----------|-------------------|
| `throttle_until_ts` | `number?` | `SystemStatus` | ❌ No countdown timer |
| `api_weight_estimate` | `number` | `SystemStatus` | ❌ Not shown |
| `open_orders_count` | `number` | `SystemStatus` | ❌ Not in SystemStateBar |
| `locked_usdt` | `number` | `SystemStatus` | ❌ Not shown |
| `event.pnl` | `number?` | `ActivityEvent` | ❌ Not rendered on FILL events |
| `error_count` | `number` | `ComponentHealth` | ❌ Not shown |
| `last_error` | `string?` | `ComponentHealth` | ❌ Not shown |
| `last_check_ts` | `number` | `ComponentHealth` | ❌ Not shown |
| `signal.reason` | `string?` | `SignalView` | ✅ Partial (`text-xs text-gray-500`) |
| `gate.reason` | `string?` | `GateResult` | ❌ Not shown on FAIL gates |

---

## 🏗️ Proposed Redesigned Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  ◈ OCTIVAULT  ·  LIVE MODE                    ● 1s ago   [⚙]      │
├──────────────────────────────────────────────────────────────────────┤
│  $12,450.23   +2.41%   |   $340 free · 8 pos · 3 orders            │
│  ──── PORTFOLIO ─────────  ──── SYSTEM ────────────────────────────  │
│                             TRENDING · HEALTHY · CLEAR               │
├──────────────────────────┬───────────────────────────────────────────┤
│  SIGNAL INTERPRETER      │  CAPITAL ALLOCATION                      │
│  BTCUSDT  [▲ BUY]  2m ago│  ╔══════╗  Free   $4,120  32%  ████░░  │
│                          │  ║      ║  Active  $7.8k  61%  ███████  │
│  Signals  │  Gates       │  ╚══════╝  Reserve  $530   4%  ░        │
│  RSI  87% │ ✓ CAP_GUARD  │                                          │
│  MAH  72% │ ✓ THR_GUARD  │  Exposure 68% ████████░  LPos 22% ████  │
│  VOL  —   │ ✗ RSK_GUARD  │                                          │
│           │              │  ⚠️ 1 warning: exposure near limit       │
│  MOMENTUM_BREAKOUT  79%  │                                          │
├──────────────────────────┴───────────────────────────────────────────┤
│  POSITIONS (8)                    All · Leaders · Recovering · Dust  │
│                                                                      │
│  BTCUSDT  LEADER  ↑ TAKE PROFIT   0.234 · $63,248 · +$494 +3.52%  │
│  ─────────────────────────────────────────────────────────────────   │
│  ETHUSDT  RECOVERING  WAIT         1.21 · $2,410  ·  -$34  -1.2%  │
│  XRPUSDT  DUST  CLEAN DUST          450 · $0.58   ·   -$2  -0.8%  │
├──────────────────────────┬───────────────────────────────────────────┤
│  ACTIVITY LOG  (20)      │  GOVERNANCE CONTROLS  🛡️               │
│                          │                                           │
│  │ 🧠 BUY BTCUSDT  2m   │  [Pause Buy] [Resume] [Safe] [Normal]   │
│  │ ⚡ ORDER  2m          │                                           │
│  │ ✓ FILL +$12  3m      │  ╔═════════════════════════════════╗     │
│  │ 🔒 THROTTLE  5m      │  ║  [Cancel Orders]  [⚠ HALT ALL] ║     │
│  │ 🧠 HOLD ETHUSDT  8m  │  ╚═════════════════════════════════╝     │
│                          │                                           │
│                          │  [Resume Trading]                        │
│                          │  Last: PAUSE_BUYING  14m · ✓            │
├──────────────────────────┴─────────────────────────────────────────┤
│  SYSTEM HEALTH                                                        │
│  [ENGINE ● HEALTHY]  [API ● DEGRADED · 3 errors]  [DB ● HEALTHY]   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Redesign Priorities

### Priority 1 — Critical UX Fixes (Do First)
- [ ] **Remove pulse from static elements** — only active/live-changing data pulses
- [ ] **Render missing data fields** — `throttle_until_ts` countdown, `gate.reason` on FAIL, `event.pnl` on FILL, `error_count`/`last_error` in health
- [ ] **AI Action elevation** — move AI recommended action to top of each position card
- [ ] **Emergency zone distinction** — HALT ALL needs to be visually distinct from other buttons

### Priority 2 — Visual Hierarchy (Do Second)
- [ ] **Unify metric font sizes** — define `Hero / Feature / Data` tiers and apply consistently
- [ ] **SystemStateBar grouping** — split Portfolio metrics vs System State metrics
- [ ] **Add dollar amounts to CapitalHealthPanel** — ratios without absolutes are ambiguous
- [ ] **Position card compactness** — horizontal layout to reduce card height

### Priority 3 — Polish & Completeness (Do Third)
- [ ] **Vertical timeline line** in ActivityTimeline
- [ ] **Last governance action log** in GovernancePanel
- [ ] **Mode badge in header** (LIVE/PAPER) with distinct color
- [ ] **Last-updated timestamp** in header
- [ ] **`open_orders_count`** added to SystemStateBar
- [ ] **Warning count badge** on CapitalHealthPanel header
- [ ] **Keyboard trap** in confirmation modal
- [ ] **Extract** SystemHealth to its own component

---

## 🎨 Proposed Token Updates

### Animation
```css
/* Only on live-changing values */
.live-pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }

/* One-time fade-in on data update */
@keyframes value-in {
  from { opacity: 0; transform: translateY(-4px); }
  to   { opacity: 1; transform: translateY(0); }
}
.value-update { animation: value-in 0.3s ease-out; }
```

### Color Semantic Additions
```css
/* Distinguish financial gain/loss from system health */
--color-gain:    #34d399;  /* emerald-400 — profit */
--color-loss:    #f87171;  /* red-400     — loss */
--color-healthy: #4ade80;  /* green-400   — system OK */
--color-alert:   #fb923c;  /* orange-400  — caution */
--color-danger:  #ef4444;  /* red-500     — critical */
--color-live:    #22d3ee;  /* cyan-400    — live mode */
--color-paper:   #818cf8;  /* indigo-400  — paper mode */
```

### Panel Depth
```css
/* Create visual depth hierarchy */
body:       bg-gray-950
page:       bg-gray-950
panel:      bg-gray-900  border border-gray-800
card:       bg-gray-800  border border-gray-700
input/tag:  bg-gray-700  border border-gray-600
```

---

## 📊 Design Score

| Category | Score | Notes |
|----------|-------|-------|
| Information Architecture | 7/10 | Right data, some buried items |
| Visual Hierarchy | 5/10 | Flat — too many elements at equal weight |
| Typography | 6/10 | Inconsistent sizing across components |
| Color Semantics | 6/10 | Overloaded green/red, pulse overuse |
| Interaction Design | 7/10 | Good patterns, some keyboard gaps |
| Responsiveness | 6/10 | Mobile has compression issues in StateBar |
| Accessibility | 5/10 | aria-labels present but low contrast issues |
| Data Completeness | 5/10 | Several populated API fields not displayed |
| Component Architecture | 8/10 | Clean separation, well-structured |
| **Overall** | **6.1/10** | Solid foundation, needs a design pass |

---

*Design review by GitHub Copilot · June 1, 2026 · Based on full component audit of `phase-3/wiring`*
