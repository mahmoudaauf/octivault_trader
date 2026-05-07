/**
 * TypeScript data contracts matching backend API
 */

export type SystemHealthStatus = "HEALTHY" | "DEGRADED" | "CRITICAL" | "INITIALIZING"
export type ThrottleStatus = "CLEAR" | "PENDING" | "ACTIVE"
export type MarketRegime = "TRENDING" | "RANGING" | "CHOPPY" | "UNKNOWN"
export type CapitalState = "HEALTHY" | "CAUTION" | "WARNING" | "CRITICAL"
export type PositionStatus = "LEADER" | "WEAK" | "STALE" | "DUST" | "RECOVERING"
export type EventType = "DECISION" | "EXECUTION" | "FILL" | "THROTTLE" | "RECOVERY" | "HEALTH" | "CONTROL"
export type ActionType = "BUY" | "SELL" | "NONE" | "HOLD" | "TAKE_PROFIT" | "ROTATE_OUT" | "CLEAN_DUST" | "WAIT"

export interface SystemStatus {
  nav_usdt: number
  free_usdt: number
  locked_usdt: number
  growth_24h_pct: number
  active_positions_count: number
  open_orders_count: number
  mode: string
  market_regime: MarketRegime
  system_health: SystemHealthStatus
  capital_state: CapitalState
  throttle_status: ThrottleStatus
  throttle_until_ts?: number
  api_weight_estimate: number
  timestamp: number
}

export interface SignalView {
  source: string
  symbol: string
  direction: ActionType
  confidence: number
  reason?: string
}

export interface GateResult {
  gate_name: string
  passed: boolean
  reason?: string
}

export interface DecisionExplanation {
  symbol?: string
  action?: ActionType
  signals: SignalView[]
  gates: GateResult[]
  playbook?: string
  confidence?: number
  blocked_reason?: string
  timestamp: number
}

export interface PositionView {
  symbol: string
  quantity: number
  entry_price?: number
  current_price?: number
  unrealized_pnl?: number
  unrealized_pnl_pct?: number
  status: PositionStatus
  ai_action?: ActionType
}

export interface CapitalHealth {
  free_ratio: number
  active_ratio: number
  reserve_ratio: number
  dust_ratio: number
  exposure_ratio: number
  largest_position_pct: number
  state: CapitalState
  warnings: string[]
}

export interface Portfolio {
  positions: PositionView[]
  health: CapitalHealth
}

export interface ActivityEvent {
  timestamp: number
  event_type: EventType
  symbol?: string
  action?: ActionType
  details?: string
  pnl?: number
}

export interface ActivityResponse {
  events: ActivityEvent[]
  total: number
}

export interface ComponentHealth {
  component: string
  status: SystemHealthStatus
  error_count: number
  last_error?: string
  last_check_ts: number
}

export interface HealthResponse {
  overall: SystemHealthStatus
  components: ComponentHealth[]
  timestamp: number
}

export interface ControlActionResponse {
  success: boolean
  action: string
  reason?: string
  timestamp: number
}
