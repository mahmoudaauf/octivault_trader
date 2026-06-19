export type SystemHealth = 'HEALTHY' | 'DEGRADED' | 'CRITICAL';
export type SystemMode = 'BOOTING' | 'HYDRATING' | 'VALIDATING' | 'READY';

export interface SystemStatus {
  nav: number;
  growth_pct: number;
  free_capital: number;
  active_positions: number;
  mode: SystemMode;
  health: SystemHealth;
  is_throttled: boolean;
  uptime: string;
}

export interface AIDecision {
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  rationale: string;
  signals: {
    momentum: number;
    volatility: number;
    trend: 'BULL' | 'BEAR' | 'NEUTRAL';
  };
  gates: {
    drawdown_ok: boolean;
    liquidity_ok: boolean;
    governance_ok: boolean;
  };
  timestamp: string;
}

export interface Position {
  symbol: string;
  entry_price: number;
  current_price: number;
  qty: number;
  pnl_pct: number;
  value_usdt: number;
  ai_guidance: string;
}

export interface ActivityEvent {
  id: string;
  type: 'ORDER' | 'SIGNAL' | 'SYSTEM' | 'THROTTLE';
  message: string;
  timestamp: string;
}