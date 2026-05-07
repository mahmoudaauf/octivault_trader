/**
 * Top system state bar - quick status glance
 */

import React from "react"
import { SystemStatus } from "@/lib/types"

interface Props {
  status: SystemStatus | null
}

export function SystemStateBar({ status }: Props) {
  if (!status) {
    return (
      <div className="bg-gray-900 border-b border-gray-700 p-6">
        <div className="text-gray-400">Loading system state...</div>
      </div>
    )
  }

  const healthColor =
    status.system_health === "HEALTHY"
      ? "text-green-400"
      : status.system_health === "DEGRADED"
      ? "text-yellow-400"
      : "text-red-500"

  const capitalColor =
    status.capital_state === "HEALTHY"
      ? "text-green-400"
      : status.capital_state === "CAUTION"
      ? "text-yellow-400"
      : status.capital_state === "WARNING"
      ? "text-orange-400"
      : "text-red-500"

  const throttleColor =
    status.throttle_status === "CLEAR"
      ? "text-green-400"
      : status.throttle_status === "PENDING"
      ? "text-yellow-400"
      : "text-red-500"

  return (
    <div className="bg-gray-900 border-b border-gray-700 p-6">
      <div className="grid grid-cols-4 gap-8 md:grid-cols-8">
        {/* NAV */}
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wider">NAV</div>
          <div className="text-2xl font-mono font-bold text-white">
            ${status.nav_usdt.toFixed(2)}
          </div>
        </div>

        {/* 24h Growth */}
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wider">24h Growth</div>
          <div className={`text-2xl font-mono font-bold ${status.growth_24h_pct >= 0 ? "text-green-400" : "text-red-500"}`}>
            {status.growth_24h_pct >= 0 ? "+" : ""}{status.growth_24h_pct.toFixed(2)}%
          </div>
        </div>

        {/* Free Capital */}
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wider">Free</div>
          <div className="text-2xl font-mono font-bold text-white">
            ${status.free_usdt.toFixed(2)}
          </div>
        </div>

        {/* Active Positions */}
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wider">Positions</div>
          <div className="text-2xl font-mono font-bold text-blue-400">
            {status.active_positions_count}
          </div>
        </div>

        {/* Mode */}
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wider">Mode</div>
          <div className="text-sm font-mono text-purple-400">{status.mode}</div>
        </div>

        {/* Market Regime */}
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wider">Regime</div>
          <div className="text-sm font-mono text-cyan-400">{status.market_regime}</div>
        </div>

        {/* System Health */}
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wider">Health</div>
          <div className={`text-sm font-mono font-bold ${healthColor}`}>{status.system_health}</div>
        </div>

        {/* Throttle Status */}
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wider">Throttle</div>
          <div className={`text-sm font-mono font-bold ${throttleColor}`}>{status.throttle_status}</div>
        </div>
      </div>
    </div>
  )
}
