/**
 * Portfolio Intelligence Cards
 * Shows active positions with AI guidance
 */

import React from "react"
import { PositionView } from "@/lib/types"

interface Props {
  positions: PositionView[] | null
}

export function PortfolioCards({ positions }: Props) {
  if (!positions) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded p-6">
        <h2 className="text-lg font-bold text-white mb-4">Portfolio</h2>
        <div className="text-gray-400">Loading positions...</div>
      </div>
    )
  }

  if (positions.length === 0) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded p-6">
        <h2 className="text-lg font-bold text-white mb-4">Portfolio</h2>
        <div className="text-gray-400 text-sm">No active positions</div>
      </div>
    )
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "LEADER":
        return "bg-green-900/30 border-green-700 text-green-300"
      case "WEAK":
        return "bg-orange-900/30 border-orange-700 text-orange-300"
      case "STALE":
        return "bg-gray-700/30 border-gray-600 text-gray-300"
      case "DUST":
        return "bg-red-900/30 border-red-700 text-red-300"
      case "RECOVERING":
        return "bg-blue-900/30 border-blue-700 text-blue-300"
      default:
        return "bg-gray-700/30 border-gray-600 text-gray-300"
    }
  }

  const getActionColor = (action?: string) => {
    switch (action) {
      case "HOLD":
        return "text-gray-400"
      case "TAKE_PROFIT":
        return "text-green-400"
      case "ROTATE_OUT":
        return "text-yellow-400"
      case "CLEAN_DUST":
        return "text-red-400"
      case "WAIT":
        return "text-blue-400"
      default:
        return "text-gray-400"
    }
  }

  return (
    <div className="bg-gray-800 border border-gray-700 rounded p-6">
      <h2 className="text-lg font-bold text-white mb-6">Portfolio Positions</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {positions.map((pos) => (
          <div
            key={pos.symbol}
            className={`border rounded p-4 ${getStatusColor(pos.status)}`}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
              <div className="text-lg font-mono font-bold">{pos.symbol}</div>
              <div className={`text-xs font-bold uppercase px-2 py-1 rounded bg-black/40`}>
                {pos.status}
              </div>
            </div>

            {/* Quantity */}
            <div className="grid grid-cols-2 gap-4 mb-4 pb-4 border-b border-current/20">
              <div>
                <div className="text-xs text-current/70 uppercase">Quantity</div>
                <div className="text-lg font-mono font-bold">{pos.quantity.toFixed(4)}</div>
              </div>
              <div>
                <div className="text-xs text-current/70 uppercase">Current Price</div>
                <div className="text-lg font-mono font-bold">
                  {pos.current_price ? `$${pos.current_price.toFixed(2)}` : "—"}
                </div>
              </div>
            </div>

            {/* Entry & PnL */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <div className="text-xs text-current/70 uppercase">Entry</div>
                <div className="text-sm font-mono">
                  {pos.entry_price ? `$${pos.entry_price.toFixed(2)}` : "—"}
                </div>
              </div>
              <div>
                <div className="text-xs text-current/70 uppercase">Unrealized PnL</div>
                <div
                  className={`text-sm font-mono font-bold ${
                    (pos.unrealized_pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"
                  }`}
                >
                  {pos.unrealized_pnl ? `$${pos.unrealized_pnl.toFixed(2)}` : "—"}
                </div>
              </div>
            </div>

            {/* PnL % */}
            {pos.unrealized_pnl_pct && (
              <div className="mb-4 pb-4 border-b border-current/20">
                <div className="text-xs text-current/70 uppercase">PnL %</div>
                <div
                  className={`text-lg font-mono font-bold ${
                    pos.unrealized_pnl_pct >= 0 ? "text-green-400" : "text-red-400"
                  }`}
                >
                  {pos.unrealized_pnl_pct >= 0 ? "+" : ""}
                  {pos.unrealized_pnl_pct.toFixed(2)}%
                </div>
              </div>
            )}

            {/* AI Action */}
            {pos.ai_action && (
              <div>
                <div className="text-xs text-current/70 uppercase mb-1">AI Action</div>
                <div className={`text-sm font-mono font-bold ${getActionColor(pos.ai_action)}`}>
                  → {pos.ai_action}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
