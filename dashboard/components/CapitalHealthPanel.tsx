/**
 * Capital Health Panel
 * Shows capital allocation and health metrics
 */

import React from "react"
import { CapitalHealth } from "@/lib/types"

interface Props {
  health: CapitalHealth | null
}

export function CapitalHealthPanel({ health }: Props) {
  if (!health) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded p-6">
        <h2 className="text-lg font-bold text-white mb-4">Capital Health</h2>
        <div className="text-gray-400">Loading capital data...</div>
      </div>
    )
  }

  const stateColor =
    health.state === "HEALTHY"
      ? "text-green-400"
      : health.state === "CAUTION"
      ? "text-yellow-400"
      : health.state === "WARNING"
      ? "text-orange-400"
      : "text-red-500"

  const ProgressBar = ({ value, color }: { value: number; color: string }) => (
    <div className="w-full bg-gray-700 rounded h-2">
      <div
        className={`h-2 rounded ${color}`}
        style={{ width: `${Math.min(value * 100, 100)}%` }}
      />
    </div>
  )

  return (
    <div className="bg-gray-800 border border-gray-700 rounded p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-bold text-white">Capital Health</h2>
        <div className={`font-mono font-bold ${stateColor}`}>{health.state}</div>
      </div>

      {/* Allocation Rings */}
      <div className="grid grid-cols-3 gap-6 mb-8">
        {/* Free Capital */}
        <div className="text-center">
          <div className="relative w-20 h-20 mx-auto mb-3">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
              <circle
                cx="50"
                cy="50"
                r="45"
                fill="none"
                stroke="#374151"
                strokeWidth="8"
              />
              <circle
                cx="50"
                cy="50"
                r="45"
                fill="none"
                stroke="#10b981"
                strokeWidth="8"
                strokeDasharray={`${Math.max(0, health.free_ratio) * 282.7} 282.7`}
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="text-sm font-mono font-bold text-white">
                  {(health.free_ratio * 100).toFixed(0)}%
                </div>
              </div>
            </div>
          </div>
          <div className="text-xs text-gray-500 uppercase">Free</div>
        </div>

        {/* Active Capital */}
        <div className="text-center">
          <div className="relative w-20 h-20 mx-auto mb-3">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
              <circle
                cx="50"
                cy="50"
                r="45"
                fill="none"
                stroke="#374151"
                strokeWidth="8"
              />
              <circle
                cx="50"
                cy="50"
                r="45"
                fill="none"
                stroke="#3b82f6"
                strokeWidth="8"
                strokeDasharray={`${Math.max(0, health.active_ratio) * 282.7} 282.7`}
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="text-sm font-mono font-bold text-white">
                  {(health.active_ratio * 100).toFixed(0)}%
                </div>
              </div>
            </div>
          </div>
          <div className="text-xs text-gray-500 uppercase">Active</div>
        </div>

        {/* Reserve Capital */}
        <div className="text-center">
          <div className="relative w-20 h-20 mx-auto mb-3">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
              <circle
                cx="50"
                cy="50"
                r="45"
                fill="none"
                stroke="#374151"
                strokeWidth="8"
              />
              <circle
                cx="50"
                cy="50"
                r="45"
                fill="none"
                stroke="#8b5cf6"
                strokeWidth="8"
                strokeDasharray={`${Math.max(0, health.reserve_ratio) * 282.7} 282.7`}
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="text-sm font-mono font-bold text-white">
                  {(health.reserve_ratio * 100).toFixed(0)}%
                </div>
              </div>
            </div>
          </div>
          <div className="text-xs text-gray-500 uppercase">Reserve</div>
        </div>
      </div>

      {/* Risk Metrics */}
      <div className="space-y-4 mb-6 pb-6 border-b border-gray-700">
        <div>
          <div className="flex justify-between items-center mb-2">
            <div className="text-xs text-gray-500 uppercase">Exposure Ratio</div>
            <div className="text-sm font-mono font-bold text-white">
              {(health.exposure_ratio * 100).toFixed(1)}%
            </div>
          </div>
          <ProgressBar value={health.exposure_ratio} color="bg-blue-500" />
        </div>

        <div>
          <div className="flex justify-between items-center mb-2">
            <div className="text-xs text-gray-500 uppercase">Largest Position</div>
            <div className="text-sm font-mono font-bold text-white">
              {health.largest_position_pct.toFixed(1)}%
            </div>
          </div>
          <ProgressBar value={health.largest_position_pct / 100} color="bg-orange-500" />
        </div>

        <div>
          <div className="flex justify-between items-center mb-2">
            <div className="text-xs text-gray-500 uppercase">Dust Ratio</div>
            <div className="text-sm font-mono font-bold text-white">
              {(health.dust_ratio * 100).toFixed(1)}%
            </div>
          </div>
          <ProgressBar value={health.dust_ratio} color="bg-red-500" />
        </div>
      </div>

      {/* Warnings */}
      {health.warnings && health.warnings.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs text-gray-500 uppercase">Warnings</div>
          {health.warnings.map((warning, idx) => (
            <div key={idx} className="text-sm text-yellow-400 flex items-center gap-2">
              <span>⚠️</span>
              {warning}
            </div>
          ))}
        </div>
      )}

      {health.warnings?.length === 0 && (
        <div className="text-sm text-green-400">✓ No warnings</div>
      )}
    </div>
  )
}
