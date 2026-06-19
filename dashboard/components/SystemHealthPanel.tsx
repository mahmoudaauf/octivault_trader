import React from "react"
import { formatDistanceToNow } from "date-fns"
import { HealthResponse, ComponentHealth, SystemHealthStatus } from "@/lib/types"

interface Props { health: HealthResponse | null }

const STATUS_BADGE: Record<SystemHealthStatus, string> = {
  HEALTHY:      "badge badge-green",
  DEGRADED:     "badge badge-yellow",
  CRITICAL:     "badge badge-red",
  INITIALIZING: "badge badge-blue",
}

const STATUS_DOT: Record<SystemHealthStatus, string> = {
  HEALTHY:      "bg-emerald-400",
  DEGRADED:     "bg-amber-400 live-pulse",
  CRITICAL:     "bg-red-500 live-pulse",
  INITIALIZING: "bg-blue-400",
}

function ComponentRow({ c }: { c: ComponentHealth }) {
  return (
    <div className={`flex items-start gap-3 rounded-lg px-3 py-2.5 border transition-colors ${
      c.status === "HEALTHY"
        ? "bg-emerald-950/10 border-emerald-900/20"
        : c.status === "DEGRADED"
        ? "bg-amber-950/15 border-amber-900/30"
        : "bg-red-950/15 border-red-900/30"
    }`}>
      <span className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${STATUS_DOT[c.status]}`} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-semibold text-slate-300">{c.component}</span>
          <span className={STATUS_BADGE[c.status]}>{c.status}</span>
        </div>
        {c.last_error && (
          <p className="text-[10px] text-gray-500 mt-0.5 truncate">{c.last_error}</p>
        )}
        <div className="flex items-center gap-3 mt-1">
          {c.error_count > 0 && (
            <span className="text-[9px] font-mono text-red-500">{c.error_count} errors</span>
          )}
          <span className="text-[9px] font-mono text-gray-600">
            checked {formatDistanceToNow(new Date(c.last_check_ts * 1000), { addSuffix: true })}
          </span>
        </div>
      </div>
    </div>
  )
}

export function SystemHealthPanel({ health }: Props) {
  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-3">
          <span className="card-title">System Health</span>
          {health && (
            <span className={STATUS_BADGE[health.overall]}>{health.overall}</span>
          )}
        </div>
        {health && (
          <span className="text-[10px] font-mono text-gray-600">
            {health.components.filter(c => c.status === "HEALTHY").length}/{health.components.length} healthy
          </span>
        )}
      </div>

      <div className="card-body flex flex-col gap-2">
        {!health ? (
          [...Array(4)].map((_, i) => <div key={i} className="skeleton h-14 w-full" />)
        ) : health.components.length === 0 ? (
          <div className="flex items-center justify-center py-8">
            <p className="text-xs text-gray-600">No health data available</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {health.components.map(c => <ComponentRow key={c.component} c={c} />)}
          </div>
        )}
      </div>
    </div>
  )
}
