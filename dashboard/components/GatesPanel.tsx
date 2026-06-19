import React, { useRef } from "react"
import { formatDistanceToNow } from "date-fns"
import { DecisionExplanation, GateResult } from "@/lib/types"

interface Props { decision: DecisionExplanation | null }

interface GateHistory {
  passed: boolean
  reason?: string
  failCount: number
  flipCount: number
  lastChanged: Date
  lastSeen: Date
}

export function GatesPanel({ decision }: Props) {
  const histRef = useRef<Record<string, GateHistory>>({})

  const gates: GateResult[] = decision?.gates ?? []

  // Update history
  gates.forEach(g => {
    const prev = histRef.current[g.gate_name]
    const now  = new Date()
    if (!prev) {
      histRef.current[g.gate_name] = {
        passed:      g.passed,
        reason:      g.reason,
        failCount:   g.passed ? 0 : 1,
        flipCount:   0,
        lastChanged: now,
        lastSeen:    now,
      }
    } else {
      const flipped = prev.passed !== g.passed
      histRef.current[g.gate_name] = {
        passed:      g.passed,
        reason:      g.reason,
        failCount:   g.passed ? prev.failCount : prev.failCount + 1,
        flipCount:   flipped ? prev.flipCount + 1 : prev.flipCount,
        lastChanged: flipped ? now : prev.lastChanged,
        lastSeen:    now,
      }
    }
  })

  const hist = histRef.current
  const names = Object.keys(hist).sort((a, b) => {
    const aPass = hist[a].passed ? 1 : 0
    const bPass = hist[b].passed ? 1 : 0
    return aPass - bPass || a.localeCompare(b)
  })

  const failing = names.filter(n => !hist[n].passed)
  const passing = names.filter(n =>  hist[n].passed)
  const total   = names.length
  const passCount = passing.length
  const passRate  = total > 0 ? passCount / total : 1

  // SVG ring
  const R = 18, C = 2 * Math.PI * R
  const filled = C * passRate
  const ringColor = passRate === 1 ? "#10b981" : passRate >= 0.5 ? "#f59e0b" : "#ef4444"

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Gate Monitor</span>
        {total > 0 && (
          <div className="flex items-center gap-3">
            {/* Pass rate ring */}
            <svg width="40" height="40" viewBox="0 0 40 40">
              <circle cx="20" cy="20" r={R} fill="none" stroke="#1a2336" strokeWidth="3.5" />
              <circle cx="20" cy="20" r={R} fill="none"
                stroke={ringColor} strokeWidth="3.5"
                strokeDasharray={`${filled} ${C}`}
                strokeLinecap="round"
                transform="rotate(-90 20 20)"
                style={{ transition: "stroke-dasharray 0.5s ease" }}
              />
              <text x="20" y="24" textAnchor="middle"
                fontSize="9" fontWeight="700" fill={ringColor} fontFamily="monospace">
                {Math.round(passRate * 100)}%
              </text>
            </svg>
            <div className="text-right">
              <div className={`text-xs font-mono font-bold ${failing.length > 0 ? "text-red-400 live-pulse" : "text-emerald-400"}`}>
                {failing.length > 0 ? `${failing.length} BLOCKING` : "ALL CLEAR"}
              </div>
              <div className="text-[10px] text-gray-600">{passCount}/{total} passed</div>
            </div>
          </div>
        )}
      </div>

      <div className="card-body flex flex-col gap-2">
        {names.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 py-6">
            <div className="w-10 h-10 rounded-full border border-dashed border-gray-800 flex items-center justify-center text-xl opacity-30">
              🔒
            </div>
            <p className="text-xs text-gray-600 text-center">Awaiting next decision cycle</p>
          </div>
        ) : (
          <div className="flex flex-col gap-1.5 overflow-y-auto flex-1 min-h-0">
            {names.map(name => {
              const g = hist[name]
              return (
                <div key={name} className={g.passed ? "gate-row gate-row-pass" : "gate-row gate-row-fail"}>
                  {/* Icon */}
                  <span className={`text-base mt-0.5 ${g.passed ? "text-emerald-400" : "text-red-400"}`}>
                    {g.passed ? "✓" : "✗"}
                  </span>

                  {/* Body */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={`text-xs font-semibold ${g.passed ? "text-emerald-300" : "text-red-300"}`}>
                        {name.replace(/_/g, " ")}
                      </span>
                      {g.flipCount > 1 && (
                        <span className="badge badge-yellow">⚡ {g.flipCount}x</span>
                      )}
                      {!g.passed && g.failCount > 1 && (
                        <span className="badge badge-red">{g.failCount} fails</span>
                      )}
                    </div>
                    {!g.passed && g.reason && (
                      <p className="text-[10px] text-gray-500 mt-0.5 truncate">{g.reason}</p>
                    )}
                  </div>

                  {/* Time */}
                  <span className="text-[9px] text-gray-600 font-mono shrink-0">
                    {formatDistanceToNow(g.lastChanged, { addSuffix: true })}
                  </span>
                </div>
              )
            })}
          </div>
        )}

        {decision?.playbook && names.length > 0 && (
          <div className="pt-3 border-t border-gray-800/60 shrink-0">
            <p className="text-[10px] text-gray-500 leading-relaxed">
              <span className="text-gray-600 font-semibold">Playbook: </span>
              {decision.playbook}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
