/**
 * AI Brain / Decision Explanation Panel
 * Shows why the AI made its last decision
 */

import React from "react"
import { DecisionExplanation } from "@/lib/types"

interface Props {
  decision: DecisionExplanation | null
}

export function AIBrainPanel({ decision }: Props) {
  if (!decision) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded p-6">
        <h2 className="text-lg font-bold text-white mb-4">AI Brain</h2>
        <div className="text-gray-400">No decision data available</div>
      </div>
    )
  }

  const actionColor =
    decision.action === "BUY"
      ? "bg-green-900 text-green-300"
      : decision.action === "SELL"
      ? "bg-red-900 text-red-300"
      : "bg-gray-700 text-gray-300"

  return (
    <div className="bg-gray-800 border border-gray-700 rounded p-6">
      <h2 className="text-lg font-bold text-white mb-6">AI Brain</h2>

      {/* Symbol & Action */}
      <div className="mb-6">
        <div className="flex items-center gap-4">
          <div className="text-sm text-gray-500">Symbol:</div>
          <div className="text-2xl font-mono font-bold text-white">
            {decision.symbol || "—"}
          </div>

          <div className="ml-auto">
            <div className={`px-4 py-2 rounded font-mono font-bold ${actionColor}`}>
              {decision.action || "NO ACTION"}
            </div>
          </div>
        </div>
      </div>

      {/* Decision Blocked? */}
      {decision.blocked_reason && (
        <div className="mb-6 p-4 bg-red-900/30 border border-red-700 rounded">
          <div className="text-red-300 text-sm">
            <strong>Decision Blocked:</strong> {decision.blocked_reason}
          </div>
        </div>
      )}

      {/* Signals */}
      {decision.signals && decision.signals.length > 0 && (
        <div className="mb-6">
          <div className="text-sm font-bold text-gray-400 uppercase mb-3">Signals</div>
          <div className="space-y-2">
            {decision.signals.map((signal, idx) => (
              <div key={idx} className="flex justify-between items-center bg-gray-700/50 p-3 rounded">
                <div>
                  <div className="text-sm text-gray-300">
                    <span className="font-mono font-bold text-cyan-300">{signal.source}</span>
                    {" "}
                    <span className="text-purple-300">{signal.direction}</span>
                  </div>
                  {signal.reason && <div className="text-xs text-gray-500 mt-1">{signal.reason}</div>}
                </div>
                <div className="text-right">
                  <div className="text-lg font-mono font-bold text-white">
                    {(signal.confidence * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Gates */}
      {decision.gates && decision.gates.length > 0 && (
        <div className="mb-6">
          <div className="text-sm font-bold text-gray-400 uppercase mb-3">Gate Checks</div>
          <div className="space-y-2">
            {decision.gates.map((gate, idx) => (
              <div key={idx} className="flex justify-between items-center">
                <div className="text-sm text-gray-300">{gate.gate_name}</div>
                <div className={gate.passed ? "text-green-400 font-bold" : "text-red-400 font-bold"}>
                  {gate.passed ? "✓ PASS" : "✗ FAIL"}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Playbook & Confidence */}
      <div className="grid grid-cols-2 gap-4 mt-6 pt-4 border-t border-gray-700">
        {decision.playbook && (
          <div>
            <div className="text-xs text-gray-500 uppercase">Playbook</div>
            <div className="text-sm font-mono text-purple-300">{decision.playbook}</div>
          </div>
        )}
        {decision.confidence && (
          <div>
            <div className="text-xs text-gray-500 uppercase">Confidence</div>
            <div className="text-lg font-mono font-bold text-white">
              {(decision.confidence * 100).toFixed(0)}%
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
