/**
 * AI Trading Command Center - Main Dashboard Page
 */

import React, { useState, useEffect } from "react"
import Head from "next/head"
import {
  SystemStatus,
  DecisionExplanation,
  Portfolio,
  ActivityResponse,
  HealthResponse,
} from "@/lib/types"
import { apiClient } from "@/lib/api"
import { SystemStateBar } from "@/components/SystemStateBar"
import { AIBrainPanel } from "@/components/AIBrainPanel"
import { CapitalHealthPanel } from "@/components/CapitalHealthPanel"
import { PortfolioCards } from "@/components/PortfolioCards"
import { ActivityTimeline } from "@/components/ActivityTimeline"
import { GovernancePanel } from "@/components/GovernancePanel"

export default function Dashboard() {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [aiState, setAIState] = useState<DecisionExplanation | null>(null)
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null)
  const [activity, setActivity] = useState<ActivityResponse | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)

  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null)
  const [connected, setConnected] = useState(false)

  // Start polling on mount
  useEffect(() => {
    const stopPolling = apiClient.startPolling(
      {
        status: (s) => {
          setStatus(s)
          setConnected(true)
        },
        aiState: (s) => setAIState(s),
        portfolio: (p) => setPortfolio(p),
        activity: (a) => setActivity(a),
        health: (h) => setHealth(h),
        error: (e) => {
          console.error("API error:", e)
          setConnected(false)
        },
      },
      2000 // Poll every 2 seconds
    )

    return stopPolling
  }, [])

  const showMessage = (type: "success" | "error", text: string) => {
    setMessage({ type, text })
    setTimeout(() => setMessage(null), 5000)
  }

  return (
    <>
      <Head>
        <title>AI Trading Command Center</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="min-h-screen bg-gray-950 text-white">
        {/* Header */}
        <div className="bg-gray-900 border-b border-gray-700 px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">🚀 AI Trading Command Center</h1>
            <p className="text-sm text-gray-500">Autonomous trading system supervision</p>
          </div>

          {/* Connection Status */}
          <div className="flex items-center gap-3">
            <div
              className={`w-3 h-3 rounded-full ${connected ? "bg-green-500 animate-pulse" : "bg-red-500"}`}
            />
            <span className="text-sm font-mono text-gray-400">
              {connected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </div>

        {/* Messages */}
        {message && (
          <div
            className={`${message.type === "success" ? "bg-green-900" : "bg-red-900"} border-b ${message.type === "success" ? "border-green-700" : "border-red-700"} px-6 py-3 text-sm`}
          >
            {message.type === "success" ? "✓" : "✗"} {message.text}
          </div>
        )}

        {/* Main Content */}
        <div className="p-6 space-y-6">
          {/* System State Bar */}
          <SystemStateBar status={status} />

          {/* Row 1: AI Brain & Capital Health */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <AIBrainPanel decision={aiState} />
            <CapitalHealthPanel health={portfolio?.health ?? null} />
          </div>

          {/* Row 2: Portfolio & Timeline */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <PortfolioCards positions={portfolio?.positions ?? null} />
            </div>
            <ActivityTimeline events={activity?.events ?? null} />
          </div>

          {/* Row 3: Governance Controls */}
          <GovernancePanel
            onActionSuccess={(msg) => showMessage("success", msg)}
            onActionError={(msg) => showMessage("error", msg)}
          />

          {/* Health Status (Optional) */}
          {health && (
            <div className="bg-gray-800 border border-gray-700 rounded p-6">
              <h2 className="text-lg font-bold text-white mb-4">System Health</h2>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {health.components.map((comp) => (
                  <div
                    key={comp.component}
                    className={`p-4 rounded border ${
                      comp.status === "HEALTHY"
                        ? "bg-green-900/20 border-green-700"
                        : comp.status === "DEGRADED"
                        ? "bg-yellow-900/20 border-yellow-700"
                        : "bg-red-900/20 border-red-700"
                    }`}
                  >
                    <div className="font-mono font-bold text-sm">{comp.component}</div>
                    <div
                      className={`text-sm mt-1 ${
                        comp.status === "HEALTHY"
                          ? "text-green-400"
                          : comp.status === "DEGRADED"
                          ? "text-yellow-400"
                          : "text-red-400"
                      }`}
                    >
                      {comp.status}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-gray-700 px-6 py-4 text-xs text-gray-600 text-center mt-12">
          <p>Autonomous AI Trading System • Supervision & Governance Only</p>
          <p>All control actions route through backend governance endpoints</p>
        </div>
      </div>
    </>
  )
}
