/**
 * AI Governance Panel
 * Safe control endpoints for the operator
 */

import React, { useState } from "react"
import { apiClient } from "@/lib/api"

interface Props {
  onActionSuccess?: (message: string) => void
  onActionError?: (message: string) => void
}

export function GovernancePanel({ onActionSuccess, onActionError }: Props) {
  const [loading, setLoading] = useState<string | null>(null)
  const [confirmAction, setConfirmAction] = useState<string | null>(null)

  const handleAction = async (action: string, fn: () => Promise<any>) => {
    setLoading(action)
    try {
      const result = await fn()
      onActionSuccess?.(result.reason || `${action} completed successfully`)
      setConfirmAction(null)
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error"
      onActionError?.(message)
    } finally {
      setLoading(null)
    }
  }

  const buttonBase =
    "px-4 py-2 rounded font-mono text-sm font-bold transition disabled:opacity-50 disabled:cursor-not-allowed"

  const Button = ({
    label,
    action,
    fn,
    color,
    requiresConfirm = false,
  }: {
    label: string
    action: string
    fn: () => Promise<any>
    color: string
    requiresConfirm?: boolean
  }) => (
    <button
      className={`${buttonBase} ${color}`}
      disabled={loading !== null}
      onClick={() => {
        if (requiresConfirm) {
          setConfirmAction(action)
        } else {
          handleAction(action, fn)
        }
      }}
    >
      {loading === action ? "Processing..." : label}
    </button>
  )

  return (
    <div className="bg-gray-800 border border-gray-700 rounded p-6">
      <h2 className="text-lg font-bold text-white mb-6">Governance Controls</h2>

      {/* Confirmation Modal */}
      {confirmAction && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 border-2 border-red-700 rounded p-6 max-w-md">
            <h3 className="text-lg font-bold text-red-400 mb-4">Confirm Action</h3>
            <p className="text-gray-300 mb-6">
              {confirmAction === "PAUSE_ALL"
                ? "This will HALT ALL TRADING immediately. Are you sure?"
                : confirmAction === "PAUSE_BUYING"
                ? "Pause new BUY decisions (system continues monitoring)?"
                : confirmAction === "SAFE_MODE"
                ? "Activate SAFE MODE (reduced position sizes)?"
                : confirmAction === "CANCEL_ORDERS"
                ? "Cancel all open orders?"
                : "Are you sure?"}
            </p>
            <div className="flex gap-4">
              <button
                className="flex-1 px-4 py-2 bg-gray-700 text-gray-300 rounded font-mono text-sm font-bold hover:bg-gray-600"
                onClick={() => setConfirmAction(null)}
              >
                Cancel
              </button>
              <button
                className="flex-1 px-4 py-2 bg-red-700 text-red-200 rounded font-mono text-sm font-bold hover:bg-red-600"
                onClick={() => {
                  const fn =
                    confirmAction === "PAUSE_ALL"
                      ? () => apiClient.pauseAllTrading()
                      : confirmAction === "PAUSE_BUYING"
                      ? () => apiClient.pauseBuying()
                      : confirmAction === "SAFE_MODE"
                      ? () => apiClient.forceSafeMode()
                      : confirmAction === "CANCEL_ORDERS"
                      ? () => apiClient.cancelOpenOrders()
                      : () => Promise.reject("Unknown action")

                  handleAction(confirmAction, fn)
                }}
              >
                Confirm
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Trading Controls */}
      <div className="mb-8">
        <div className="text-sm text-gray-400 uppercase mb-4 font-bold">Trading Controls</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Button
            label="Pause Buying"
            action="PAUSE_BUYING"
            fn={() => apiClient.pauseBuying()}
            color="bg-yellow-700 hover:bg-yellow-600 text-yellow-200"
            requiresConfirm={true}
          />
          <Button
            label="Resume Buying"
            action="RESUME_BUYING"
            fn={() => apiClient.resumeBuying()}
            color="bg-green-700 hover:bg-green-600 text-green-200"
          />
          <Button
            label="Safe Mode"
            action="SAFE_MODE"
            fn={() => apiClient.forceSafeMode()}
            color="bg-orange-700 hover:bg-orange-600 text-orange-200"
            requiresConfirm={true}
          />
          <Button
            label="Resume Normal"
            action="RESUME_NORMAL"
            fn={() => apiClient.resumeNormal()}
            color="bg-blue-700 hover:bg-blue-600 text-blue-200"
          />
        </div>
      </div>

      {/* Emergency Controls */}
      <div className="mb-8 pb-8 border-b border-gray-700">
        <div className="text-sm text-gray-400 uppercase mb-4 font-bold">Emergency</div>
        <div className="grid grid-cols-2 gap-3">
          <Button
            label="Cancel Orders"
            action="CANCEL_ORDERS"
            fn={() => apiClient.cancelOpenOrders()}
            color="bg-red-900 hover:bg-red-800 text-red-300"
            requiresConfirm={true}
          />
          <Button
            label="⚠️ HALT ALL"
            action="PAUSE_ALL"
            fn={() => apiClient.pauseAllTrading()}
            color="bg-red-700 hover:bg-red-600 text-red-100 text-lg font-bold"
            requiresConfirm={true}
          />
        </div>
      </div>

      {/* Recovery Controls */}
      <div>
        <div className="text-sm text-gray-400 uppercase mb-4 font-bold">Recovery</div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          <Button
            label="Resume Trading"
            action="RESUME_TRADING"
            fn={() => apiClient.resumeTrading()}
            color="bg-green-800 hover:bg-green-700 text-green-300"
          />
        </div>
      </div>

      {/* Info */}
      <div className="mt-8 pt-6 border-t border-gray-700 text-xs text-gray-500">
        <p>
          💡 Controls are safety-bounded. All actions call backend governance endpoints only.
          Direct exchange access is disabled.
        </p>
      </div>
    </div>
  )
}
