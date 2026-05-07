/**
 * Autonomous Activity Timeline
 * Live event feed showing recent trading activity
 */

import React from "react"
import { ActivityEvent } from "@/lib/types"
import { formatDistanceToNow } from "date-fns"

interface Props {
  events: ActivityEvent[] | null
}

export function ActivityTimeline({ events }: Props) {
  if (!events) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded p-6">
        <h2 className="text-lg font-bold text-white mb-4">Activity</h2>
        <div className="text-gray-400">Loading events...</div>
      </div>
    )
  }

  const getEventIcon = (eventType: string) => {
    switch (eventType) {
      case "DECISION":
        return "🧠"
      case "EXECUTION":
        return "⚡"
      case "FILL":
        return "✓"
      case "THROTTLE":
        return "🔒"
      case "RECOVERY":
        return "🔄"
      case "HEALTH":
        return "💚"
      case "CONTROL":
        return "🎛️"
      default:
        return "📌"
    }
  }

  const getEventColor = (eventType: string) => {
    switch (eventType) {
      case "DECISION":
        return "text-purple-400"
      case "EXECUTION":
        return "text-green-400"
      case "FILL":
        return "text-green-400"
      case "THROTTLE":
        return "text-red-400"
      case "RECOVERY":
        return "text-blue-400"
      case "HEALTH":
        return "text-cyan-400"
      case "CONTROL":
        return "text-yellow-400"
      default:
        return "text-gray-400"
    }
  }

  const recentEvents = events.slice(-20).reverse() // Last 20, newest first

  return (
    <div className="bg-gray-800 border border-gray-700 rounded p-6">
      <h2 className="text-lg font-bold text-white mb-6">Activity Timeline</h2>

      {recentEvents.length === 0 ? (
        <div className="text-gray-400 text-sm">No recent activity</div>
      ) : (
        <div className="space-y-4 max-h-96 overflow-y-auto">
          {recentEvents.map((event, idx) => (
            <div key={idx} className="flex gap-4 pb-4 border-b border-gray-700 last:border-b-0">
              {/* Icon & Time */}
              <div className="flex-shrink-0 w-12">
                <div className="text-2xl">{getEventIcon(event.event_type)}</div>
                <div className="text-xs text-gray-500 mt-1">
                  {formatDistanceToNow(event.timestamp * 1000, { addSuffix: true })}
                </div>
              </div>

              {/* Event Details */}
              <div className="flex-grow">
                {/* Action */}
                {event.action && (
                  <div className="text-sm">
                    <span className="font-mono font-bold text-white">{event.action}</span>
                    {event.symbol && (
                      <span className="text-cyan-400 font-mono ml-2">{event.symbol}</span>
                    )}
                  </div>
                )}

                {/* Details */}
                {event.details && (
                  <div className="text-sm text-gray-300 mt-1">{event.details}</div>
                )}

                {/* PnL */}
                {event.pnl !== undefined && (
                  <div
                    className={`text-sm font-mono mt-1 ${
                      event.pnl >= 0 ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    PnL: {event.pnl >= 0 ? "+" : ""}${event.pnl.toFixed(2)}
                  </div>
                )}

                {/* Event Type Badge */}
                <div className="mt-2">
                  <span className={`text-xs font-bold uppercase ${getEventColor(event.event_type)}`}>
                    {event.event_type}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
