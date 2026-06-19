import React, { ReactNode } from "react"

interface TooltipProps {
  text: string
  children: ReactNode
}

export function Tooltip({ text, children }: TooltipProps) {
  return (
    <span className="relative group">
      {children}
      <span className="pointer-events-none absolute z-10 left-1/2 -translate-x-1/2 mt-2 w-max max-w-xs px-3 py-1 rounded bg-gray-900 text-xs text-gray-100 opacity-0 group-hover:opacity-100 group-focus:opacity-100 transition-opacity duration-200 shadow-lg border border-gray-700 whitespace-pre-line">
        {text}
      </span>
    </span>
  )
}
