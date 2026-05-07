/**
 * API client for AI Command Center
 * Handles all communication with the backend
 */

import axios, { AxiosInstance } from "axios"
import {
  SystemStatus,
  DecisionExplanation,
  Portfolio,
  ActivityResponse,
  HealthResponse,
  ControlActionResponse,
} from "./types"

export class CommandCenterAPIClient {
  private client: AxiosInstance
  private baseURL: string

  constructor(baseURL: string = "http://localhost:8000") {
    this.baseURL = baseURL
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 10000,
      headers: {
        "Content-Type": "application/json",
      },
    })
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Read-only endpoints
  // ──────────────────────────────────────────────────────────────────────────

  async getStatus(): Promise<SystemStatus> {
    const response = await this.client.get<SystemStatus>("/api/status")
    return response.data
  }

  async getAIState(): Promise<DecisionExplanation> {
    const response = await this.client.get<DecisionExplanation>("/api/ai-state")
    return response.data
  }

  async getPortfolio(): Promise<Portfolio> {
    const response = await this.client.get<Portfolio>("/api/portfolio")
    return response.data
  }

  async getActivity(limit: number = 50): Promise<ActivityResponse> {
    const response = await this.client.get<ActivityResponse>("/api/activity", {
      params: { limit },
    })
    return response.data
  }

  async getHealth(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>("/api/health")
    return response.data
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Control endpoints (require confirmation)
  // ──────────────────────────────────────────────────────────────────────────

  async pauseBuying(): Promise<ControlActionResponse> {
    const response = await this.client.post<ControlActionResponse>(
      "/api/control/pause-buying",
      {},
      { params: { confirmed: true } }
    )
    return response.data
  }

  async resumeBuying(): Promise<ControlActionResponse> {
    const response = await this.client.post<ControlActionResponse>(
      "/api/control/resume-buying"
    )
    return response.data
  }

  async forceSafeMode(): Promise<ControlActionResponse> {
    const response = await this.client.post<ControlActionResponse>(
      "/api/control/force-safe-mode",
      {},
      { params: { confirmed: true } }
    )
    return response.data
  }

  async resumeNormal(): Promise<ControlActionResponse> {
    const response = await this.client.post<ControlActionResponse>(
      "/api/control/resume-normal"
    )
    return response.data
  }

  async cancelOpenOrders(): Promise<ControlActionResponse> {
    const response = await this.client.post<ControlActionResponse>(
      "/api/control/cancel-open-orders",
      {},
      { params: { confirmed: true } }
    )
    return response.data
  }

  async pauseAllTrading(): Promise<ControlActionResponse> {
    const response = await this.client.post<ControlActionResponse>(
      "/api/control/pause-all",
      {},
      { params: { confirmed: true } }
    )
    return response.data
  }

  async resumeTrading(): Promise<ControlActionResponse> {
    const response = await this.client.post<ControlActionResponse>(
      "/api/control/resume-trading"
    )
    return response.data
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Polling loop for real-time updates
  // ──────────────────────────────────────────────────────────────────────────

  startPolling(
    onUpdate: {
      status?: (s: SystemStatus) => void
      aiState?: (s: DecisionExplanation) => void
      portfolio?: (p: Portfolio) => void
      activity?: (a: ActivityResponse) => void
      health?: (h: HealthResponse) => void
      error?: (e: Error) => void
    },
    intervalMs: number = 2000
  ): () => void {
    const poll = async () => {
      try {
        const [status, aiState, portfolio, activity, health] = await Promise.all([
          this.getStatus(),
          this.getAIState(),
          this.getPortfolio(),
          this.getActivity(100),
          this.getHealth(),
        ])

        onUpdate.status?.(status)
        onUpdate.aiState?.(aiState)
        onUpdate.portfolio?.(portfolio)
        onUpdate.activity?.(activity)
        onUpdate.health?.(health)
      } catch (error) {
        onUpdate.error?.(error as Error)
      }
    }

    const intervalId = setInterval(poll, intervalMs)
    poll() // Call immediately

    return () => clearInterval(intervalId) // Return cleanup function
  }
}

// Export singleton
export const apiClient = new CommandCenterAPIClient()
