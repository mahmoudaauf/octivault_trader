import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logger = logging.getLogger("DashboardServer")

class DashboardServer:
    """
    FastAPI server providing real-time visibility into SharedState via WebSockets and REST.
    Designed to run alongside the main Phase 9 event loop.
    """
    def __init__(self, shared_state: Any, host: str = "0.0.0.0", port: int = 8000):
        self.shared_state = shared_state
        self.host = host
        self.port = port
        self.app = FastAPI(title="Octivault P9 Dashboard")
        
        # Enable CORS for local development
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
        self.connections: List[WebSocket] = []
        self._running = False

    def _setup_routes(self):
        @self.app.get("/api/state")
        async def get_state():
            """Returns a full snapshot of the current system state."""
            nav = self.shared_state.get_nav_quote()
            balances = self.shared_state.balances
            positions = self.shared_state.get_positions_snapshot()
            scores = self.shared_state.agent_scores
            readiness = self.shared_state.get_readiness_snapshot()
            
            return {
                "nav": nav,
                "balances": balances,
                "positions": positions,
                "scores": scores,
                "readiness": readiness,
                "timestamp": time.time()
            }

        @self.app.get("/api/config")
        async def get_config():
            """Returns the current dynamic configuration."""
            return getattr(self.shared_state, "dynamic_config", {})

        @self.app.post("/api/config")
        async def update_config(config: Dict[str, Any]):
            """Updates the dynamic configuration."""
            if hasattr(self.shared_state, "dynamic_config"):
                self.shared_state.dynamic_config.update(config)
                await self.shared_state.emit_event("ConfigUpdated", config)
                return {"status": "success"}
            return JSONResponse(status_code=400, content={"status": "error", "message": "Dynamic config not enabled"})

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.connections.append(websocket)
            logger.info(f"New dashboard connection. Total: {len(self.connections)}")
            try:
                # Send initial state
                state = await get_state()
                await websocket.send_json({"type": "initial_state", "data": state})
                
                # Keep connection open
                while True:
                    await websocket.receive_text() # Wait for client messages or close
            except WebSocketDisconnect:
                self.connections.remove(websocket)
                logger.info(f"Dashboard disconnected. Total: {len(self.connections)}")

    async def start(self):
        """Launches the server and the event broadcaster."""
        self._running = True
        
        # Start broadcaster task
        asyncio.create_task(self._broadcaster_loop())
        
        # Configure and start Uvicorn
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        
        logger.info(f"🚀 Dashboard Server starting at http://{self.host}:{self.port}")
        await server.serve()

    async def _broadcaster_loop(self):
        """Listens for SharedState events and broadcasts them to all connected clients."""
        logger.info("Broadcaster loop started.")
        # Subscribe to SharedState events
        queue = await self.shared_state.subscribe_events("dashboard_server")
        
        try:
            while self._running:
                event = await queue.get()
                if not self.connections:
                    continue
                
                # Broadcast to all active WebSockets
                payload = json.dumps({
                    "type": "event",
                    "event": event["name"],
                    "data": event["data"],
                    "timestamp": event["timestamp"]
                })
                
                disconnected = []
                for ws in self.connections:
                    try:
                        await ws.send_text(payload)
                    except Exception:
                        disconnected.append(ws)
                
                for ws in disconnected:
                    if ws in self.connections:
                        self.connections.remove(ws)
        finally:
            await self.shared_state.unsubscribe("dashboard_server")
            logger.info("Broadcaster loop stopped.")

    def stop(self):
        self._running = False
