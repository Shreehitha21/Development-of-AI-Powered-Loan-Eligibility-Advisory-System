from __future__ import annotations

import json
from typing import Dict, List

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, user_id: int, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.setdefault(user_id, []).append(websocket)

    def disconnect(self, user_id: int, websocket: WebSocket) -> None:
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

    async def send_to_user(self, user_id: int, message: dict) -> None:
        data = json.dumps(message)
        for ws in self.active_connections.get(user_id, []):
            await ws.send_text(data)

    async def broadcast(self, message: dict) -> None:
        data = json.dumps(message)
        for user_conns in self.active_connections.values():
            for ws in user_conns:
                await ws.send_text(data)
