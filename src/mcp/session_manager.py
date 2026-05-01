import uuid
from dataclasses import dataclass, field


@dataclass
class Session:
    connection_id: str
    device_id: str       # IP-derived: device_10_0_0_117
    user_id: str         # Mem0 user key: xiaozhi_device_10_0_0_117
    session_id: str      # From ESP32 hello
    proto_version: int = 1
    listen_mode: str = "auto"


class SessionManager:
    def __init__(self, user_id_prefix: str = "xiaozhi_"):
        self._sessions: dict[str, Session] = {}
        self._prefix = user_id_prefix

    def create(self, addr: tuple, session_id: str, proto_version: int) -> Session:
        connection_id = str(uuid.uuid4())[:8]
        device_id = f"device_{addr[0].replace('.', '_')}"
        user_id = f"{self._prefix}{device_id}"
        session = Session(
            connection_id=connection_id,
            device_id=device_id,
            user_id=user_id,
            session_id=session_id,
            proto_version=proto_version,
        )
        self._sessions[connection_id] = session
        return session

    def remove(self, connection_id: str) -> None:
        self._sessions.pop(connection_id, None)

    @staticmethod
    def new_request_id() -> str:
        return str(uuid.uuid4())[:8]
