import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from agent_memory import ConversationMemory


SESSION_STORE_FILE = Path("logs/agent_sessions.json")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_sessions() -> Dict[str, Any]:
    if not SESSION_STORE_FILE.exists():
        return {}

    try:
        with SESSION_STORE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        return {}

    return data


def _save_sessions(sessions: Dict[str, Any]):
    SESSION_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with SESSION_STORE_FILE.open("w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


def generate_session_id() -> str:
    return f"agent-{uuid.uuid4().hex[:12]}"


def get_or_create_session(
    session_id: Optional[str] = None,
) -> Tuple[str, ConversationMemory, bool]:
    """Return session id, memory object, and whether it was newly created."""

    sessions = _load_sessions()
    normalized_session_id = (session_id or "").strip()

    if not normalized_session_id:
        normalized_session_id = generate_session_id()

    session_data = sessions.get(normalized_session_id)

    if session_data:
        memory = ConversationMemory.from_dict(session_data.get("memory", {}))
        session_data["updated_at"] = _now()
        sessions[normalized_session_id] = session_data
        _save_sessions(sessions)
        return normalized_session_id, memory, False

    memory = ConversationMemory()
    sessions[normalized_session_id] = {
        "session_id": normalized_session_id,
        "created_at": _now(),
        "updated_at": _now(),
        "memory": memory.to_dict(),
    }
    _save_sessions(sessions)

    return normalized_session_id, memory, True


def save_session(session_id: str, memory: ConversationMemory):
    sessions = _load_sessions()
    normalized_session_id = session_id.strip()

    if not normalized_session_id:
        raise ValueError("session_id 不能为空")

    existing = sessions.get(normalized_session_id, {})
    sessions[normalized_session_id] = {
        "session_id": normalized_session_id,
        "created_at": existing.get("created_at", _now()),
        "updated_at": _now(),
        "memory": memory.to_dict(),
    }

    _save_sessions(sessions)


def reset_session(session_id: str) -> Dict[str, Any]:
    sessions = _load_sessions()
    normalized_session_id = session_id.strip()

    if not normalized_session_id or normalized_session_id not in sessions:
        raise KeyError(session_id)

    memory = ConversationMemory()
    sessions[normalized_session_id] = {
        "session_id": normalized_session_id,
        "created_at": sessions[normalized_session_id].get("created_at", _now()),
        "updated_at": _now(),
        "memory": memory.to_dict(),
    }
    _save_sessions(sessions)

    return sessions[normalized_session_id]


def get_session_snapshot(session_id: str) -> Dict[str, Any]:
    sessions = _load_sessions()
    normalized_session_id = session_id.strip()

    if not normalized_session_id or normalized_session_id not in sessions:
        raise KeyError(session_id)

    return sessions[normalized_session_id]
