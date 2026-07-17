"""Read/write operations on chat_sessions/chat_messages -- sibling to crud.py's
ingestion-catalog operations, split into its own file since chat persistence is a
distinct concern (query-time, not ingestion-time) sharing only the same DuckDB
connection/table-creation machinery. Same "explicit connection argument" convention
as crud.py, for the same reason (verification scripts can pass an isolated
in-memory connection)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

import duckdb

from .connection import get_connection
from .errors import NotFoundError
from .models import ChatMessageRecord, ChatSessionRecord

_TITLE_MAX_LEN = 60


def _new_id() -> str:
    return str(uuid.uuid4())


def _title_from_message(message: str) -> str:
    stripped = " ".join(message.split())
    if len(stripped) <= _TITLE_MAX_LEN:
        return stripped or "New chat"
    return stripped[: _TITLE_MAX_LEN - 1].rstrip() + "…"


def create_session(first_message: str, connection: duckdb.DuckDBPyConnection = None) -> ChatSessionRecord:
    connection = connection or get_connection()
    session_id = _new_id()
    now = datetime.now()
    connection.execute(
        "INSERT INTO chat_sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        [session_id, _title_from_message(first_message), now, now],
    )
    return ChatSessionRecord(session_id=session_id, title=_title_from_message(first_message), created_at=now, updated_at=now)


def list_sessions(connection: duckdb.DuckDBPyConnection = None) -> list[ChatSessionRecord]:
    connection = connection or get_connection()
    rows = connection.execute(
        "SELECT session_id, title, created_at, updated_at FROM chat_sessions ORDER BY updated_at DESC"
    ).fetchall()
    return [ChatSessionRecord(session_id=r[0], title=r[1], created_at=r[2], updated_at=r[3]) for r in rows]


def get_session(session_id: str, connection: duckdb.DuckDBPyConnection = None) -> ChatSessionRecord:
    connection = connection or get_connection()
    row = connection.execute(
        "SELECT session_id, title, created_at, updated_at FROM chat_sessions WHERE session_id = ?", [session_id]
    ).fetchone()
    if row is None:
        raise NotFoundError(f"No chat session {session_id!r}")
    return ChatSessionRecord(session_id=row[0], title=row[1], created_at=row[2], updated_at=row[3])


def touch_session(session_id: str, connection: duckdb.DuckDBPyConnection = None) -> None:
    connection = connection or get_connection()
    connection.execute("UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?", [datetime.now(), session_id])


def delete_session(session_id: str, connection: duckdb.DuckDBPyConnection = None) -> None:
    connection = connection or get_connection()
    connection.execute("DELETE FROM chat_messages WHERE session_id = ?", [session_id])
    connection.execute("DELETE FROM chat_sessions WHERE session_id = ?", [session_id])


def append_message(
    session_id: str,
    role: str,
    content: str,
    citations: Optional[list[dict]] = None,
    model_used: Optional[str] = None,
    connection: duckdb.DuckDBPyConnection = None,
) -> ChatMessageRecord:
    connection = connection or get_connection()
    message_id = _new_id()
    now = datetime.now()
    citations_json = json.dumps(citations) if citations is not None else None
    connection.execute(
        "INSERT INTO chat_messages (message_id, session_id, role, content, citations, model_used, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [message_id, session_id, role, content, citations_json, model_used, now],
    )
    return ChatMessageRecord(
        message_id=message_id,
        session_id=session_id,
        role=role,
        content=content,
        citations=citations,
        model_used=model_used,
        created_at=now,
    )


def list_messages(session_id: str, connection: duckdb.DuckDBPyConnection = None) -> list[ChatMessageRecord]:
    connection = connection or get_connection()
    rows = connection.execute(
        "SELECT message_id, session_id, role, content, citations, model_used, created_at "
        "FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
        [session_id],
    ).fetchall()
    return [
        ChatMessageRecord(
            message_id=r[0],
            session_id=r[1],
            role=r[2],
            content=r[3],
            citations=json.loads(r[4]) if r[4] is not None else None,
            model_used=r[5],
            created_at=r[6],
        )
        for r in rows
    ]
