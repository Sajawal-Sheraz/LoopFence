"""Session state storage.

All state changes go through ``SessionStore.update``, which must run ``fn`` atomically for that
session. The in-memory store uses a lock; a future Redis store can implement the same contract
with an optimistic transaction (WATCH/MULTI) while the rules stay in Python.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable
from typing import Protocol

from .models import SessionState


class SessionStore(Protocol):
    def create(self, state: SessionState) -> None:
        """Register a new session. Raises ``ValueError`` if the id already exists."""
        ...

    def update[T](self, session_id: str, fn: Callable[[SessionState], T]) -> T:
        """Run ``fn`` on the session's state atomically and persist its changes."""
        ...

    def snapshot(self, session_id: str) -> SessionState:
        """A detached copy of the session's state, safe to read without locking."""
        ...


class InMemoryStore:
    """Thread-safe, process-local store. One lock for the whole store (v1 simplicity)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}

    def create(self, state: SessionState) -> None:
        with self._lock:
            if state.session_id in self._sessions:
                raise ValueError(f"session {state.session_id!r} already exists")
            self._sessions[state.session_id] = state

    def update[T](self, session_id: str, fn: Callable[[SessionState], T]) -> T:
        with self._lock:
            return fn(self._sessions[session_id])

    def snapshot(self, session_id: str) -> SessionState:
        with self._lock:
            return copy.deepcopy(self._sessions[session_id])
