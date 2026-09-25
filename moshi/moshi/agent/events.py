# SPDX-License-Identifier: MIT
"""Lightweight event bus for decoupled component communication."""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine


logger = logging.getLogger(__name__)


class EventKind(Enum):
    SPEECH_STARTED = auto()
    SPEECH_STOPPED = auto()
    TRANSCRIPT_PARTIAL = auto()
    TRANSCRIPT_FINAL = auto()
    ERROR = auto()


@dataclass
class Event:
    kind: EventKind
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: __import__("time").time())


EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Simple async pub/sub event bus."""

    def __init__(self):
        self._listeners: dict[EventKind, list[EventHandler]] = {}

    def on(self, event_kind: EventKind, handler: EventHandler):
        self._listeners.setdefault(event_kind, []).append(handler)

    def off(self, event_kind: EventKind, handler: EventHandler):
        handlers = self._listeners.get(event_kind, [])
        if handler in handlers:
            handlers.remove(handler)

    async def emit(self, event: Event):
        for handler in self._listeners.get(event.kind, []):
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Event handler error for {event.kind}: {e}")
