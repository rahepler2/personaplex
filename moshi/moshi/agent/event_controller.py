# SPDX-License-Identifier: MIT
"""
Phase 2 event controller for the agent orchestration layer.

Receives turn.candidate events from the TurnPolicy and decides
whether to accept, defer, or reject the turn.  For the PoC the
policy is "always accept", but the class is structured so more
sophisticated decision logic can be plugged in later.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Callable, Coroutine, Any, Optional

from .transcript import TranscriptEvent

logger = logging.getLogger(__name__)


@dataclass
class AcceptedTurn:
    """A turn that has been accepted by the event controller."""
    turn_id: str
    text: str
    timestamp: float


TurnAcceptedCallback = Callable[[AcceptedTurn], Coroutine[Any, Any, None]]


class EventController:
    """Decides whether to accept turn candidates.

    For the PoC this always accepts.  Swap in a custom ``accept_policy``
    callable to add smarter logic (confidence thresholds, debounce,
    intent classification, etc.).

    Usage::

        ctrl = EventController()
        events = ctrl.push(turn_candidate_event)
        # events contains a turn.accepted TranscriptEvent

        # Pull-based consumption:
        accepted = await ctrl.get_accepted_turn()

        # Push-based consumption:
        ctrl = EventController(on_accepted=my_callback)
    """

    def __init__(
        self,
        *,
        accept_policy: Optional[Callable[[TranscriptEvent], bool]] = None,
        on_accepted: Optional[TurnAcceptedCallback] = None,
    ):
        self._accept_policy = accept_policy or self._default_policy
        self._on_accepted = on_accepted
        self._accepted_queue: asyncio.Queue[AcceptedTurn] = asyncio.Queue()
        self._accepted_turns: list[AcceptedTurn] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, event: TranscriptEvent) -> list[TranscriptEvent]:
        """Process a transcript event.  Returns any new events to emit.

        Handles:
        - turn.candidate  -> decide accept/reject -> emit turn.accepted
        - turn.cancelled   -> acknowledge cancellation
        """
        results: list[TranscriptEvent] = []

        if event.type == "turn.candidate":
            if self._accept_policy(event):
                import time
                accepted = AcceptedTurn(
                    turn_id=event.turn_id or str(uuid.uuid4()),
                    text=event.text or "",
                    timestamp=time.time(),
                )
                self._accepted_turns.append(accepted)
                self._accepted_queue.put_nowait(accepted)

                results.append(TranscriptEvent(
                    type="turn.accepted",
                    turn_id=accepted.turn_id,
                    text=accepted.text,
                ))
                logger.info(
                    f"Turn accepted: {accepted.turn_id} "
                    f"(text={accepted.text[:60]!r})"
                )

        elif event.type == "turn.cancelled":
            logger.info(
                f"Turn cancelled acknowledged: {event.turn_id}"
            )

        return results

    async def push_async(self, event: TranscriptEvent) -> list[TranscriptEvent]:
        """Async variant of push that also fires the on_accepted callback."""
        results = self.push(event)
        if self._on_accepted is not None:
            for r in results:
                if r.type == "turn.accepted":
                    # Find the matching AcceptedTurn
                    for at in reversed(self._accepted_turns):
                        if at.turn_id == r.turn_id:
                            await self._on_accepted(at)
                            break
        return results

    async def get_accepted_turn(self) -> AcceptedTurn:
        """Block until the next turn is accepted.  Pull-based API."""
        return await self._accepted_queue.get()

    @property
    def accepted_turns(self) -> list[AcceptedTurn]:
        """All turns accepted during this session (read-only view)."""
        return list(self._accepted_turns)

    def reset(self):
        """Clear state between sessions."""
        self._accepted_turns.clear()
        # Drain the queue
        while not self._accepted_queue.empty():
            try:
                self._accepted_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ------------------------------------------------------------------
    # Policies
    # ------------------------------------------------------------------

    @staticmethod
    def _default_policy(event: TranscriptEvent) -> bool:
        """PoC policy: always accept if there is non-empty text."""
        return bool(event.text and event.text.strip())
