# SPDX-License-Identifier: MIT
"""
Turn detection policy.

Phase 1B: Pause-based turn detection. Listens for finalized
utterances and speech silence to decide when the user has
completed a conversational turn.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .transcript import TranscriptEvent

logger = logging.getLogger(__name__)

SILENCE_THRESHOLD_S = 1.0


@dataclass
class TurnConfig:
    """Tuneable turn-detection parameters."""
    silence_threshold: float = SILENCE_THRESHOLD_S
    min_utterance_length: int = 1


@dataclass
class TurnCandidate:
    """Proposed turn ready for the event controller."""
    turn_id: str
    utterance_id: str
    text: str
    silence_ms: float
    timestamp: float


class TurnPolicy:
    """Pause-based turn detection.

    After an utterance is finalized, if silence exceeds
    ``silence_threshold`` seconds a turn.candidate event is emitted.
    The event controller (Phase 2) decides whether to accept it.
    """

    def __init__(self, config: Optional[TurnConfig] = None):
        self.config = config or TurnConfig()
        self._last_utterance: Optional[TranscriptEvent] = None
        self._last_speech_stop: float = 0.0
        self._pending_turn: Optional[TurnCandidate] = None
        self._emitted_for_utterance: Optional[str] = None

    def push(self, event: TranscriptEvent) -> list[TranscriptEvent]:
        """Process an event from the assembler or VAD."""
        results: list[TranscriptEvent] = []

        if event.type == "utterance.finalized":
            self._last_utterance = event
            self._pending_turn = None

        elif event.type == "speech.stopped":
            self._last_speech_stop = time.time()

        elif event.type == "speech.started":
            if self._pending_turn is not None:
                results.append(TranscriptEvent(
                    type="turn.cancelled",
                    turn_id=self._pending_turn.turn_id,
                ))
                self._pending_turn = None

        return results

    def tick(self) -> Optional[TranscriptEvent]:
        """Called periodically to check silence duration."""
        if self._last_utterance is None:
            return None

        if self._emitted_for_utterance == self._last_utterance.utterance_id:
            return None

        text = self._last_utterance.text or ""
        if len(text.strip()) < self.config.min_utterance_length:
            return None

        now = time.time()
        silence = now - self._last_speech_stop if self._last_speech_stop > 0 else 0

        if silence >= self.config.silence_threshold:
            turn_id = str(uuid.uuid4())
            self._pending_turn = TurnCandidate(
                turn_id=turn_id,
                utterance_id=self._last_utterance.utterance_id or "",
                text=text,
                silence_ms=silence * 1000,
                timestamp=now,
            )
            self._emitted_for_utterance = self._last_utterance.utterance_id

            logger.info(
                f"Turn candidate: {turn_id} "
                f"(silence={silence*1000:.0f}ms, text={text[:60]!r})"
            )

            return TranscriptEvent(
                type="turn.candidate",
                turn_id=turn_id,
                utterance_id=self._last_utterance.utterance_id,
                text=text,
            )

        return None

    def reset(self):
        """Clear state between sessions."""
        self._last_utterance = None
        self._last_speech_stop = 0.0
        self._pending_turn = None
        self._emitted_for_utterance = None
