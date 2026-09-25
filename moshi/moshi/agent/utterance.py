# SPDX-License-Identifier: MIT
"""
Utterance assembly from ASR transcript segments.

Phase 1B: Gap-based merge of sequential ASR final segments into
coherent utterances. Emits utterance.updated / utterance.finalized events.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .transcript import TranscriptEvent

logger = logging.getLogger(__name__)

MERGE_GAP_S = 1.5


@dataclass
class UtteranceFragment:
    """One ASR final segment contributing to an utterance."""
    segment_id: str
    sequence: int
    text: str
    timestamp: float


@dataclass
class UtteranceUpdate:
    """Emitted each time a new fragment extends the current utterance."""
    utterance_id: str
    text: str
    segment_ids: list[str]
    timestamp: float


@dataclass
class UtteranceFinal:
    """Emitted when a gap exceeds MERGE_GAP_S, closing the utterance."""
    utterance_id: str
    text: str
    segment_ids: list[str]
    start_time: float
    end_time: float
    duration_ms: float


class UtteranceAssembler:
    """Merges sequential ASR finals into utterances based on time gaps.

    When the gap between the last fragment and a new one exceeds
    ``merge_gap`` seconds the current utterance is finalized and a
    new one begins.
    """

    def __init__(self, merge_gap: float = MERGE_GAP_S):
        self.merge_gap = merge_gap
        self._utterance_id: Optional[str] = None
        self._fragments: list[UtteranceFragment] = []
        self._start_time: float = 0.0
        self._last_time: float = 0.0

    def push(self, event: TranscriptEvent) -> list[TranscriptEvent]:
        """Accept a ``user_transcript.final`` event, return assembled events."""
        if event.type != "user_transcript.final":
            return []

        now = time.time()
        results: list[TranscriptEvent] = []

        if self._utterance_id is not None and (now - self._last_time) > self.merge_gap:
            results.append(self._finalize(now))

        if self._utterance_id is None:
            self._utterance_id = str(uuid.uuid4())
            self._fragments = []
            self._start_time = now

        frag = UtteranceFragment(
            segment_id=event.segment_id or "",
            sequence=event.sequence or 0,
            text=event.text or "",
            timestamp=now,
        )
        self._fragments.append(frag)
        self._last_time = now

        merged_text = " ".join(f.text for f in self._fragments if f.text)
        results.append(TranscriptEvent(
            type="utterance.updated",
            utterance_id=self._utterance_id,
            text=merged_text,
            segment_ids=[f.segment_id for f in self._fragments],
        ))

        return results

    def flush(self) -> Optional[TranscriptEvent]:
        """Force-finalize the current utterance (e.g. on session end)."""
        if self._utterance_id is not None:
            return self._finalize(time.time())
        return None

    def tick(self) -> Optional[TranscriptEvent]:
        """Called periodically; finalizes if the gap has elapsed."""
        if self._utterance_id is None:
            return None
        now = time.time()
        if (now - self._last_time) > self.merge_gap:
            return self._finalize(now)
        return None

    def _finalize(self, now: float) -> TranscriptEvent:
        merged_text = " ".join(f.text for f in self._fragments if f.text)
        segment_ids = [f.segment_id for f in self._fragments]
        duration = (now - self._start_time) * 1000

        event = TranscriptEvent(
            type="utterance.finalized",
            utterance_id=self._utterance_id,
            text=merged_text,
            segment_ids=segment_ids,
            duration_ms=duration,
        )

        logger.info(
            f"Utterance finalized: {self._utterance_id} "
            f"({len(self._fragments)} fragments, {duration:.0f}ms)"
        )

        self._utterance_id = None
        self._fragments = []
        return event
