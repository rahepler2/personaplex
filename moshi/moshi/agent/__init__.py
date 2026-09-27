# SPDX-License-Identifier: MIT
"""
Agent orchestration layer for PersonaPlex.

Phase 1A: Pluggable ASR transcript capture
Phase 1B: Utterance assembly and turn detection
Phase 2:  Event controller (turn accept/reject decisions)
"""

from .events import Event, EventBus, EventKind
from .transcript import (
    TranscriptEvent,
    TranscriptProvider,
    MockTranscriptProvider,
    WhisperTranscriptProvider,
)
from .utterance import UtteranceAssembler, UtteranceFragment, UtteranceUpdate, UtteranceFinal
from .turn_policy import TurnPolicy, TurnConfig, TurnCandidate
from .event_controller import EventController, AcceptedTurn, TurnAcceptedCallback
from .session import VoiceSession

__all__ = [
    "Event",
    "EventBus",
    "EventKind",
    "TranscriptEvent",
    "TranscriptProvider",
    "MockTranscriptProvider",
    "WhisperTranscriptProvider",
    "UtteranceAssembler",
    "UtteranceFragment",
    "UtteranceUpdate",
    "UtteranceFinal",
    "TurnPolicy",
    "TurnConfig",
    "TurnCandidate",
    "EventController",
    "AcceptedTurn",
    "TurnAcceptedCallback",
    "VoiceSession",
]
