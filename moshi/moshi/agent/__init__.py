# SPDX-License-Identifier: MIT
"""
Agent orchestration layer for PersonaPlex.

Phase 1A: Pluggable ASR transcript capture
Phase 1B: Utterance assembly and turn detection
Phase 2:  Event controller (turn accept/reject decisions)
Phase 3:  Sidecar LLM orchestrator (tool calling + RAG reasoning)
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
from .sidecar import SidecarLLM, SidecarConfig, SidecarProvider, SidecarResponse, ToolCall, ToolResult
from .orchestrator import ConversationOrchestrator, OrchestratorConfig, OrchestratorEvent, OrchestratorCallback

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
    "SidecarLLM",
    "SidecarConfig",
    "SidecarProvider",
    "SidecarResponse",
    "ToolCall",
    "ToolResult",
    "ConversationOrchestrator",
    "OrchestratorConfig",
    "OrchestratorEvent",
    "OrchestratorCallback",
]
