# SPDX-License-Identifier: MIT
"""
VoiceSession: lifecycle wrapper for the agent orchestration pipeline.

Binds a TranscriptProvider, UtteranceAssembler, and TurnPolicy into a
single session object that processes PCM audio and yields ready-to-send
TranscriptEvents.
"""

import asyncio
import logging
import time
from typing import AsyncIterator, Optional

import numpy as np

from .events import Event, EventBus, EventKind
from .transcript import TranscriptEvent, TranscriptProvider
from .utterance import UtteranceAssembler
from .turn_policy import TurnPolicy, TurnConfig

logger = logging.getLogger(__name__)

TICK_INTERVAL_S = 0.1


class VoiceSession:
    """Wraps provider + assembler + policy into a coherent session.

    Usage::

        session = VoiceSession(provider)
        async for event in session.run():
            # forward event to WebSocket client
            ...
    """

    def __init__(
        self,
        provider: TranscriptProvider,
        merge_gap: float = 1.5,
        turn_config: Optional[TurnConfig] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.provider = provider
        self.assembler = UtteranceAssembler(merge_gap=merge_gap)
        self.turn_policy = TurnPolicy(config=turn_config)
        self.event_bus = event_bus
        self._closed = False

    async def push_audio(self, pcm: np.ndarray) -> None:
        """Forward PCM to the underlying provider."""
        if not self._closed:
            await self.provider.push_audio(pcm)

    async def run(self) -> AsyncIterator[TranscriptEvent]:
        """Consume provider events, run assembler + policy, yield results."""
        tick_task = asyncio.create_task(self._tick_loop())
        try:
            async for event in self.provider.events():
                if self._closed:
                    break

                yield event

                if self.event_bus:
                    await self._emit_bus_event(event)

                assembled = self.assembler.push(event)
                for ae in assembled:
                    yield ae
                    policy_events = self.turn_policy.push(ae)
                    for pe in policy_events:
                        yield pe

                policy_events = self.turn_policy.push(event)
                for pe in policy_events:
                    yield pe

        finally:
            tick_task.cancel()
            try:
                await tick_task
            except asyncio.CancelledError:
                pass

    async def _tick_loop(self):
        """Periodic tick for assembler gap detection and turn policy."""
        while not self._closed:
            await asyncio.sleep(TICK_INTERVAL_S)

            tick_event = self.assembler.tick()
            if tick_event is not None:
                pass

            turn_event = self.turn_policy.tick()
            if turn_event is not None:
                pass

    async def close(self):
        """Shut down the session and flush pending state."""
        self._closed = True
        flush_event = self.assembler.flush()
        if flush_event is not None:
            logger.info(f"Session flush: {flush_event.utterance_id}")
        self.turn_policy.reset()
        await self.provider.close()

    async def _emit_bus_event(self, event: TranscriptEvent):
        """Map TranscriptEvent types to EventBus events."""
        if self.event_bus is None:
            return
        kind_map = {
            "speech.started": EventKind.SPEECH_STARTED,
            "speech.stopped": EventKind.SPEECH_STOPPED,
            "user_transcript.partial": EventKind.TRANSCRIPT_PARTIAL,
            "user_transcript.final": EventKind.TRANSCRIPT_FINAL,
        }
        kind = kind_map.get(event.type)
        if kind is not None:
            await self.event_bus.emit(Event(
                kind=kind,
                data={"text": event.text, "segment_id": event.segment_id},
            ))
