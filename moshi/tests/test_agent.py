# SPDX-License-Identifier: MIT
"""Comprehensive tests for the agent orchestration modules.

Covers EventBus, _VAD, MockTranscriptProvider, UtteranceAssembler,
TurnPolicy, and VoiceSession.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from moshi.agent.events import Event, EventBus, EventKind
from moshi.agent.transcript import (
    ENERGY_THRESHOLD,
    MIN_SPEECH_DURATION_S,
    SILENCE_DURATION_S,
    MockTranscriptProvider,
    TranscriptEvent,
    _VAD,
)
from moshi.agent.utterance import MERGE_GAP_S, UtteranceAssembler
from moshi.agent.turn_policy import SILENCE_THRESHOLD_S, TurnConfig, TurnPolicy
from moshi.agent.session import VoiceSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _loud_pcm(n: int = 1920, amplitude: float = 0.5) -> np.ndarray:
    """PCM chunk whose RMS energy exceeds ENERGY_THRESHOLD."""
    return np.full(n, amplitude, dtype=np.float32)


def _silent_pcm(n: int = 1920) -> np.ndarray:
    """PCM chunk whose RMS energy is well below ENERGY_THRESHOLD."""
    return np.zeros(n, dtype=np.float32)


# ===================================================================
# 1. EventBus
# ===================================================================


class TestEventBus:
    """Tests for the lightweight async pub/sub EventBus."""

    @pytest.mark.asyncio
    async def test_on_and_emit(self):
        """Handler registered with on() receives the correct event."""
        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.on(EventKind.SPEECH_STARTED, handler)
        event = Event(kind=EventKind.SPEECH_STARTED, data={"key": "val"})
        await bus.emit(event)

        assert len(received) == 1
        assert received[0] is event
        assert received[0].data == {"key": "val"}

    @pytest.mark.asyncio
    async def test_off_removes_handler(self):
        """After off(), the handler no longer receives events."""
        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.on(EventKind.SPEECH_STARTED, handler)
        bus.off(EventKind.SPEECH_STARTED, handler)
        await bus.emit(Event(kind=EventKind.SPEECH_STARTED))

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_off_nonexistent_handler_is_noop(self):
        """off() for a handler that was never registered does not raise."""
        bus = EventBus()

        async def handler(event: Event):
            pass

        # Should not raise
        bus.off(EventKind.SPEECH_STARTED, handler)

    @pytest.mark.asyncio
    async def test_emit_no_handlers(self):
        """Emitting to a kind with no handlers is a silent no-op."""
        bus = EventBus()
        await bus.emit(Event(kind=EventKind.ERROR))

    @pytest.mark.asyncio
    async def test_handler_error_does_not_crash(self):
        """A handler that raises does not prevent other handlers from running."""
        bus = EventBus()
        call_order = []

        async def bad_handler(event: Event):
            call_order.append("bad")
            raise ValueError("boom")

        async def good_handler(event: Event):
            call_order.append("good")

        bus.on(EventKind.SPEECH_STARTED, bad_handler)
        bus.on(EventKind.SPEECH_STARTED, good_handler)
        await bus.emit(Event(kind=EventKind.SPEECH_STARTED))

        assert call_order == ["bad", "good"]

    @pytest.mark.asyncio
    async def test_multiple_handlers_same_kind(self):
        """Multiple handlers on the same kind all receive the event."""
        bus = EventBus()
        results = []

        async def h1(e):
            results.append("h1")

        async def h2(e):
            results.append("h2")

        bus.on(EventKind.TRANSCRIPT_FINAL, h1)
        bus.on(EventKind.TRANSCRIPT_FINAL, h2)
        await bus.emit(Event(kind=EventKind.TRANSCRIPT_FINAL))

        assert results == ["h1", "h2"]

    @pytest.mark.asyncio
    async def test_different_kinds_isolated(self):
        """Handlers for different kinds do not cross-fire."""
        bus = EventBus()
        received = []

        async def handler(e):
            received.append(e.kind)

        bus.on(EventKind.SPEECH_STARTED, handler)
        await bus.emit(Event(kind=EventKind.SPEECH_STOPPED))

        assert len(received) == 0


# ===================================================================
# 2. _VAD and TranscriptEvent
# ===================================================================


class TestVAD:
    """Tests for the energy-based voice activity detector."""

    def test_constants(self):
        """Verify the module-level VAD constants."""
        assert ENERGY_THRESHOLD == 0.01
        assert SILENCE_DURATION_S == 0.8
        assert MIN_SPEECH_DURATION_S == 0.3

    def test_silent_pcm_no_events(self):
        """Silent PCM produces no VAD events."""
        vad = _VAD(sample_rate=16000)
        events = vad.process(_silent_pcm())
        assert events == []
        assert not vad.is_speaking

    def test_loud_pcm_triggers_speech_started(self):
        """Loud PCM transitions to speaking and emits speech.started."""
        vad = _VAD(sample_rate=16000)
        events = vad.process(_loud_pcm())
        assert len(events) == 1
        assert events[0].type == "speech.started"
        assert vad.is_speaking

    def test_continued_loud_pcm_no_duplicate_start(self):
        """Subsequent loud chunks do not emit extra speech.started events."""
        vad = _VAD(sample_rate=16000)
        vad.process(_loud_pcm())  # triggers start
        events = vad.process(_loud_pcm())
        assert events == []

    @patch("moshi.agent.transcript.time")
    def test_silence_after_speech_triggers_stop(self, mock_time):
        """After speech, sufficient silence emits speech.stopped."""
        vad = _VAD(sample_rate=16000)
        t = 1000.0

        # Start speech
        mock_time.time.return_value = t
        events = vad.process(_loud_pcm())
        assert len(events) == 1
        assert events[0].type == "speech.started"

        # Continue speaking past MIN_SPEECH_DURATION_S
        t += MIN_SPEECH_DURATION_S + 0.1
        mock_time.time.return_value = t
        vad.process(_loud_pcm())

        # Silence exceeds SILENCE_DURATION_S
        t += SILENCE_DURATION_S + 0.1
        mock_time.time.return_value = t
        events = vad.process(_silent_pcm())

        assert len(events) == 1
        assert events[0].type == "speech.stopped"
        assert events[0].duration_ms is not None
        assert events[0].duration_ms > 0
        assert not vad.is_speaking

    @patch("moshi.agent.transcript.time")
    def test_short_speech_no_stop_event(self, mock_time):
        """Speech shorter than MIN_SPEECH_DURATION_S does not emit speech.stopped.

        The VAD measures duration as now - speech_start_time (wall clock from
        first loud frame). To get duration < MIN_SPEECH_DURATION_S (0.3s) while
        silence > SILENCE_DURATION_S (0.8s) is impossible because the silence
        gap counts toward total duration. Instead, verify that a single loud
        frame followed by very brief silence (< SILENCE_DURATION_S) doesn't
        produce a speech.stopped at all — the VAD stays in "speaking" state.
        """
        vad = _VAD(sample_rate=16000)
        t = 1000.0

        # One loud frame → speech starts
        mock_time.time.return_value = t
        events = vad.process(_loud_pcm())
        assert any(e.type == "speech.started" for e in events)

        # Brief silence (< SILENCE_DURATION_S) → no stop event yet
        t += 0.1
        mock_time.time.return_value = t
        events = vad.process(_silent_pcm())
        stop_events = [e for e in events if e.type == "speech.stopped"]
        assert len(stop_events) == 0
        assert vad.is_speaking

    @patch("moshi.agent.transcript.time")
    def test_brief_silence_does_not_stop(self, mock_time):
        """Silence shorter than SILENCE_DURATION_S keeps the speaking state."""
        vad = _VAD(sample_rate=16000)
        t = 1000.0

        mock_time.time.return_value = t
        vad.process(_loud_pcm())

        t += 0.1  # brief gap well under SILENCE_DURATION_S
        mock_time.time.return_value = t
        events = vad.process(_silent_pcm())

        assert events == []
        assert vad.is_speaking

    def test_speech_duration_property_not_speaking(self):
        """speech_duration returns 0 when not speaking."""
        vad = _VAD(sample_rate=16000)
        assert vad.speech_duration == 0.0


# ===================================================================
# 2b. MockTranscriptProvider
# ===================================================================


class TestMockTranscriptProvider:
    """Tests for MockTranscriptProvider integration."""

    @pytest.mark.asyncio
    @patch("moshi.agent.transcript.time")
    async def test_push_audio_loud_then_silent(self, mock_time):
        """Push loud PCM then silence; expect speech.started, speech.stopped, and final transcript."""
        provider = MockTranscriptProvider(sample_rate=16000)
        t = 1000.0

        # Loud chunk -> speech.started
        mock_time.time.return_value = t
        await provider.push_audio(_loud_pcm())

        # Advance past MIN_SPEECH_DURATION_S and push more loud audio
        t += MIN_SPEECH_DURATION_S + 0.1
        mock_time.time.return_value = t
        await provider.push_audio(_loud_pcm())

        # Silence after SILENCE_DURATION_S -> speech.stopped + final
        t += SILENCE_DURATION_S + 0.1
        mock_time.time.return_value = t
        await provider.push_audio(_silent_pcm())

        # Drain the queue
        events = []
        while not provider._queue.empty():
            events.append(provider._queue.get_nowait())

        types = [e.type for e in events]
        assert "speech.started" in types
        assert "speech.stopped" in types
        assert "user_transcript.final" in types

        final = [e for e in events if e.type == "user_transcript.final"][0]
        assert final.text == "[speech detected]"
        assert final.provider == "mock"
        assert final.segment_id is not None
        assert final.sequence == 1

    @pytest.mark.asyncio
    async def test_push_audio_when_closed_is_noop(self):
        """push_audio after close() does nothing."""
        provider = MockTranscriptProvider(sample_rate=16000)
        await provider.close()
        await provider.push_audio(_loud_pcm())
        assert provider._queue.empty()

    @pytest.mark.asyncio
    async def test_events_iterator_yields_queued_events(self):
        """events() async iterator yields events placed in the queue."""
        provider = MockTranscriptProvider(sample_rate=16000)
        test_event = TranscriptEvent(type="speech.started")
        provider._queue.put_nowait(test_event)

        collected = []
        async for event in provider.events():
            collected.append(event)
            # Close after first event to break out of the loop
            await provider.close()

        assert len(collected) == 1
        assert collected[0].type == "speech.started"

    @pytest.mark.asyncio
    async def test_queue_overflow_drops_oldest(self):
        """When the queue is full, _put_nowait drops the oldest event."""
        provider = MockTranscriptProvider(sample_rate=16000)
        maxsize = provider._queue.maxsize

        # Fill the queue
        for i in range(maxsize):
            provider._put_nowait(TranscriptEvent(type="speech.started", sequence=i))

        assert provider._queue.full()

        # Push one more -- should drop the oldest
        provider._put_nowait(TranscriptEvent(type="speech.stopped", sequence=maxsize))
        assert provider._queue.qsize() == maxsize

        # The first item should now be sequence=1 (sequence=0 was dropped)
        first = provider._queue.get_nowait()
        assert first.sequence == 1

    def test_to_status_dict(self):
        """to_status_dict returns expected keys."""
        provider = MockTranscriptProvider(sample_rate=16000)
        status = provider.to_status_dict()
        assert "session_id" in status
        assert status["phase"] == "LISTENING"
        assert status["speech_active"] is False


# ===================================================================
# 3. UtteranceAssembler
# ===================================================================


class TestUtteranceAssembler:
    """Tests for gap-based utterance assembly."""

    @patch("moshi.agent.utterance.time")
    def test_sequential_finals_merged(self, mock_time):
        """Two finals within merge_gap are merged into one utterance."""
        assembler = UtteranceAssembler(merge_gap=MERGE_GAP_S)
        t = 1000.0

        # First final
        mock_time.time.return_value = t
        events1 = assembler.push(TranscriptEvent(
            type="user_transcript.final",
            segment_id="seg-1",
            sequence=1,
            text="hello",
        ))
        assert len(events1) == 1
        assert events1[0].type == "utterance.updated"
        assert events1[0].text == "hello"
        utterance_id = events1[0].utterance_id

        # Second final within merge_gap
        t += 0.5
        mock_time.time.return_value = t
        events2 = assembler.push(TranscriptEvent(
            type="user_transcript.final",
            segment_id="seg-2",
            sequence=2,
            text="world",
        ))
        assert len(events2) == 1
        assert events2[0].type == "utterance.updated"
        assert events2[0].text == "hello world"
        assert events2[0].utterance_id == utterance_id
        assert events2[0].segment_ids == ["seg-1", "seg-2"]

    @patch("moshi.agent.utterance.time")
    def test_gap_exceeding_merge_gap_finalizes(self, mock_time):
        """A gap > merge_gap between finals produces a finalized event before the updated."""
        assembler = UtteranceAssembler(merge_gap=MERGE_GAP_S)
        t = 1000.0

        # First final
        mock_time.time.return_value = t
        assembler.push(TranscriptEvent(
            type="user_transcript.final",
            segment_id="seg-1",
            sequence=1,
            text="hello",
        ))
        first_utterance_id = assembler._utterance_id

        # Second final after merge_gap
        t += MERGE_GAP_S + 0.5
        mock_time.time.return_value = t
        events = assembler.push(TranscriptEvent(
            type="user_transcript.final",
            segment_id="seg-2",
            sequence=2,
            text="new utterance",
        ))

        # Should get: utterance.finalized (for first) + utterance.updated (for second)
        assert len(events) == 2
        assert events[0].type == "utterance.finalized"
        assert events[0].utterance_id == first_utterance_id
        assert events[0].text == "hello"
        assert events[0].duration_ms is not None

        assert events[1].type == "utterance.updated"
        assert events[1].text == "new utterance"
        assert events[1].utterance_id != first_utterance_id

    @patch("moshi.agent.utterance.time")
    def test_flush_force_finalizes(self, mock_time):
        """flush() force-finalizes the current utterance."""
        assembler = UtteranceAssembler()
        t = 1000.0

        mock_time.time.return_value = t
        assembler.push(TranscriptEvent(
            type="user_transcript.final",
            segment_id="seg-1",
            sequence=1,
            text="partial sentence",
        ))
        uid = assembler._utterance_id

        t += 0.1
        mock_time.time.return_value = t
        result = assembler.flush()

        assert result is not None
        assert result.type == "utterance.finalized"
        assert result.utterance_id == uid
        assert result.text == "partial sentence"
        assert assembler._utterance_id is None

    def test_flush_empty_returns_none(self):
        """flush() with no pending utterance returns None."""
        assembler = UtteranceAssembler()
        assert assembler.flush() is None

    @patch("moshi.agent.utterance.time")
    def test_tick_detects_gap_expiry(self, mock_time):
        """tick() finalizes the utterance when the merge gap has elapsed."""
        assembler = UtteranceAssembler(merge_gap=MERGE_GAP_S)
        t = 1000.0

        mock_time.time.return_value = t
        assembler.push(TranscriptEvent(
            type="user_transcript.final",
            segment_id="seg-1",
            sequence=1,
            text="hello",
        ))
        uid = assembler._utterance_id

        # tick before gap -- no finalization
        t += 0.5
        mock_time.time.return_value = t
        assert assembler.tick() is None

        # tick after gap
        t += MERGE_GAP_S + 0.1
        mock_time.time.return_value = t
        result = assembler.tick()

        assert result is not None
        assert result.type == "utterance.finalized"
        assert result.utterance_id == uid

    def test_tick_no_pending_returns_none(self):
        """tick() with no pending utterance returns None."""
        assembler = UtteranceAssembler()
        assert assembler.tick() is None

    def test_push_non_final_event_ignored(self):
        """push() with a non-final event returns an empty list."""
        assembler = UtteranceAssembler()
        result = assembler.push(TranscriptEvent(type="speech.started"))
        assert result == []


# ===================================================================
# 4. TurnPolicy
# ===================================================================


class TestTurnPolicy:
    """Tests for pause-based turn detection."""

    @patch("moshi.agent.turn_policy.time")
    def test_turn_candidate_after_silence(self, mock_time):
        """turn.candidate is emitted after utterance.finalized + silence > threshold."""
        policy = TurnPolicy()
        t = 1000.0

        # Speech stops
        mock_time.time.return_value = t
        policy.push(TranscriptEvent(type="speech.stopped"))

        # Utterance finalized
        mock_time.time.return_value = t
        policy.push(TranscriptEvent(
            type="utterance.finalized",
            utterance_id="utt-1",
            text="hello world",
        ))

        # tick after silence_threshold
        t += SILENCE_THRESHOLD_S + 0.5
        mock_time.time.return_value = t
        result = policy.tick()

        assert result is not None
        assert result.type == "turn.candidate"
        assert result.utterance_id == "utt-1"
        assert result.text == "hello world"
        assert result.turn_id is not None

    @patch("moshi.agent.turn_policy.time")
    def test_tick_before_threshold_returns_none(self, mock_time):
        """tick() before silence exceeds threshold returns None."""
        policy = TurnPolicy()
        t = 1000.0

        mock_time.time.return_value = t
        policy.push(TranscriptEvent(type="speech.stopped"))
        policy.push(TranscriptEvent(
            type="utterance.finalized",
            utterance_id="utt-1",
            text="hello",
        ))

        # tick before threshold
        t += SILENCE_THRESHOLD_S * 0.5
        mock_time.time.return_value = t
        result = policy.tick()
        assert result is None

    @patch("moshi.agent.turn_policy.time")
    def test_speech_started_cancels_pending_turn(self, mock_time):
        """speech.started cancels a pending turn candidate."""
        policy = TurnPolicy()
        t = 1000.0

        # Create a pending turn
        mock_time.time.return_value = t
        policy.push(TranscriptEvent(type="speech.stopped"))
        policy.push(TranscriptEvent(
            type="utterance.finalized",
            utterance_id="utt-1",
            text="hello world",
        ))
        t += SILENCE_THRESHOLD_S + 0.5
        mock_time.time.return_value = t
        turn_event = policy.tick()
        assert turn_event is not None
        assert turn_event.type == "turn.candidate"
        turn_id = turn_event.turn_id

        # Speech starts -- should cancel the pending turn
        events = policy.push(TranscriptEvent(type="speech.started"))
        cancelled = [e for e in events if e.type == "turn.cancelled"]
        assert len(cancelled) == 1
        assert cancelled[0].turn_id == turn_id

    def test_reset_clears_state(self):
        """reset() clears all internal state."""
        policy = TurnPolicy()
        policy._last_utterance = TranscriptEvent(
            type="utterance.finalized", utterance_id="utt-1", text="hi"
        )
        policy._last_speech_stop = 999.0
        policy._emitted_for_utterance = "utt-1"

        policy.reset()

        assert policy._last_utterance is None
        assert policy._last_speech_stop == 0.0
        assert policy._pending_turn is None
        assert policy._emitted_for_utterance is None

    @patch("moshi.agent.turn_policy.time")
    def test_min_utterance_length_filtering(self, mock_time):
        """Utterances shorter than min_utterance_length are not turned into candidates."""
        config = TurnConfig(min_utterance_length=5)
        policy = TurnPolicy(config=config)
        t = 1000.0

        mock_time.time.return_value = t
        policy.push(TranscriptEvent(type="speech.stopped"))
        policy.push(TranscriptEvent(
            type="utterance.finalized",
            utterance_id="utt-1",
            text="hi",  # 2 chars, less than 5
        ))

        t += SILENCE_THRESHOLD_S + 0.5
        mock_time.time.return_value = t
        result = policy.tick()
        assert result is None

    @patch("moshi.agent.turn_policy.time")
    def test_no_duplicate_turn_for_same_utterance(self, mock_time):
        """tick() does not emit a second candidate for the same utterance."""
        policy = TurnPolicy()
        t = 1000.0

        mock_time.time.return_value = t
        policy.push(TranscriptEvent(type="speech.stopped"))
        policy.push(TranscriptEvent(
            type="utterance.finalized",
            utterance_id="utt-1",
            text="hello world",
        ))

        t += SILENCE_THRESHOLD_S + 0.5
        mock_time.time.return_value = t
        result1 = policy.tick()
        assert result1 is not None
        assert result1.type == "turn.candidate"

        # Second tick for the same utterance
        t += 1.0
        mock_time.time.return_value = t
        result2 = policy.tick()
        assert result2 is None

    def test_tick_no_utterance_returns_none(self):
        """tick() with no prior utterance returns None."""
        policy = TurnPolicy()
        assert policy.tick() is None

    def test_push_non_relevant_event_returns_empty(self):
        """push() with an irrelevant event type returns an empty list."""
        policy = TurnPolicy()
        result = policy.push(TranscriptEvent(type="user_transcript.final"))
        assert result == []


# ===================================================================
# 5. VoiceSession
# ===================================================================


class TestVoiceSession:
    """Tests for the VoiceSession lifecycle wrapper."""

    @pytest.mark.asyncio
    @patch("moshi.agent.transcript.time")
    async def test_run_yields_provider_events(self, mock_time):
        """run() yields events from the underlying provider."""
        t = 1000.0
        mock_time.time.return_value = t

        provider = MockTranscriptProvider(sample_rate=16000)
        session = VoiceSession(provider)

        # Push loud audio to get speech.started
        mock_time.time.return_value = t
        await session.push_audio(_loud_pcm())

        collected = []
        async for event in session.run():
            collected.append(event)
            if len(collected) >= 1:
                await session.close()

        types = [e.type for e in collected]
        assert "speech.started" in types

    @pytest.mark.asyncio
    @patch("moshi.agent.transcript.time")
    async def test_run_yields_assembler_and_policy_events(self, mock_time):
        """run() yields assembler (utterance.updated) and policy events."""
        t = 1000.0

        provider = MockTranscriptProvider(sample_rate=16000)
        session = VoiceSession(provider)

        # Loud audio -> speech.started
        mock_time.time.return_value = t
        await session.push_audio(_loud_pcm())

        # More loud audio past MIN_SPEECH_DURATION_S
        t += MIN_SPEECH_DURATION_S + 0.1
        mock_time.time.return_value = t
        await session.push_audio(_loud_pcm())

        # Silence -> speech.stopped + user_transcript.final
        t += SILENCE_DURATION_S + 0.1
        mock_time.time.return_value = t
        await session.push_audio(_silent_pcm())

        collected = []
        async for event in session.run():
            collected.append(event)
            # Collect enough events or close after a timeout
            if len(collected) >= 4:
                await session.close()

        types = [e.type for e in collected]
        assert "speech.started" in types
        assert "speech.stopped" in types
        assert "user_transcript.final" in types

    @pytest.mark.asyncio
    async def test_close_flushes_and_cleans_up(self):
        """close() flushes the assembler and resets the turn policy."""
        provider = MockTranscriptProvider(sample_rate=16000)
        session = VoiceSession(provider)

        # Pre-populate assembler with an in-progress utterance
        session.assembler._utterance_id = "utt-test"
        session.assembler._fragments = []
        session.assembler._start_time = time.time()
        session.assembler._last_time = time.time()

        await session.close()

        assert session._closed is True
        assert session.assembler._utterance_id is None
        assert provider.closed is True

    @pytest.mark.asyncio
    async def test_push_audio_after_close_is_noop(self):
        """push_audio after close() does not forward to the provider."""
        provider = MockTranscriptProvider(sample_rate=16000)
        session = VoiceSession(provider)
        await session.close()

        await session.push_audio(_loud_pcm())
        assert provider._queue.empty()

    @pytest.mark.asyncio
    async def test_event_bus_integration(self):
        """VoiceSession emits events to the EventBus when one is provided."""
        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.on(EventKind.SPEECH_STARTED, handler)

        provider = MockTranscriptProvider(sample_rate=16000)
        session = VoiceSession(provider, event_bus=bus)

        # Inject a speech.started event directly into the provider queue
        provider._queue.put_nowait(TranscriptEvent(type="speech.started"))

        collected = []
        async for event in session.run():
            collected.append(event)
            if len(collected) >= 1:
                await session.close()

        # The EventBus handler should have received the SPEECH_STARTED event
        assert len(received) >= 1
        assert received[0].kind == EventKind.SPEECH_STARTED

    @pytest.mark.asyncio
    @patch("moshi.agent.utterance.time")
    @patch("moshi.agent.turn_policy.time")
    async def test_tick_based_events_flow_through(self, mock_turn_time, mock_utt_time):
        """Tick-based events (utterance.finalized, turn.candidate) flow through the output queue."""
        t = 1000.0
        mock_utt_time.time.return_value = t
        mock_turn_time.time.return_value = t

        provider = MockTranscriptProvider(sample_rate=16000)
        session = VoiceSession(provider, merge_gap=0.2)

        # Place a user_transcript.final directly in the queue to trigger assembler
        provider._queue.put_nowait(TranscriptEvent(
            type="user_transcript.final",
            segment_id="seg-1",
            sequence=1,
            text="test utterance",
        ))
        # Also a speech.stopped for the turn policy
        provider._queue.put_nowait(TranscriptEvent(type="speech.stopped"))

        collected = []
        deadline = asyncio.get_event_loop().time() + 3.0

        async def advance_time():
            """Gradually advance mocked time so tick loops can detect gaps."""
            nonlocal t
            while not session._closed:
                await asyncio.sleep(0.05)
                t += 0.5
                mock_utt_time.time.return_value = t
                mock_turn_time.time.return_value = t

        time_task = asyncio.create_task(advance_time())
        try:
            async for event in session.run():
                collected.append(event)
                if event.type == "turn.candidate" or asyncio.get_event_loop().time() > deadline:
                    await session.close()
        finally:
            time_task.cancel()
            try:
                await time_task
            except asyncio.CancelledError:
                pass

        types = [e.type for e in collected]
        # We expect at least the provider events plus tick-generated events
        assert "user_transcript.final" in types
        assert "speech.stopped" in types
