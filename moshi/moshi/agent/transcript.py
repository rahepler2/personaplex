# SPDX-License-Identifier: MIT
"""
Pluggable ASR transcript capture.

Phase 1A: Fan out improved PCM to pluggable TranscriptProvider (ASR)
interfaces with PersonaPlex's native audio processing pipeline.
Emits speech/turn events.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncIterator, Callable, Optional

import numpy as np

from .events import Event, EventBus, EventKind

logger = logging.getLogger(__name__)


class SessionPhase(Enum):
    IDLE = auto()
    LISTENING = auto()
    USER_SPEAKING = auto()
    CLOSED = auto()


@dataclass
class TranscriptEvent:
    """A single ASR event from a TranscriptProvider."""
    type: str
    segment_id: Optional[str] = None
    sequence: Optional[int] = None
    text: Optional[str] = None
    start_ms: Optional[float] = None
    end_ms: Optional[float] = None
    confidence: Optional[float] = None
    language: Optional[str] = None
    duration_ms: Optional[float] = None
    provider: Optional[str] = None
    utterance_id: Optional[str] = None
    segment_ids: Optional[list[str]] = None
    turn_id: Optional[str] = None


# -----------------------------------------------------------------------
# TranscriptProvider protocol
# -----------------------------------------------------------------------

class TranscriptProvider:
    """Base class for ASR transcript providers.

    This does NOT call the LLM directly. It emits TranscriptEvents;
    Phase 2's event controller decides whether to accept, defer,
    or reroute.
    """

    async def push_audio(self, pcm: np.ndarray) -> None:
        raise NotImplementedError

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        raise NotImplementedError
        yield  # type: ignore

    async def close(self) -> None:
        pass


# -----------------------------------------------------------------------
# VAD (shared energy-based voice activity detection)
# -----------------------------------------------------------------------

ENERGY_THRESHOLD = 0.01
SILENCE_DURATION_S = 0.8
MIN_SPEECH_DURATION_S = 0.3

MAX_QUEUE_SECONDS = 10.0


class _SequenceCounter:
    """Monotonic counter for ASR segment ordering."""
    def __init__(self):
        self.value = 0

    def next(self) -> int:
        self.value += 1
        return self.value


class _VAD:
    """Energy-based VAD shared between TranscriptProvider instances."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.is_speaking = False
        self.speech_start_time = 0.0
        self.silence_start_time = 0.0
        self.speech_samples = 0

    def process(self, pcm: np.ndarray) -> list[TranscriptEvent]:
        """Process a PCM chunk. Returns 'speech.started', 'speech.stopped', or empty list."""
        energy = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
        now = time.time()

        events: list[TranscriptEvent] = []

        if energy > ENERGY_THRESHOLD:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = now
                self.silence_start_time = now
                self.speech_samples = 0
                events.append(TranscriptEvent(type="speech.started"))
            self.silence_start_time = now
            self.speech_samples += len(pcm)
        elif self.is_speaking:
            if now - self.silence_start_time > SILENCE_DURATION_S:
                duration = now - self.speech_start_time
                if duration >= MIN_SPEECH_DURATION_S:
                    events.append(TranscriptEvent(
                        type="speech.stopped",
                        duration_ms=duration * 1000,
                    ))
                self.is_speaking = False
                self.speech_samples = 0

        return events

    @property
    def speech_duration(self) -> float:
        if not self.is_speaking:
            return 0.0
        return (time.time() - self.speech_start_time) * 1000


# -----------------------------------------------------------------------
# TranscriptProvider implementations
# -----------------------------------------------------------------------

class MockTranscriptProvider(TranscriptProvider):
    """Energy-based VAD placeholder that emits fixed transcript events.

    This does NOT call the LLM directly. It emits TranscriptEvents;
    Phase 2's event controller decides whether to accept, defer,
    or reroute.
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.session_id = str(uuid.uuid4())
        self._queue: asyncio.Queue[TranscriptEvent] = asyncio.Queue(
            maxsize=int(MAX_QUEUE_SECONDS * sample_rate / 1920 + 1),
        )
        self._vad = _VAD(sample_rate)
        self._seq = _SequenceCounter()
        self._event_bus: Optional[EventBus] = None
        self.closed = False
        self._worker_task: Optional[asyncio.Task] = None

    def _ensure_audio(self):
        if self._queue is None:
            self._queue = asyncio.Queue(
                maxsize=int(MAX_QUEUE_SECONDS * self.sample_rate / 1920 + 1),
            )

    async def push_audio(self, pcm: np.ndarray) -> None:
        """Accept a PCM chunk, run VAD, accumulate speech segments, dispatch transcript events."""
        if self.closed:
            return

        vad_events = self._vad.process(pcm)
        for ev in vad_events:
            self._put_nowait(ev)

            if isinstance(ev, TranscriptEvent) and ev.type == "speech.started":
                logger.info(f"MOCK ASR: speech started in session {self.session_id}")

            if ev.type == "speech.stopped":
                speech_duration = ev.duration_ms or 0.0
                seq = self._seq.next()
                segment_id = str(uuid.uuid4())
                text = "[speech detected]"

                self._put_nowait(TranscriptEvent(
                    type="user_transcript.final",
                    segment_id=segment_id,
                    sequence=seq,
                    text=text,
                    start_ms=None,
                    end_ms=None,
                    confidence=None,
                    language=None,
                    provider="mock",
                ))
                logger.info(f"MOCK ASR: speech segment finalized (duration={speech_duration:.0f}ms)")

    def _put_nowait(self, event: TranscriptEvent):
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(event)

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        """Yield transcript events. Runs until close() is called."""
        while not self.closed:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue

    async def close(self) -> None:
        self.closed = True
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def to_status_dict(self) -> dict:
        return {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "phase": "LISTENING" if not self.closed else "CLOSED",
            "stats": len(self._queue) if self._queue else 0,
            "speech_duration_ms": self._vad.speech_duration,
            "speech_active": self._vad.is_speaking,
        }


class WhisperTranscriptProvider(TranscriptProvider):
    """OpenAI Whisper-based ASR via local inference.

    Transcribes audio via Whisper, uses energy-based VAD to detect
    speech intervals, dispatches Whisper transcript events with
    actual text (not placeholder). The Whisper task runs in a
    background thread to avoid blocking the event loop.

    NOTE: Requires ``openai-whisper`` package (``pip install openai-whisper``).
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.session_id = str(uuid.uuid4())
        self._queue: asyncio.Queue[TranscriptEvent] = asyncio.Queue(
            maxsize=int(MAX_QUEUE_SECONDS * sample_rate / 1920 + 1),
        )
        self._vad = _VAD(sample_rate)
        self._seq = _SequenceCounter()
        self._event_bus: Optional[EventBus] = None
        self.closed = False
        self._worker_task: Optional[asyncio.Task] = None
        self._speech_buffer: list[np.ndarray] = []
        self._model = None
        self._target_sr = 16000

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            import whisper  # type: ignore
            self._model = whisper.load_model("base")
            logger.info("Whisper ASR: loaded 'base' model")
        except ImportError:
            logger.warning(
                "openai-whisper not installed. Install: pip install openai-whisper"
            )

    async def push_audio(self, pcm: np.ndarray) -> None:
        """Accept PCM, run VAD, accumulate and transcribe on speech end."""
        if self.closed:
            return

        vad_events = self._vad.process(pcm)
        for ev in vad_events:
            self._put_nowait(ev)

            if ev.type == "speech.started":
                self._speech_buffer = []
                logger.info(f"Whisper ASR: speech started in session {self.session_id}")

            if ev.type == "speech.stopped":
                if self._speech_buffer:
                    audio_data = np.concatenate(self._speech_buffer)
                    self._speech_buffer = []
                    seq = self._seq.next()
                    segment_id = str(uuid.uuid4())
                    await self._transcribe_and_emit(audio_data, seq, segment_id, ev.duration_ms)

        if self._vad.is_speaking:
            self._speech_buffer.append(pcm.copy())

    async def _transcribe_and_emit(
        self,
        audio: np.ndarray,
        seq: int,
        segment_id: str,
        duration_ms: Optional[float],
    ) -> None:
        self._ensure_model()
        if self._model is None:
            self._put_nowait(TranscriptEvent(
                type="user_transcript.final",
                segment_id=segment_id,
                sequence=seq,
                text="[whisper not available]",
                provider="whisper",
            ))
            return

        try:
            audio_f32 = audio.astype(np.float32)
            if audio_f32.max() > 1.0:
                audio_f32 = audio_f32 / 32768.0
            # Resample to 16kHz if needed
            if self.sample_rate != self._target_sr:
                try:
                    import librosa  # type: ignore
                    audio_f32 = librosa.resample(
                        audio_f32, orig_sr=self.sample_rate, target_sr=self._target_sr,
                    )
                except ImportError:
                    ratio = self._target_sr / self.sample_rate
                    indices = np.arange(0, len(audio_f32), 1 / ratio).astype(int)
                    indices = indices[indices < len(audio_f32)]
                    audio_f32 = audio_f32[indices]

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(
                    audio_f32, fp16=False
                ),
            )
            text = result.get("text", "").strip()
            avg_confidence = None
            segments = result.get("segments", [])
            if segments:
                avg_confidence = sum(s.get("avg_logprob", 0) for s in segments) / len(segments)

            self._put_nowait(TranscriptEvent(
                type="user_transcript.final",
                segment_id=segment_id,
                sequence=seq,
                text=text,
                confidence=avg_confidence,
                language=result.get("language"),
                provider="whisper",
                duration_ms=duration_ms,
            ))
            logger.info(
                f"Whisper ASR: transcribed segment {segment_id}: \"{text[:80]}\" "
                f"(duration={duration_ms}ms)"
            )
        except Exception as e:
            logger.error(f"Whisper ASR transcription error: {e}")
            self._put_nowait(TranscriptEvent(
                type="user_transcript.final",
                segment_id=segment_id,
                sequence=seq,
                text=f"[transcription error: {e}]",
                provider="whisper",
            ))

    def _put_nowait(self, event: TranscriptEvent):
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(event)
            if hasattr(self, '_drops_logged'):
                self._drops_logged += 1
            else:
                self._drops_logged = 1
                if self._drops_logged % 50 == 1:
                    logger.warning(
                        f"Whisper ASR: event queue overflow, "
                        f"{self._drops_logged} drops total"
                    )

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        """Yield transcript events."""
        while not self.closed:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue

    async def close(self) -> None:
        self.closed = True
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
