# SPDX-License-Identifier: MIT

"""
RAG (Retrieval-Augmented Generation) knowledge base for PersonaPlex.

Manages the retrieval pipeline:
- Accumulates decoded user/agent text during conversation
- Detects when to trigger a knowledge retrieval query
- Queries MCP servers asynchronously for relevant context
- Provides retrieved context for system prompt enrichment
- Sends RAG results back through the WebSocket for display
"""

import asyncio
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable

from .mcp_client import MCPClient, MCPToolResult

logger = logging.getLogger(__name__)

# Characters that indicate a sentence boundary
SENTENCE_BOUNDARIES = {'.', '?', '!'}

# Minimum characters accumulated before triggering a RAG query
MIN_QUERY_LENGTH = 15

# Minimum seconds between RAG queries to avoid flooding
MIN_QUERY_INTERVAL = 3.0

# Maximum number of context chunks to keep
MAX_CONTEXT_CHUNKS = 10

# Maximum characters per context chunk
MAX_CHUNK_LENGTH = 500


@dataclass
class RAGContext:
    """A piece of retrieved context from the knowledge base."""
    content: str
    source: str
    query: str
    timestamp: float
    relevance_score: float = 0.0


@dataclass
class RAGConfig:
    """Configuration for the RAG system."""
    enabled: bool = True
    min_query_length: int = MIN_QUERY_LENGTH
    min_query_interval: float = MIN_QUERY_INTERVAL
    max_context_chunks: int = MAX_CONTEXT_CHUNKS
    max_chunk_length: int = MAX_CHUNK_LENGTH
    enrich_system_prompt: bool = True
    max_system_prompt_context: int = 3
    system_prompt_max_chars: int = 300


class KnowledgeBase:
    """Manages RAG retrieval for a PersonaPlex conversation.

    Integrates with MCP servers to fetch relevant knowledge based on
    what the user says during conversation. Works asynchronously so
    it doesn't block the real-time audio pipeline.

    Usage:
        kb = KnowledgeBase(mcp_client, config)

        # Enrich system prompt before conversation starts
        enriched_prompt = await kb.enrich_system_prompt("You are a teacher.")

        # During conversation, feed decoded text
        kb.add_text(decoded_text, source="user")

        # Retrieved context is sent via the callback
        kb = KnowledgeBase(mcp_client, config, on_context=send_to_client)
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        config: Optional[RAGConfig] = None,
        on_context: Optional[Callable[[RAGContext], Awaitable[None]]] = None,
    ):
        self._mcp = mcp_client
        self._config = config or RAGConfig()
        self._on_context = on_context

        # Text accumulation buffers
        self._user_text_buffer: list[str] = []
        self._agent_text_buffer: list[str] = []
        self._current_sentence: list[str] = []

        # RAG state
        self._context_store: deque[RAGContext] = deque(maxlen=self._config.max_context_chunks)
        self._last_query_time: float = 0.0
        self._pending_queries: int = 0
        self._query_task: Optional[asyncio.Task] = None

        # Conversation history for context
        self._conversation_turns: list[dict[str, str]] = []

    def add_text(self, text: str, source: str = "user"):
        """Add decoded text from the conversation.

        Called as the model decodes text tokens. Accumulates text
        and triggers RAG queries when sentence boundaries are detected.

        Args:
            text: Decoded text fragment (usually a single token/character)
            source: "user" or "agent"
        """
        if not self._config.enabled or not self._mcp.is_connected:
            return

        self._current_sentence.append(text)

        if source == "user":
            self._user_text_buffer.append(text)
        else:
            self._agent_text_buffer.append(text)

        # Check for sentence boundary
        stripped = text.strip()
        if stripped and stripped[-1] in SENTENCE_BOUNDARIES:
            sentence = "".join(self._current_sentence).strip()
            self._current_sentence = []

            if source == "user" and len(sentence) >= self._config.min_query_length:
                self._maybe_trigger_query(sentence)

    def _maybe_trigger_query(self, sentence: str):
        """Trigger a RAG query if enough time has passed."""
        now = time.monotonic()
        if now - self._last_query_time < self._config.min_query_interval:
            return
        if self._pending_queries > 0:
            return

        self._last_query_time = now
        self._pending_queries += 1

        # Build query from recent user context
        query = self._build_query(sentence)

        # Fire async query
        loop = asyncio.get_event_loop()
        self._query_task = loop.create_task(self._execute_query(query))

    def _build_query(self, trigger_sentence: str) -> str:
        """Build a search query from accumulated user text."""
        recent_user = "".join(self._user_text_buffer[-200:])
        if len(recent_user) > len(trigger_sentence) + 20:
            return recent_user.strip()
        return trigger_sentence

    async def _execute_query(self, query: str):
        """Execute a RAG query against MCP servers."""
        try:
            results = await self._mcp.search(query)
            for result in results:
                if result.is_error:
                    continue
                content = result.content[:self._config.max_chunk_length]
                context = RAGContext(
                    content=content,
                    source="mcp",
                    query=query,
                    timestamp=time.time(),
                )
                self._context_store.append(context)
                logger.info(f"RAG result for '{query[:50]}...': {content[:100]}...")

                if self._on_context:
                    await self._on_context(context)
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
        finally:
            self._pending_queries -= 1

    async def enrich_system_prompt(self, original_prompt: str) -> str:
        """Enrich the system prompt with relevant knowledge before conversation.

        Queries the knowledge base using the system prompt text itself
        to pre-fetch relevant context.

        Args:
            original_prompt: The original text prompt for the persona

        Returns:
            Enhanced prompt with retrieved context prepended
        """
        if not self._config.enrich_system_prompt:
            return original_prompt
        if not self._config.enabled:
            return original_prompt
        if not self._mcp.is_connected:
            return original_prompt

        try:
            results = await self._mcp.search(original_prompt)
            context_parts = []
            total_chars = 0
            for result in results[:self._config.max_system_prompt_context]:
                if result.is_error:
                    continue
                chunk = result.content[:self._config.system_prompt_max_chars]
                if total_chars + len(chunk) > self._config.system_prompt_max_chars:
                    break
                context_parts.append(chunk)
                total_chars += len(chunk)

            if context_parts:
                context_block = " ".join(context_parts)
                enriched = (
                    f"{original_prompt} "
                    f"Use the following knowledge when relevant: {context_block}"
                )
                logger.info(f"Enriched system prompt with {len(context_parts)} context chunks")
                return enriched

        except Exception as e:
            logger.error(f"Failed to enrich system prompt: {e}")

        return original_prompt

    def get_recent_context(self, max_items: int = 5) -> list[RAGContext]:
        """Get the most recent retrieved context items."""
        items = list(self._context_store)
        return items[-max_items:]

    def get_context_summary(self) -> str:
        """Get a text summary of all retrieved context."""
        if not self._context_store:
            return ""
        parts = [ctx.content for ctx in self._context_store]
        return " ".join(parts)

    def reset(self):
        """Reset state for a new conversation."""
        self._user_text_buffer.clear()
        self._agent_text_buffer.clear()
        self._current_sentence.clear()
        self._context_store.clear()
        self._last_query_time = 0.0
        self._pending_queries = 0
        self._conversation_turns.clear()
        if self._query_task and not self._query_task.done():
            self._query_task.cancel()

    @property
    def has_context(self) -> bool:
        return len(self._context_store) > 0

    @property
    def context_count(self) -> int:
        return len(self._context_store)
