# SPDX-License-Identifier: MIT
"""
Sidecar LLM client for tool calling and RAG-augmented reasoning.

The Moshi model handles real-time audio generation but has no native
tool calling capability. This sidecar sends accepted user turns to an
external LLM (Claude or OpenAI) that can reason about tool calls and
incorporate RAG context into responses.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


class SidecarProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class SidecarConfig:
    """Configuration for the sidecar LLM."""
    provider: SidecarProvider = SidecarProvider.ANTHROPIC
    api_key: Optional[str] = None
    model: str = ""
    max_tokens: int = 1024
    temperature: float = 0.7
    api_base_url: Optional[str] = None
    max_tool_rounds: int = 5

    def __post_init__(self):
        if not self.api_key:
            if self.provider == SidecarProvider.ANTHROPIC:
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif self.provider == SidecarProvider.OPENAI:
                self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.model:
            if self.provider == SidecarProvider.ANTHROPIC:
                self.model = "claude-sonnet-4-20250514"
            elif self.provider == SidecarProvider.OPENAI:
                self.model = "gpt-4o"
        if not self.api_base_url:
            if self.provider == SidecarProvider.ANTHROPIC:
                self.api_base_url = "https://api.anthropic.com"
            elif self.provider == SidecarProvider.OPENAI:
                self.api_base_url = "https://api.openai.com"


@dataclass
class ToolCall:
    """A tool call requested by the sidecar LLM."""
    id: str
    name: str
    arguments: dict[str, Any]
    server_name: Optional[str] = None


@dataclass
class ToolResult:
    """Result from executing a tool call via MCP."""
    call_id: str
    tool_name: str
    content: str
    is_error: bool = False


@dataclass
class SidecarResponse:
    """Complete response from the sidecar LLM after tool calling."""
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    turn_id: Optional[str] = None
    model: Optional[str] = None


class SidecarLLM:
    """External LLM client for reasoning and tool calling.

    Sends user turns to Claude or OpenAI with MCP tool schemas,
    handles the multi-round tool calling loop, and returns the
    final text response with all tool results.
    """

    def __init__(self, config: SidecarConfig):
        self._config = config
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    @property
    def is_configured(self) -> bool:
        return bool(self._config.api_key)

    @property
    def provider(self) -> SidecarProvider:
        return self._config.provider

    async def chat(
        self,
        messages: list[dict],
        system: str = "",
        tools: Optional[list[dict]] = None,
    ) -> dict:
        """Send a chat request to the LLM API. Returns raw API response."""
        await self._ensure_session()

        if self._config.provider == SidecarProvider.ANTHROPIC:
            return await self._chat_anthropic(messages, system, tools)
        elif self._config.provider == SidecarProvider.OPENAI:
            return await self._chat_openai(messages, system, tools)
        raise ValueError(f"Unknown provider: {self._config.provider}")

    async def _chat_anthropic(
        self,
        messages: list[dict],
        system: str,
        tools: Optional[list[dict]],
    ) -> dict:
        url = f"{self._config.api_base_url}/v1/messages"
        headers = {
            "x-api-key": self._config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "messages": messages,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools
        if self._config.temperature is not None:
            body["temperature"] = self._config.temperature

        async with self._session.post(url, headers=headers, json=body) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(
                    f"Anthropic API error {resp.status}: {error_text[:500]}"
                )
            return await resp.json()

    async def _chat_openai(
        self,
        messages: list[dict],
        system: str,
        tools: Optional[list[dict]],
    ) -> dict:
        url = f"{self._config.api_base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "content-type": "application/json",
        }
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        body: dict[str, Any] = {
            "model": self._config.model,
            "messages": all_messages,
            "max_tokens": self._config.max_tokens,
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    },
                }
                for t in tools
            ]
        if self._config.temperature is not None:
            body["temperature"] = self._config.temperature

        async with self._session.post(url, headers=headers, json=body) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(
                    f"OpenAI API error {resp.status}: {error_text[:500]}"
                )
            return await resp.json()

    def parse_response(self, raw: dict) -> tuple[str, list[ToolCall]]:
        """Parse API response into text and tool calls."""
        if self._config.provider == SidecarProvider.ANTHROPIC:
            return self._parse_anthropic(raw)
        elif self._config.provider == SidecarProvider.OPENAI:
            return self._parse_openai(raw)
        raise ValueError(f"Unknown provider: {self._config.provider}")

    def _parse_anthropic(self, raw: dict) -> tuple[str, list[ToolCall]]:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in raw.get("content", []):
            if block["type"] == "text":
                text_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block.get("input", {}),
                ))
        return " ".join(text_parts), tool_calls

    def _parse_openai(self, raw: dict) -> tuple[str, list[ToolCall]]:
        choice = raw["choices"][0]
        msg = choice["message"]
        text = msg.get("content") or ""
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls", []):
            args = tc["function"].get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=args,
            ))
        return text, tool_calls

    def format_tool_result_message(
        self,
        call_id: str,
        content: str,
        is_error: bool = False,
    ) -> dict:
        """Format a tool result as a conversation message."""
        if self._config.provider == SidecarProvider.ANTHROPIC:
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": content,
            }
            if is_error:
                block["is_error"] = True
            return {"role": "user", "content": [block]}
        elif self._config.provider == SidecarProvider.OPENAI:
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            }
        raise ValueError(f"Unknown provider: {self._config.provider}")

    def format_assistant_message(self, raw: dict) -> dict:
        """Extract the assistant message from a raw API response."""
        if self._config.provider == SidecarProvider.ANTHROPIC:
            return {"role": "assistant", "content": raw.get("content", [])}
        elif self._config.provider == SidecarProvider.OPENAI:
            return raw["choices"][0]["message"]
        raise ValueError(f"Unknown provider: {self._config.provider}")
