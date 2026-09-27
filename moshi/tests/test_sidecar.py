# SPDX-License-Identifier: MIT
"""Tests for the sidecar LLM client and conversation orchestrator."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from moshi.agent.sidecar import (
    SidecarConfig,
    SidecarLLM,
    SidecarProvider,
    SidecarResponse,
    ToolCall,
    ToolResult,
)
from moshi.agent.orchestrator import (
    ConversationOrchestrator,
    OrchestratorConfig,
    OrchestratorEvent,
)
from moshi.agent.event_controller import AcceptedTurn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _anthropic_text_response(text: str) -> dict:
    """Simulate an Anthropic Messages API response with text only."""
    return {
        "content": [{"type": "text", "text": text}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
    }


def _anthropic_tool_response(text: str, tool_name: str, tool_id: str, args: dict) -> dict:
    """Simulate an Anthropic response with a tool use block."""
    return {
        "content": [
            {"type": "text", "text": text},
            {"type": "tool_use", "id": tool_id, "name": tool_name, "input": args},
        ],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "tool_use",
    }


def _openai_text_response(text: str) -> dict:
    """Simulate an OpenAI chat completions response."""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": text,
            },
            "finish_reason": "stop",
        }],
    }


def _openai_tool_response(text: str, tool_name: str, tool_id: str, args: dict) -> dict:
    """Simulate an OpenAI response with tool calls."""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": text,
                "tool_calls": [{
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }


def _make_turn(text: str = "What is the weather?", turn_id: str = "t1") -> AcceptedTurn:
    return AcceptedTurn(turn_id=turn_id, text=text, timestamp=time.time())


def _make_mcp_client(tools=None, call_result=None):
    """Create a mock MCP client with optional tools and call results."""
    mock = MagicMock()
    mock.is_connected = True
    if tools is None:
        tools = []
    mock.available_tools = tools
    if call_result is not None:
        mock.call_tool = AsyncMock(return_value=call_result)
    else:
        mock.call_tool = AsyncMock(return_value=MagicMock(content="result", is_error=False))
    return mock


def _make_mcp_tool(name="search", description="Search docs", server_name="docs"):
    mock = MagicMock()
    mock.name = name
    mock.description = description
    mock.input_schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    mock.server_name = server_name
    return mock


# ===================================================================
# 1. SidecarConfig
# ===================================================================


class TestSidecarConfig:

    def test_anthropic_defaults(self):
        config = SidecarConfig(api_key="sk-test")
        assert config.provider == SidecarProvider.ANTHROPIC
        assert config.model == "claude-sonnet-4-20250514"
        assert config.api_base_url == "https://api.anthropic.com"

    def test_openai_defaults(self):
        config = SidecarConfig(provider=SidecarProvider.OPENAI, api_key="sk-test")
        assert config.model == "gpt-4o"
        assert config.api_base_url == "https://api.openai.com"

    def test_env_var_anthropic(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            config = SidecarConfig()
            assert config.api_key == "env-key"

    def test_env_var_openai(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            config = SidecarConfig(provider=SidecarProvider.OPENAI)
            assert config.api_key == "env-key"

    def test_explicit_overrides(self):
        config = SidecarConfig(
            api_key="my-key",
            model="claude-opus-4-20250514",
            api_base_url="https://custom.api.com",
            max_tokens=2048,
        )
        assert config.api_key == "my-key"
        assert config.model == "claude-opus-4-20250514"
        assert config.api_base_url == "https://custom.api.com"
        assert config.max_tokens == 2048


# ===================================================================
# 2. SidecarLLM response parsing
# ===================================================================


class TestSidecarResponseParsing:

    def test_parse_anthropic_text_only(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))
        raw = _anthropic_text_response("Hello there!")
        text, tool_calls = sidecar.parse_response(raw)
        assert text == "Hello there!"
        assert tool_calls == []

    def test_parse_anthropic_with_tool_use(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))
        raw = _anthropic_tool_response(
            "Let me search for that.",
            "search",
            "toolu_123",
            {"query": "weather"},
        )
        text, tool_calls = sidecar.parse_response(raw)
        assert text == "Let me search for that."
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "toolu_123"
        assert tool_calls[0].name == "search"
        assert tool_calls[0].arguments == {"query": "weather"}

    def test_parse_openai_text_only(self):
        sidecar = SidecarLLM(SidecarConfig(
            provider=SidecarProvider.OPENAI, api_key="test",
        ))
        raw = _openai_text_response("Hello!")
        text, tool_calls = sidecar.parse_response(raw)
        assert text == "Hello!"
        assert tool_calls == []

    def test_parse_openai_with_tool_calls(self):
        sidecar = SidecarLLM(SidecarConfig(
            provider=SidecarProvider.OPENAI, api_key="test",
        ))
        raw = _openai_tool_response(
            "",
            "search",
            "call_abc",
            {"query": "weather"},
        )
        text, tool_calls = sidecar.parse_response(raw)
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_abc"
        assert tool_calls[0].name == "search"
        assert tool_calls[0].arguments == {"query": "weather"}


# ===================================================================
# 3. Tool result formatting
# ===================================================================


class TestToolResultFormatting:

    def test_anthropic_tool_result(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))
        msg = sidecar.format_tool_result_message("toolu_123", "sunny, 72F")
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["tool_use_id"] == "toolu_123"
        assert msg["content"][0]["content"] == "sunny, 72F"

    def test_anthropic_tool_result_error(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))
        msg = sidecar.format_tool_result_message("toolu_123", "not found", is_error=True)
        assert msg["content"][0]["is_error"] is True

    def test_openai_tool_result(self):
        sidecar = SidecarLLM(SidecarConfig(
            provider=SidecarProvider.OPENAI, api_key="test",
        ))
        msg = sidecar.format_tool_result_message("call_abc", "sunny, 72F")
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_abc"
        assert msg["content"] == "sunny, 72F"


# ===================================================================
# 4. Assistant message formatting
# ===================================================================


class TestAssistantMessageFormatting:

    def test_anthropic_assistant_message(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))
        raw = _anthropic_text_response("Hello")
        msg = sidecar.format_assistant_message(raw)
        assert msg["role"] == "assistant"
        assert msg["content"] == [{"type": "text", "text": "Hello"}]

    def test_openai_assistant_message(self):
        sidecar = SidecarLLM(SidecarConfig(
            provider=SidecarProvider.OPENAI, api_key="test",
        ))
        raw = _openai_text_response("Hello")
        msg = sidecar.format_assistant_message(raw)
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello"


# ===================================================================
# 5. SidecarLLM.is_configured
# ===================================================================


class TestSidecarIsConfigured:

    def test_configured_with_key(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))
        assert sidecar.is_configured is True

    def test_not_configured_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            sidecar = SidecarLLM(SidecarConfig(api_key=None))
            assert sidecar.is_configured is False


# ===================================================================
# 6. ConversationOrchestrator — text-only turn (no tool calls)
# ===================================================================


class TestOrchestratorTextOnly:

    @pytest.mark.asyncio
    async def test_process_turn_text_only(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))
        sidecar.chat = AsyncMock(return_value=_anthropic_text_response("It's sunny!"))

        mcp = _make_mcp_client()
        events_received = []

        async def on_event(ev):
            events_received.append(ev)

        orch = ConversationOrchestrator(
            sidecar=sidecar,
            mcp_client=mcp,
            on_event=on_event,
        )

        turn = _make_turn("What's the weather?")
        response = await orch.process_turn(turn)

        assert response.text == "It's sunny!"
        assert response.tool_calls == []
        assert response.turn_id == "t1"
        assert len(events_received) == 1
        assert events_received[0].type == "sidecar_response"

    @pytest.mark.asyncio
    async def test_conversation_history_maintained(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))
        sidecar.chat = AsyncMock(return_value=_anthropic_text_response("Response"))

        orch = ConversationOrchestrator(
            sidecar=sidecar,
            mcp_client=_make_mcp_client(),
        )

        await orch.process_turn(_make_turn("Hello", "t1"))
        await orch.process_turn(_make_turn("How are you?", "t2"))

        assert len(orch._history) == 4  # 2 user + 2 assistant
        assert orch._history[0] == {"role": "user", "content": "Hello"}
        assert orch._history[1] == {"role": "assistant", "content": "Response"}


# ===================================================================
# 7. ConversationOrchestrator — tool calling loop
# ===================================================================


class TestOrchestratorToolCalling:

    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))

        # First call returns tool use, second call returns final text
        sidecar.chat = AsyncMock(side_effect=[
            _anthropic_tool_response(
                "Searching...", "search", "toolu_1", {"query": "weather NYC"},
            ),
            _anthropic_text_response("It's 72F and sunny in NYC."),
        ])

        tool = _make_mcp_tool()
        mcp = _make_mcp_client(
            tools=[tool],
            call_result=MagicMock(content="NYC: 72F, sunny", is_error=False),
        )

        events = []
        orch = ConversationOrchestrator(
            sidecar=sidecar,
            mcp_client=mcp,
            on_event=AsyncMock(side_effect=lambda ev: events.append(ev)),
        )

        response = await orch.process_turn(_make_turn("Weather in NYC?"))

        assert response.text == "It's 72F and sunny in NYC."
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"
        assert len(response.tool_results) == 1
        assert response.tool_results[0].content == "NYC: 72F, sunny"
        mcp.call_tool.assert_called_once_with("docs", "search", {"query": "weather NYC"})

        event_types = [e.type for e in events]
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "sidecar_response" in event_types

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))
        sidecar.chat = AsyncMock(side_effect=[
            _anthropic_tool_response(
                "Let me check.", "unknown_tool", "toolu_1", {},
            ),
            _anthropic_text_response("I couldn't find that tool."),
        ])

        mcp = _make_mcp_client(tools=[])
        orch = ConversationOrchestrator(
            sidecar=sidecar,
            mcp_client=mcp,
        )

        response = await orch.process_turn(_make_turn("Do something"))
        assert len(response.tool_results) == 1
        assert response.tool_results[0].is_error is True
        assert "not found" in response.tool_results[0].content

    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))
        sidecar.chat = AsyncMock(side_effect=[
            _anthropic_tool_response(
                "Searching...", "search", "toolu_1", {"query": "test"},
            ),
            _anthropic_text_response("Sorry, the search failed."),
        ])

        tool = _make_mcp_tool()
        mcp = _make_mcp_client(tools=[tool])
        mcp.call_tool = AsyncMock(side_effect=RuntimeError("connection refused"))

        orch = ConversationOrchestrator(
            sidecar=sidecar,
            mcp_client=mcp,
        )

        response = await orch.process_turn(_make_turn("Search for something"))
        assert len(response.tool_results) == 1
        assert response.tool_results[0].is_error is True
        assert "connection refused" in response.tool_results[0].content


# ===================================================================
# 8. ConversationOrchestrator — RAG context injection
# ===================================================================


class TestOrchestratorRAGContext:

    @pytest.mark.asyncio
    async def test_rag_context_in_system_prompt(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))
        sidecar.chat = AsyncMock(return_value=_anthropic_text_response("Based on the docs..."))

        kb = MagicMock()
        kb.has_context = True
        kb.get_context_summary.return_value = "Python is a programming language."

        orch = ConversationOrchestrator(
            sidecar=sidecar,
            mcp_client=_make_mcp_client(),
            knowledge_base=kb,
            config=OrchestratorConfig(persona_prompt="You are a teacher."),
        )

        await orch.process_turn(_make_turn("Tell me about Python"))

        call_args = sidecar.chat.call_args
        system_prompt = call_args.kwargs.get("system", "")
        assert "You are a teacher." in system_prompt
        assert "Python is a programming language." in system_prompt

    @pytest.mark.asyncio
    async def test_no_rag_when_empty(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))
        sidecar.chat = AsyncMock(return_value=_anthropic_text_response("Hello"))

        kb = MagicMock()
        kb.has_context = False

        orch = ConversationOrchestrator(
            sidecar=sidecar,
            mcp_client=_make_mcp_client(),
            knowledge_base=kb,
        )

        await orch.process_turn(_make_turn("Hello"))
        call_args = sidecar.chat.call_args
        system_prompt = call_args.kwargs.get("system", "")
        assert "knowledge" not in system_prompt.lower() or "Relevant knowledge" not in system_prompt


# ===================================================================
# 9. ConversationOrchestrator — tool schema conversion
# ===================================================================


class TestOrchestratorToolSchemas:

    @pytest.mark.asyncio
    async def test_mcp_tools_passed_to_sidecar(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))
        sidecar.chat = AsyncMock(return_value=_anthropic_text_response("OK"))

        tool = _make_mcp_tool(name="search", description="Search docs")
        mcp = _make_mcp_client(tools=[tool])

        orch = ConversationOrchestrator(
            sidecar=sidecar,
            mcp_client=mcp,
        )

        await orch.process_turn(_make_turn("Search for X"))

        call_args = sidecar.chat.call_args
        tools_arg = call_args.kwargs.get("tools", [])
        assert len(tools_arg) == 1
        assert tools_arg[0]["name"] == "search"
        assert tools_arg[0]["description"] == "Search docs"

    @pytest.mark.asyncio
    async def test_no_tools_when_mcp_disconnected(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))
        sidecar.chat = AsyncMock(return_value=_anthropic_text_response("OK"))

        mcp = MagicMock()
        mcp.is_connected = False

        orch = ConversationOrchestrator(
            sidecar=sidecar,
            mcp_client=mcp,
        )

        await orch.process_turn(_make_turn("Hello"))
        call_args = sidecar.chat.call_args
        tools_arg = call_args.kwargs.get("tools", None)
        assert tools_arg is None


# ===================================================================
# 10. ConversationOrchestrator — error handling
# ===================================================================


class TestOrchestratorErrors:

    @pytest.mark.asyncio
    async def test_sidecar_api_error(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))
        sidecar.chat = AsyncMock(side_effect=RuntimeError("API error 500"))

        events = []
        orch = ConversationOrchestrator(
            sidecar=sidecar,
            mcp_client=_make_mcp_client(),
            on_event=AsyncMock(side_effect=lambda ev: events.append(ev)),
        )

        response = await orch.process_turn(_make_turn("Hello"))
        assert response.text == ""
        assert any(e.type == "error" for e in events)

    @pytest.mark.asyncio
    async def test_history_trimming(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))
        sidecar.chat = AsyncMock(return_value=_anthropic_text_response("OK"))

        orch = ConversationOrchestrator(
            sidecar=sidecar,
            mcp_client=_make_mcp_client(),
            config=OrchestratorConfig(max_history_turns=2),
        )

        for i in range(5):
            await orch.process_turn(_make_turn(f"Message {i}", f"t{i}"))

        # max_history_turns=2: trim keeps last 4, then adds assistant = 5
        # but next trim brings it back to 4 + assistant = 5 until trim runs again
        assert len(orch._history) <= 5

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        sidecar = SidecarLLM(SidecarConfig(api_key="test"))
        sidecar.chat = AsyncMock(return_value=_anthropic_text_response("OK"))

        orch = ConversationOrchestrator(
            sidecar=sidecar,
            mcp_client=_make_mcp_client(),
        )

        await orch.process_turn(_make_turn("Hello"))
        assert len(orch._history) > 0

        orch.reset()
        assert len(orch._history) == 0
        assert orch._processing is False
