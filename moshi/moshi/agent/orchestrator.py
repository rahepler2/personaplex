# SPDX-License-Identifier: MIT
"""
Conversation orchestrator for PersonaPlex.

Coordinates between the Moshi audio model, sidecar LLM, MCP tools,
and RAG knowledge base to provide full tool calling and context-aware
responses.  When a user turn is accepted the orchestrator:

1. Assembles context (conversation history + RAG)
2. Sends to sidecar LLM with available MCP tool schemas
3. Executes any requested tool calls via MCP
4. Returns the final response with all tool results
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional

from .sidecar import SidecarLLM, SidecarResponse, ToolCall, ToolResult
from .event_controller import AcceptedTurn

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the conversation orchestrator."""
    max_history_turns: int = 20
    max_rag_context_chars: int = 2000
    persona_prompt: str = ""


@dataclass
class OrchestratorEvent:
    """Event emitted by the orchestrator for the client."""
    type: str  # "tool_call", "tool_result", "sidecar_response", "error"
    data: dict[str, Any] = field(default_factory=dict)
    turn_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


OrchestratorCallback = Callable[[OrchestratorEvent], Awaitable[None]]


class ConversationOrchestrator:
    """Orchestrates tool calling and RAG for a PersonaPlex conversation.

    Sits between the voice pipeline (Moshi) and the external LLM.
    Receives accepted user turns, builds context from conversation
    history and RAG, sends to the sidecar LLM with MCP tool schemas,
    executes tool calls, and returns the final response.
    """

    def __init__(
        self,
        sidecar: SidecarLLM,
        mcp_client: Any,
        knowledge_base: Any = None,
        config: Optional[OrchestratorConfig] = None,
        on_event: Optional[OrchestratorCallback] = None,
    ):
        self._sidecar = sidecar
        self._mcp = mcp_client
        self._kb = knowledge_base
        self._config = config or OrchestratorConfig()
        self._on_event = on_event
        self._history: list[dict] = []
        self._processing = False

    @property
    def is_ready(self) -> bool:
        return self._sidecar.is_configured

    async def process_turn(self, turn: AcceptedTurn) -> SidecarResponse:
        """Process an accepted user turn through the sidecar LLM.

        This is the main entry point called when a turn is accepted.
        """
        self._processing = True
        try:
            system = self._build_system_prompt()
            self._history.append({"role": "user", "content": turn.text})
            self._trim_history()

            tools = self._get_tool_schemas()
            response = await self._tool_loop(system, tools, turn.turn_id)
            response.turn_id = turn.turn_id

            if response.text:
                self._history.append({
                    "role": "assistant",
                    "content": response.text,
                })

            await self._emit(OrchestratorEvent(
                type="sidecar_response",
                data={
                    "text": response.text,
                    "tool_calls_count": len(response.tool_calls),
                },
                turn_id=turn.turn_id,
            ))
            return response

        except Exception as e:
            logger.error(f"Orchestrator error processing turn: {e}")
            await self._emit(OrchestratorEvent(
                type="error",
                data={"error": str(e)},
                turn_id=turn.turn_id,
            ))
            return SidecarResponse(text="", turn_id=turn.turn_id)
        finally:
            self._processing = False

    async def _tool_loop(
        self,
        system: str,
        tools: list[dict],
        turn_id: Optional[str],
    ) -> SidecarResponse:
        """Run the sidecar LLM with multi-round tool calling."""
        all_tool_calls: list[ToolCall] = []
        all_tool_results: list[ToolResult] = []
        messages = list(self._history)

        for _ in range(self._sidecar._config.max_tool_rounds):
            raw = await self._sidecar.chat(
                messages=messages,
                system=system,
                tools=tools if tools else None,
            )
            text, tool_calls = self._sidecar.parse_response(raw)

            if not tool_calls:
                return SidecarResponse(
                    text=text,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    model=self._sidecar._config.model,
                )

            all_tool_calls.extend(tool_calls)
            messages.append(self._sidecar.format_assistant_message(raw))

            for tc in tool_calls:
                await self._emit(OrchestratorEvent(
                    type="tool_call",
                    data={
                        "tool": tc.name,
                        "arguments": tc.arguments,
                        "call_id": tc.id,
                    },
                    turn_id=turn_id,
                ))

                result = await self._execute_tool(tc)
                all_tool_results.append(result)

                await self._emit(OrchestratorEvent(
                    type="tool_result",
                    data={
                        "call_id": result.call_id,
                        "tool": result.tool_name,
                        "content": result.content[:500],
                        "is_error": result.is_error,
                    },
                    turn_id=turn_id,
                ))

                messages.append(self._sidecar.format_tool_result_message(
                    tc.id, result.content, result.is_error,
                ))

        raw = await self._sidecar.chat(messages=messages, system=system)
        text, _ = self._sidecar.parse_response(raw)

        return SidecarResponse(
            text=text,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            model=self._sidecar._config.model,
        )

    async def _execute_tool(self, tc: ToolCall) -> ToolResult:
        """Execute a tool call through the MCP client."""
        server_name = tc.server_name
        if not server_name:
            for tool in self._mcp.available_tools:
                if tool.name == tc.name:
                    server_name = tool.server_name
                    break

        if not server_name:
            return ToolResult(
                call_id=tc.id,
                tool_name=tc.name,
                content=f"Tool '{tc.name}' not found on any connected MCP server",
                is_error=True,
            )

        try:
            mcp_result = await self._mcp.call_tool(
                server_name, tc.name, tc.arguments,
            )
            return ToolResult(
                call_id=tc.id,
                tool_name=tc.name,
                content=mcp_result.content,
                is_error=mcp_result.is_error,
            )
        except Exception as e:
            logger.error(f"Tool execution error ({tc.name}): {e}")
            return ToolResult(
                call_id=tc.id,
                tool_name=tc.name,
                content=str(e),
                is_error=True,
            )

    def _build_system_prompt(self) -> str:
        """Build the sidecar's system prompt with persona + RAG context."""
        parts: list[str] = []

        if self._config.persona_prompt:
            parts.append(self._config.persona_prompt)

        parts.append(
            "You are the reasoning engine for a real-time voice assistant. "
            "The user is speaking through a voice interface. Keep responses "
            "concise and conversational. Use available tools when the user's "
            "request requires external information or actions."
        )

        if self._kb and self._kb.has_context:
            context = self._kb.get_context_summary()
            if context:
                trimmed = context[:self._config.max_rag_context_chars]
                parts.append(
                    f"Relevant knowledge retrieved during this conversation:\n"
                    f"{trimmed}"
                )

        return "\n\n".join(parts)

    def _get_tool_schemas(self) -> list[dict]:
        """Convert MCP tools to sidecar LLM tool format."""
        if not self._mcp or not self._mcp.is_connected:
            return []

        tools = []
        for mcp_tool in self._mcp.available_tools:
            tools.append({
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "input_schema": mcp_tool.input_schema,
            })
        return tools

    def _trim_history(self):
        max_messages = self._config.max_history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    async def _emit(self, event: OrchestratorEvent):
        if self._on_event:
            try:
                await self._on_event(event)
            except Exception as e:
                logger.error(f"Orchestrator event callback error: {e}")

    def reset(self):
        """Reset state for a new conversation."""
        self._history.clear()
        self._processing = False
