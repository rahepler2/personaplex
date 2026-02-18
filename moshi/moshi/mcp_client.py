# SPDX-License-Identifier: MIT

"""
MCP (Model Context Protocol) client for PersonaPlex.

Connects to MCP servers that provide knowledge retrieval tools,
enabling the persona to fetch relevant information during conversations.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"  # "stdio" or "sse"
    url: Optional[str] = None  # For SSE transport


@dataclass
class MCPTool:
    """Represents a tool available from an MCP server."""
    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


@dataclass
class MCPToolResult:
    """Result from calling an MCP tool."""
    content: str
    is_error: bool = False


class MCPClient:
    """Client that manages connections to MCP servers and provides tool calling.

    Supports both stdio and SSE transports. Each MCP server can expose
    tools like 'search', 'retrieve', 'query' that the RAG system uses
    to fetch knowledge.
    """

    def __init__(self):
        self._servers: dict[str, MCPServerConfig] = {}
        self._sessions: dict[str, Any] = {}
        self._transports: dict[str, Any] = {}
        self._tools: dict[str, MCPTool] = {}
        self._running = False

    def add_server(self, config: MCPServerConfig):
        self._servers[config.name] = config

    async def connect_all(self):
        """Connect to all configured MCP servers."""
        for name, config in self._servers.items():
            try:
                await self._connect_server(config)
                logger.info(f"Connected to MCP server: {name}")
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {name}: {e}")
        self._running = True

    async def _connect_server(self, config: MCPServerConfig):
        """Connect to a single MCP server and discover its tools."""
        try:
            from mcp import ClientSession
            from mcp.client.stdio import stdio_client, StdioServerParameters
        except ImportError:
            logger.warning(
                "MCP SDK not installed. Install with: pip install mcp"
            )
            return

        if config.transport == "stdio":
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env if config.env else None,
            )
            transport = await stdio_client(server_params).__aenter__()
            read_stream, write_stream = transport
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            await session.initialize()

            self._sessions[config.name] = session
            self._transports[config.name] = transport

            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                tool_key = f"{config.name}:{tool.name}"
                self._tools[tool_key] = MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                    server_name=config.name,
                )
                logger.info(f"  Discovered tool: {tool.name} - {tool.description}")

        elif config.transport == "sse":
            if not config.url:
                raise ValueError(f"SSE transport requires a URL for server {config.name}")
            try:
                from mcp.client.sse import sse_client
            except ImportError:
                logger.warning("MCP SSE client not available")
                return

            transport = await sse_client(config.url).__aenter__()
            read_stream, write_stream = transport
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            await session.initialize()

            self._sessions[config.name] = session
            self._transports[config.name] = transport

            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                tool_key = f"{config.name}:{tool.name}"
                self._tools[tool_key] = MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                    server_name=config.name,
                )
                logger.info(f"  Discovered tool: {tool.name} - {tool.description}")

    async def call_tool(self, server_name: str, tool_name: str,
                        arguments: dict[str, Any]) -> MCPToolResult:
        """Call a tool on a specific MCP server."""
        session = self._sessions.get(server_name)
        if session is None:
            return MCPToolResult(
                content=f"MCP server '{server_name}' not connected",
                is_error=True,
            )

        try:
            result = await session.call_tool(tool_name, arguments)
            content_parts = []
            for content in result.content:
                if hasattr(content, 'text'):
                    content_parts.append(content.text)
                elif hasattr(content, 'data'):
                    content_parts.append(str(content.data))
            return MCPToolResult(
                content="\n".join(content_parts),
                is_error=result.isError if hasattr(result, 'isError') else False,
            )
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} on {server_name}: {e}")
            return MCPToolResult(content=str(e), is_error=True)

    async def search(self, query: str) -> list[MCPToolResult]:
        """Search across all connected MCP servers for relevant information.

        Looks for tools named 'search', 'retrieve', 'query', or 'lookup'
        and calls them with the query.
        """
        results = []
        search_tool_names = {"search", "retrieve", "query", "lookup",
                             "semantic_search", "vector_search", "find"}

        for tool_key, tool in self._tools.items():
            if tool.name.lower() in search_tool_names:
                arguments = self._build_search_arguments(tool, query)
                result = await self.call_tool(tool.server_name, tool.name, arguments)
                if not result.is_error and result.content.strip():
                    results.append(result)

        return results

    def _build_search_arguments(self, tool: MCPTool, query: str) -> dict[str, Any]:
        """Build tool arguments based on the tool's input schema."""
        schema = tool.input_schema
        properties = schema.get("properties", {})

        arguments: dict[str, Any] = {}
        for prop_name, prop_schema in properties.items():
            lower_name = prop_name.lower()
            if lower_name in ("query", "q", "search", "text", "question",
                              "input", "prompt", "search_query"):
                arguments[prop_name] = query
                break

        if not arguments:
            first_string_prop = None
            for prop_name, prop_schema in properties.items():
                if prop_schema.get("type") == "string":
                    first_string_prop = prop_name
                    break
            if first_string_prop:
                arguments[first_string_prop] = query

        return arguments

    @property
    def available_tools(self) -> list[MCPTool]:
        return list(self._tools.values())

    @property
    def is_connected(self) -> bool:
        return self._running and len(self._sessions) > 0

    @property
    def connected_servers(self) -> list[str]:
        return list(self._sessions.keys())

    def server_status(self, name: str) -> dict:
        """Get status info for a specific server."""
        is_connected = name in self._sessions
        tools = [t for t in self._tools.values() if t.server_name == name]
        return {
            "name": name,
            "connected": is_connected,
            "tools": [
                {"name": t.name, "description": t.description}
                for t in tools
            ],
        }

    def all_server_statuses(self) -> list[dict]:
        """Get status for all configured servers."""
        all_names = set(self._servers.keys()) | set(self._sessions.keys())
        return [self.server_status(name) for name in sorted(all_names)]

    async def connect_server_by_name(self, name: str) -> dict:
        """Connect (or reconnect) a single server by name. Returns status."""
        config = self._servers.get(name)
        if config is None:
            return {"name": name, "connected": False, "error": "Server not configured"}
        # Disconnect first if already connected
        await self.disconnect_server(name)
        try:
            await self._connect_server(config)
            self._running = True
            return self.server_status(name)
        except Exception as e:
            logger.error(f"Failed to connect to {name}: {e}")
            return {"name": name, "connected": False, "error": str(e)}

    async def disconnect_server(self, name: str):
        """Disconnect a single server."""
        session = self._sessions.pop(name, None)
        transport = self._transports.pop(name, None)
        # Remove tools from this server
        keys_to_remove = [k for k, t in self._tools.items() if t.server_name == name]
        for k in keys_to_remove:
            del self._tools[k]
        if session:
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
        if transport and hasattr(transport, '__aexit__'):
            try:
                await transport.__aexit__(None, None, None)
            except Exception:
                pass

    async def remove_server(self, name: str):
        """Remove a server entirely (disconnect + remove config)."""
        await self.disconnect_server(name)
        self._servers.pop(name, None)

    async def test_connection(self, config: MCPServerConfig) -> dict:
        """Test a connection to an MCP server without persisting it.

        Returns dict with 'success', 'tools', and optionally 'error'.
        """
        try:
            from mcp import ClientSession
        except ImportError:
            return {"success": False, "error": "MCP SDK not installed", "tools": []}

        temp_name = f"__test_{config.name}"
        temp_config = MCPServerConfig(
            name=temp_name,
            command=config.command or "",
            args=config.args,
            env=config.env,
            transport=config.transport,
            url=config.url,
        )
        try:
            await self._connect_server(temp_config)
            tools = [
                {"name": t.name, "description": t.description}
                for t in self._tools.values()
                if t.server_name == temp_name
            ]
            # Clean up test connection
            await self.disconnect_server(temp_name)
            return {"success": True, "tools": tools}
        except Exception as e:
            await self.disconnect_server(temp_name)
            return {"success": False, "error": str(e), "tools": []}

    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        for name, session in self._sessions.items():
            try:
                await session.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error disconnecting from {name}: {e}")
        for name, transport in self._transports.items():
            try:
                if hasattr(transport, '__aexit__'):
                    await transport.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing transport for {name}: {e}")
        self._sessions.clear()
        self._transports.clear()
        self._tools.clear()
        self._running = False


def load_mcp_config(config_path: str) -> list[MCPServerConfig]:
    """Load MCP server configurations from a JSON file.

    Expected format:
    {
        "mcpServers": {
            "server-name": {
                "command": "path/to/server",
                "args": ["--arg1", "value1"],
                "env": {"KEY": "VALUE"},
                "transport": "stdio"
            },
            "sse-server": {
                "transport": "sse",
                "url": "http://localhost:8080/sse"
            }
        }
    }
    """
    with open(config_path, "r") as f:
        data = json.load(f)

    configs = []
    servers = data.get("mcpServers", {})
    for name, server_data in servers.items():
        configs.append(MCPServerConfig(
            name=name,
            command=server_data.get("command", ""),
            args=server_data.get("args", []),
            env=server_data.get("env", {}),
            transport=server_data.get("transport", "stdio"),
            url=server_data.get("url"),
        ))
    return configs
