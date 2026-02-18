# SPDX-License-Identifier: MIT

"""
Admin REST API for PersonaPlex.

Provides endpoints for managing personas and MCP server connections
through the web UI.
"""

import json
import logging
from typing import Optional

from aiohttp import web

from .admin_store import AdminStore
from .mcp_client import MCPClient, MCPServerConfig

logger = logging.getLogger(__name__)


def register_admin_routes(
    app: web.Application,
    store: AdminStore,
    mcp_client: MCPClient,
):
    """Register all admin API routes on the aiohttp application."""

    # --- Persona endpoints ---

    async def list_personas(_request):
        personas = store.list_personas()
        return web.json_response([p.to_dict() for p in personas])

    async def get_persona(request):
        pid = request.match_info["id"]
        persona = store.get_persona(pid)
        if persona is None:
            return web.json_response({"error": "Not found"}, status=404)
        return web.json_response(persona.to_dict())

    async def create_persona(request):
        data = await request.json()
        persona = store.create_persona(data)
        return web.json_response(persona.to_dict(), status=201)

    async def update_persona(request):
        pid = request.match_info["id"]
        data = await request.json()
        persona = store.update_persona(pid, data)
        if persona is None:
            return web.json_response({"error": "Not found"}, status=404)
        return web.json_response(persona.to_dict())

    async def delete_persona(request):
        pid = request.match_info["id"]
        if store.delete_persona(pid):
            return web.json_response({"deleted": True})
        return web.json_response({"error": "Not found"}, status=404)

    # --- MCP Server endpoints ---

    async def list_mcp_servers(_request):
        servers = store.list_mcp_servers()
        result = []
        for s in servers:
            entry = s.to_dict()
            # Add live connection status
            status = mcp_client.server_status(s.name)
            entry["connected"] = status["connected"]
            entry["tools"] = status["tools"]
            result.append(entry)
        return web.json_response(result)

    async def get_mcp_server(request):
        sid = request.match_info["id"]
        server = store.get_mcp_server(sid)
        if server is None:
            return web.json_response({"error": "Not found"}, status=404)
        entry = server.to_dict()
        status = mcp_client.server_status(server.name)
        entry["connected"] = status["connected"]
        entry["tools"] = status["tools"]
        return web.json_response(entry)

    async def create_mcp_server(request):
        data = await request.json()
        server = store.create_mcp_server(data)
        # Register with MCP client
        config = MCPServerConfig(
            name=server.name,
            command=server.command or "",
            args=server.args,
            env=server.env,
            transport=server.transport,
            url=server.url,
        )
        mcp_client.add_server(config)
        # Auto-connect if enabled
        if server.enabled:
            connect_result = await mcp_client.connect_server_by_name(server.name)
            entry = server.to_dict()
            entry["connected"] = connect_result.get("connected", False)
            entry["tools"] = connect_result.get("tools", [])
            if "error" in connect_result:
                entry["connection_error"] = connect_result["error"]
            return web.json_response(entry, status=201)
        return web.json_response(server.to_dict(), status=201)

    async def update_mcp_server(request):
        sid = request.match_info["id"]
        data = await request.json()
        old_server = store.get_mcp_server(sid)
        server = store.update_mcp_server(sid, data)
        if server is None:
            return web.json_response({"error": "Not found"}, status=404)
        # If name or connection details changed, reconnect
        if old_server and (
            old_server.name != server.name
            or old_server.url != server.url
            or old_server.command != server.command
            or old_server.transport != server.transport
        ):
            await mcp_client.remove_server(old_server.name)
            config = MCPServerConfig(
                name=server.name,
                command=server.command or "",
                args=server.args,
                env=server.env,
                transport=server.transport,
                url=server.url,
            )
            mcp_client.add_server(config)
            if server.enabled:
                await mcp_client.connect_server_by_name(server.name)
        entry = server.to_dict()
        status = mcp_client.server_status(server.name)
        entry["connected"] = status["connected"]
        entry["tools"] = status["tools"]
        return web.json_response(entry)

    async def delete_mcp_server(request):
        sid = request.match_info["id"]
        server = store.get_mcp_server(sid)
        if server is None:
            return web.json_response({"error": "Not found"}, status=404)
        await mcp_client.remove_server(server.name)
        store.delete_mcp_server(sid)
        return web.json_response({"deleted": True})

    async def connect_mcp_server(request):
        """Connect/reconnect to a specific MCP server."""
        sid = request.match_info["id"]
        server = store.get_mcp_server(sid)
        if server is None:
            return web.json_response({"error": "Not found"}, status=404)
        result = await mcp_client.connect_server_by_name(server.name)
        return web.json_response(result)

    async def disconnect_mcp_server(request):
        """Disconnect from a specific MCP server."""
        sid = request.match_info["id"]
        server = store.get_mcp_server(sid)
        if server is None:
            return web.json_response({"error": "Not found"}, status=404)
        await mcp_client.disconnect_server(server.name)
        return web.json_response({"name": server.name, "connected": False})

    async def test_mcp_connection(request):
        """Test connecting to an MCP server without saving it."""
        data = await request.json()
        config = MCPServerConfig(
            name=data.get("name", "test"),
            command=data.get("command", ""),
            args=data.get("args", []),
            env=data.get("env", {}),
            transport=data.get("transport", "sse"),
            url=data.get("url"),
        )
        result = await mcp_client.test_connection(config)
        return web.json_response(result)

    # Register routes
    app.router.add_get("/api/admin/personas", list_personas)
    app.router.add_post("/api/admin/personas", create_persona)
    app.router.add_get("/api/admin/personas/{id}", get_persona)
    app.router.add_put("/api/admin/personas/{id}", update_persona)
    app.router.add_delete("/api/admin/personas/{id}", delete_persona)

    app.router.add_get("/api/admin/mcp-servers", list_mcp_servers)
    app.router.add_post("/api/admin/mcp-servers", create_mcp_server)
    app.router.add_get("/api/admin/mcp-servers/{id}", get_mcp_server)
    app.router.add_put("/api/admin/mcp-servers/{id}", update_mcp_server)
    app.router.add_delete("/api/admin/mcp-servers/{id}", delete_mcp_server)
    app.router.add_post("/api/admin/mcp-servers/{id}/connect", connect_mcp_server)
    app.router.add_post("/api/admin/mcp-servers/{id}/disconnect", disconnect_mcp_server)
    app.router.add_post("/api/admin/mcp-servers/test", test_mcp_connection)

    logger.info("Admin API routes registered")
