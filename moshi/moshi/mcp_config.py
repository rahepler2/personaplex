# SPDX-License-Identifier: MIT

"""
Configuration helpers for the MCP/RAG integration in PersonaPlex.
"""

import json
import logging
import os
from typing import Optional

from .mcp_client import MCPClient, MCPServerConfig, load_mcp_config
from .knowledge_base import KnowledgeBase, RAGConfig

logger = logging.getLogger(__name__)


async def setup_mcp_and_rag(
    mcp_config_path: Optional[str] = None,
    rag_enabled: bool = True,
    enrich_system_prompt: bool = True,
) -> tuple[Optional[MCPClient], Optional[RAGConfig]]:
    """Initialize the MCP client and RAG configuration.

    Args:
        mcp_config_path: Path to the MCP server configuration JSON file.
            If None, looks for MCP_CONFIG_PATH env var or ./mcp_config.json.
        rag_enabled: Whether to enable RAG retrieval.
        enrich_system_prompt: Whether to pre-fetch context for the system prompt.

    Returns:
        Tuple of (MCPClient or None, RAGConfig or None).
        Returns (None, None) if MCP is not configured.
    """
    if mcp_config_path is None:
        mcp_config_path = os.environ.get("MCP_CONFIG_PATH")

    if mcp_config_path is None:
        candidates = ["mcp_config.json", "mcp.json", ".mcp/config.json"]
        for candidate in candidates:
            if os.path.exists(candidate):
                mcp_config_path = candidate
                break

    if mcp_config_path is None or not os.path.exists(mcp_config_path):
        logger.info("No MCP configuration found. RAG knowledge base disabled.")
        return None, None

    logger.info(f"Loading MCP configuration from {mcp_config_path}")
    try:
        server_configs = load_mcp_config(mcp_config_path)
    except Exception as e:
        logger.error(f"Failed to load MCP config: {e}")
        return None, None

    if not server_configs:
        logger.info("No MCP servers configured.")
        return None, None

    mcp_client = MCPClient()
    for config in server_configs:
        mcp_client.add_server(config)

    try:
        await mcp_client.connect_all()
    except Exception as e:
        logger.error(f"Failed to connect to MCP servers: {e}")
        return None, None

    if not mcp_client.is_connected:
        logger.warning("No MCP servers connected successfully.")
        return None, None

    connected = mcp_client.connected_servers
    tools = mcp_client.available_tools
    logger.info(
        f"MCP ready: {len(connected)} server(s), {len(tools)} tool(s) available"
    )
    for tool in tools:
        logger.info(f"  - {tool.server_name}:{tool.name}: {tool.description}")

    rag_config = RAGConfig(
        enabled=rag_enabled,
        enrich_system_prompt=enrich_system_prompt,
    )

    return mcp_client, rag_config


def create_knowledge_base(
    mcp_client: Optional[MCPClient],
    rag_config: Optional[RAGConfig],
    on_context=None,
) -> Optional[KnowledgeBase]:
    """Create a KnowledgeBase instance if MCP is available.

    Args:
        mcp_client: Initialized MCPClient or None.
        rag_config: RAG configuration or None.
        on_context: Async callback for when new context is retrieved.

    Returns:
        KnowledgeBase instance or None.
    """
    if mcp_client is None or rag_config is None:
        return None
    if not rag_config.enabled:
        return None

    return KnowledgeBase(
        mcp_client=mcp_client,
        config=rag_config,
        on_context=on_context,
    )
