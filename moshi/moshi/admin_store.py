# SPDX-License-Identifier: MIT

"""
Persistent JSON-based storage for PersonaPlex admin data.

Stores personas and MCP server configurations in a JSON file
so they survive server restarts.
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = os.path.expanduser("~/.personaplex/admin_store.json")


@dataclass
class PersonaConfig:
    """A saved persona configuration."""
    id: str
    name: str
    description: str
    text_prompt: str
    voice_prompt: str
    text_temperature: float = 0.7
    text_topk: int = 25
    audio_temperature: float = 0.8
    audio_topk: int = 250
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PersonaConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MCPServerEntry:
    """A saved MCP server connection configuration."""
    id: str
    name: str
    transport: str  # "sse" or "stdio"
    url: Optional[str] = None  # For SSE
    command: Optional[str] = None  # For stdio
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MCPServerEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class AdminStore:
    """JSON file-based persistent storage for admin configurations."""

    def __init__(self, store_path: Optional[str] = None):
        self._path = store_path or DEFAULT_STORE_PATH
        self._personas: dict[str, PersonaConfig] = {}
        self._mcp_servers: dict[str, MCPServerEntry] = {}
        self._load()

    def _ensure_dir(self):
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if not os.path.exists(self._path):
            logger.info(f"No admin store found at {self._path}, starting fresh")
            self._seed_defaults()
            return
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            for p in data.get("personas", []):
                persona = PersonaConfig.from_dict(p)
                self._personas[persona.id] = persona
            for s in data.get("mcp_servers", []):
                server = MCPServerEntry.from_dict(s)
                self._mcp_servers[server.id] = server
            logger.info(
                f"Loaded {len(self._personas)} personas, "
                f"{len(self._mcp_servers)} MCP servers from {self._path}"
            )
        except Exception as e:
            logger.error(f"Failed to load admin store: {e}")
            self._seed_defaults()

    def _save(self):
        self._ensure_dir()
        data = {
            "personas": [p.to_dict() for p in self._personas.values()],
            "mcp_servers": [s.to_dict() for s in self._mcp_servers.values()],
        }
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    def _seed_defaults(self):
        """Add default personas on first run."""
        now = time.time()
        defaults = [
            PersonaConfig(
                id=str(uuid.uuid4()),
                name="Friendly Teacher",
                description="A wise and friendly teacher for Q&A",
                text_prompt="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
                voice_prompt="NATF0.pt",
                created_at=now, updated_at=now,
            ),
            PersonaConfig(
                id=str(uuid.uuid4()),
                name="Medical Office Receptionist",
                description="Handles new patient intake calls",
                text_prompt="You work for Dr. Jones's medical office, and you are receiving calls to record information for new patients. Information: Record full name, date of birth, any medication allergies, tobacco smoking history, alcohol consumption history, and any prior medical conditions. Assure the patient that this information will be confidential, if they ask.",
                voice_prompt="NATF1.pt",
                created_at=now, updated_at=now,
            ),
            PersonaConfig(
                id=str(uuid.uuid4()),
                name="Bank Agent",
                description="Bank customer service agent handling transaction issues",
                text_prompt="You work for First Neuron Bank which is a bank and your name is Alexis Kim. Information: The customer's transaction for $1,200 at Home Depot was declined. Verify customer identity. The transaction was flagged due to unusual location (transaction attempted in Miami, FL; customer normally transacts in Seattle, WA).",
                voice_prompt="NATF2.pt",
                created_at=now, updated_at=now,
            ),
            PersonaConfig(
                id=str(uuid.uuid4()),
                name="Mars Astronaut",
                description="An astronaut dealing with a reactor emergency on Mars",
                text_prompt="You enjoy having a good conversation. Have a technical discussion about fixing a reactor core on a spaceship to Mars. You are an astronaut on a Mars mission. Your name is Alex. You are already dealing with a reactor core meltdown on a Mars mission. Several ship systems are failing, and continued instability will lead to catastrophic failure. You explain what is happening and you urgently ask for help thinking through how to stabilize the reactor.",
                voice_prompt="NATM0.pt",
                created_at=now, updated_at=now,
            ),
        ]
        for p in defaults:
            self._personas[p.id] = p
        self._save()

    # --- Persona CRUD ---

    def list_personas(self) -> list[PersonaConfig]:
        return sorted(self._personas.values(), key=lambda p: p.created_at)

    def get_persona(self, persona_id: str) -> Optional[PersonaConfig]:
        return self._personas.get(persona_id)

    def create_persona(self, data: dict) -> PersonaConfig:
        now = time.time()
        persona = PersonaConfig(
            id=str(uuid.uuid4()),
            name=data.get("name", "Untitled"),
            description=data.get("description", ""),
            text_prompt=data.get("text_prompt", ""),
            voice_prompt=data.get("voice_prompt", "NATF0.pt"),
            text_temperature=data.get("text_temperature", 0.7),
            text_topk=data.get("text_topk", 25),
            audio_temperature=data.get("audio_temperature", 0.8),
            audio_topk=data.get("audio_topk", 250),
            created_at=now,
            updated_at=now,
        )
        self._personas[persona.id] = persona
        self._save()
        return persona

    def update_persona(self, persona_id: str, data: dict) -> Optional[PersonaConfig]:
        persona = self._personas.get(persona_id)
        if persona is None:
            return None
        for key in ("name", "description", "text_prompt", "voice_prompt",
                     "text_temperature", "text_topk", "audio_temperature", "audio_topk"):
            if key in data:
                setattr(persona, key, data[key])
        persona.updated_at = time.time()
        self._save()
        return persona

    def delete_persona(self, persona_id: str) -> bool:
        if persona_id in self._personas:
            del self._personas[persona_id]
            self._save()
            return True
        return False

    # --- MCP Server CRUD ---

    def list_mcp_servers(self) -> list[MCPServerEntry]:
        return sorted(self._mcp_servers.values(), key=lambda s: s.created_at)

    def get_mcp_server(self, server_id: str) -> Optional[MCPServerEntry]:
        return self._mcp_servers.get(server_id)

    def create_mcp_server(self, data: dict) -> MCPServerEntry:
        now = time.time()
        server = MCPServerEntry(
            id=str(uuid.uuid4()),
            name=data.get("name", "Untitled"),
            transport=data.get("transport", "sse"),
            url=data.get("url"),
            command=data.get("command"),
            args=data.get("args", []),
            env=data.get("env", {}),
            enabled=data.get("enabled", True),
            created_at=now,
            updated_at=now,
        )
        self._mcp_servers[server.id] = server
        self._save()
        return server

    def update_mcp_server(self, server_id: str, data: dict) -> Optional[MCPServerEntry]:
        server = self._mcp_servers.get(server_id)
        if server is None:
            return None
        for key in ("name", "transport", "url", "command", "args", "env", "enabled"):
            if key in data:
                setattr(server, key, data[key])
        server.updated_at = time.time()
        self._save()
        return server

    def delete_mcp_server(self, server_id: str) -> bool:
        if server_id in self._mcp_servers:
            del self._mcp_servers[server_id]
            self._save()
            return True
        return False
