# SPDX-License-Identifier: MIT
"""Conftest: stub out heavy dependencies (torch, etc.) so the agent
sub-package can be imported without installing the full ML stack."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1. Stub out heavy third-party packages that moshi/__init__.py
#    transitively imports but that moshi.agent does not need.
# ---------------------------------------------------------------------------

_STUB_PACKAGES = [
    "torch", "torch.nn", "torch.nn.functional", "torch.nn.utils",
    "torch.nn.utils.parametrize", "torch.distributed",
    "torch.utils", "torch.utils.checkpoint",
    "torch.cuda", "torch.backends", "torch.backends.cuda", "torch.amp",
    "safetensors", "safetensors.torch",
    "huggingface_hub", "einops", "einops.layers", "einops.layers.torch",
    "sentencepiece", "sounddevice", "sphn",
    "aiohttp", "aiohttp.web",
]

for _name in _STUB_PACKAGES:
    sys.modules.setdefault(_name, MagicMock())

# ---------------------------------------------------------------------------
# 2. Plant a lightweight moshi top-level stub so that
#    ``from moshi.agent.events import ...`` resolves the sub-package
#    without executing the real moshi/__init__.py.
# ---------------------------------------------------------------------------

_moshi_pkg_dir = Path(__file__).resolve().parent.parent / "moshi"

_stub = types.ModuleType("moshi")
_stub.__path__ = [str(_moshi_pkg_dir)]
_stub.__package__ = "moshi"
sys.modules["moshi"] = _stub
