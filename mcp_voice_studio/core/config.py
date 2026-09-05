"""Configuration: paths, env vars, defaults.

VoiceStudio .venv must be reachable. Default = ../VoiceStudio/.venv.
Override with VOICESTUDIO_VENV env var.
"""

from __future__ import annotations

import os
from pathlib import Path


def _package_root() -> Path:
    """Repo root containing mcp_voice_studio/."""
    return Path(__file__).resolve().parent.parent.parent


def package_root() -> Path:
    return _package_root()


def data_root() -> Path:
    """Runtime data dir: mcp_voice_studio/data/."""
    p = Path(__file__).resolve().parent.parent / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def voices_dir() -> Path:
    p = data_root() / "voices"
    p.mkdir(parents=True, exist_ok=True)
    return p


def outputs_dir() -> Path:
    p = data_root() / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir() -> Path:
    p = data_root() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def voice_dir(name: str) -> Path:
    """Per-voice subdir containing ref_audio.wav, ref_text.txt, profile.json."""
    p = voices_dir() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def voicestudio_venv() -> Path:
    """Path to VoiceStudio's existing venv (with torch+omnivoice)."""
    env = os.environ.get("VOICESTUDIO_VENV")
    if env:
        return Path(env)
    # Default: sibling repo on DGX
    default = package_root().parent / "VoiceStudio" / ".venv"
    if default.exists():
        return default
    raise FileNotFoundError(
        f"VoiceStudio .venv not found at {default}. Set VOICESTUDIO_VENV env var."
    )


def voicestudio_root() -> Path:
    """Path to VoiceStudio repo (parent of .venv)."""
    return voicestudio_venv().parent


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def cuda_device() -> str:
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0")
