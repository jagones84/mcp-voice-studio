"""Voice profile persistence: load/save/list/delete profiles as JSON + ref audio."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterator

from .config import voice_dir
from .models import VoiceProfile


def _profile_path(name: str) -> Path:
    return voice_dir(name) / "profile.json"


def save_voice(profile: VoiceProfile, ref_audio_src: Path) -> Path:
    """Copy ref audio into voice dir and write profile.json.

    Args:
        profile: VoiceProfile to persist.
        ref_audio_src: Source path of the reference audio (will be copied as ref_audio.wav).

    Returns:
        Path to the voice dir.
    """
    vd = voice_dir(profile.name)
    dst_audio = vd / "ref_audio.wav"
    if not ref_audio_src.exists():
        raise FileNotFoundError(f"ref audio not found: {ref_audio_src}")
    shutil.copy2(ref_audio_src, dst_audio)
    # ref_text
    (vd / "ref_text.txt").write_text(profile.ref_text, encoding="utf-8")
    # profile.json (with updated path inside the voice dir)
    profile.ref_audio_path = str(dst_audio)
    _profile_path(profile.name).write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return vd


def load_voice(name: str) -> VoiceProfile:
    p = _profile_path(name)
    if not p.exists():
        raise FileNotFoundError(f"voice profile not found: {name}")
    return VoiceProfile.model_validate_json(p.read_text(encoding="utf-8"))


def list_voices() -> list[VoiceProfile]:
    """List all voice profiles sorted by name."""
    out: list[VoiceProfile] = []
    base = voice_dir("..").parent  # voices/
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        pp = d / "profile.json"
        if pp.exists():
            try:
                out.append(VoiceProfile.model_validate_json(pp.read_text(encoding="utf-8")))
            except Exception:
                # Corrupt profile - skip
                continue
    return out


def delete_voice(name: str) -> bool:
    """Delete a voice profile dir. Returns True if deleted, False if not found."""
    vd = voice_dir(name)
    pp = vd / "profile.json"
    if not pp.exists():
        return False
    shutil.rmtree(vd)
    return True


def voice_exists(name: str) -> bool:
    return _profile_path(name).exists()
