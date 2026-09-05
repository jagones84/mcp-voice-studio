"""Voice cloning tool: save a voice profile from a reference audio file."""

from __future__ import annotations

import re
from pathlib import Path

from ..core.models import VoiceProfile
from ..core.storage import load_voice, save_voice, voice_exists


_SLUG = re.compile(r"[^a-z0-9_]+")


def _slug(name: str) -> str:
    s = name.lower().strip()
    s = _SLUG.sub("_", s).strip("_")
    if not s:
        raise ValueError("voice_name must contain at least one alphanumeric character")
    if len(s) > 64:
        raise ValueError("voice_name too long (max 64 chars after slug)")
    return s


def clone_voice_from_audio(
    ref_audio_path: str,
    ref_text: str,
    voice_name: str,
    description: str | None = None,
    language: str = "auto",
    overwrite: bool = False,
) -> dict:
    """Save a voice profile from a reference audio (5-30s WAV ideal).

    Args:
        ref_audio_path: Path to the reference audio file (WAV/MP3/etc., 5-30s ideal).
        ref_text: Transcript of what is spoken in the reference audio.
        voice_name: Unique identifier for this voice (will be slugified).
        description: Optional human-readable description.
        language: Language of the reference audio (default: 'auto').
        overwrite: If True, replace an existing profile with the same name.

    Returns:
        Dict with status, voice_name, profile_path, ref_audio_path.
    """
    name = _slug(voice_name)
    if voice_exists(name) and not overwrite:
        raise ValueError(
            f"voice '{name}' already exists. Pass overwrite=True to replace it, "
            f"or use a different voice_name."
        )
    if not ref_text or not ref_text.strip():
        raise ValueError("ref_text is required and must not be empty")
    src = Path(ref_audio_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"ref audio not found: {src}")

    profile = VoiceProfile(
        name=name,
        description=description,
        ref_audio_path=str(src),
        ref_text=ref_text.strip(),
        language=language,
        source="cloned",
    )
    vd = save_voice(profile, src)
    return {
        "status": "ok",
        "voice_name": name,
        "profile_path": str(vd / "profile.json"),
        "ref_audio_path": str(vd / "ref_audio.wav"),
        "ref_text": ref_text.strip(),
        "language": language,
        "description": description,
        "overwritten": voice_exists(name),
    }
