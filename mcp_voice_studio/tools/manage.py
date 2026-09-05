"""Voice management tool: list, get, delete voice profiles."""

from __future__ import annotations

from ..core.storage import delete_voice as _delete, list_voices, load_voice, voice_exists


def list_voices_tool() -> list[dict]:
    """List all saved voice profiles.

    Returns:
        List of dicts with name, description, language, source, ref_audio_path, created_at.
    """
    return [p.model_dump(mode="json") for p in list_voices()]


def get_voice_info(voice_name: str) -> dict:
    """Get metadata of a single voice profile.

    Args:
        voice_name: Name of the voice.

    Returns:
        Dict with full profile fields.
    """
    if not voice_exists(voice_name):
        raise ValueError(f"voice '{voice_name}' not found")
    return load_voice(voice_name).model_dump(mode="json")


def delete_voice(voice_name: str) -> dict:
    """Delete a voice profile and its ref audio.

    Args:
        voice_name: Name of the voice to delete.

    Returns:
        Dict with status and voice_name.
    """
    if not _delete(voice_name):
        raise ValueError(f"voice '{voice_name}' not found")
    return {"status": "deleted", "voice_name": voice_name}
