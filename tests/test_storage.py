"""Tests for voice profile storage and management.

These don't require GPU or omnivoice - they test pure Python logic.
"""

import json
import shutil
import wave
from pathlib import Path

import pytest

from mcp_voice_studio.core import storage
from mcp_voice_studio.core.models import VoiceProfile


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    """Create a 1-second 24kHz mono WAV in tmp."""
    p = tmp_path / "sample.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 24000)
    return p


def test_save_load_voice(tmp_path: Path, sample_wav: Path, monkeypatch):
    """Saving and loading a voice profile round-trips."""
    # Redirect data dir to tmp - need to create dirs because the patched
    # function won't auto-mkdir like the real voice_dir does
    def fake_voice_dir(name: str) -> Path:
        p = tmp_path / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(storage, "voice_dir", fake_voice_dir)

    profile = VoiceProfile(
        name="test_voice",
        ref_audio_path=str(sample_wav),
        ref_text="hello world",
        language="English",
    )
    storage.save_voice(profile, sample_wav)
    loaded = storage.load_voice("test_voice")
    assert loaded.name == "test_voice"
    assert loaded.ref_text == "hello world"
    assert loaded.language == "English"
    assert Path(loaded.ref_audio_path).exists()


def test_list_voices_empty(tmp_path: Path, monkeypatch):
    """list_voices returns [] when no profiles exist."""
    monkeypatch.setattr(storage, "voice_dir", lambda n: tmp_path / n)
    assert storage.list_voices() == []


def test_delete_voice(tmp_path: Path, sample_wav: Path, monkeypatch):
    """delete_voice removes the profile dir."""
    def fake_voice_dir(name: str) -> Path:
        p = tmp_path / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(storage, "voice_dir", fake_voice_dir)
    profile = VoiceProfile(
        name="del_me",
        ref_audio_path=str(sample_wav),
        ref_text="bye",
    )
    storage.save_voice(profile, sample_wav)
    assert storage.voice_exists("del_me")
    assert storage.delete_voice("del_me") is True
    assert not storage.voice_exists("del_me")


def test_clone_voice_from_audio_invalid_name(sample_wav: Path):
    """Empty / non-alphanumeric voice_name is rejected."""
    from mcp_voice_studio.tools.clone_voice import clone_voice_from_audio
    with pytest.raises(ValueError):
        clone_voice_from_audio(
            ref_audio_path=str(sample_wav),
            ref_text="hi",
            voice_name="!!!",
        )


def test_clone_voice_overwrite_guard(sample_wav: Path):
    """clone_voice_from_audio refuses to overwrite by default."""
    from mcp_voice_studio.tools.clone_voice import clone_voice_from_audio
    from mcp_voice_studio.core import storage

    # Clean up if leftover from previous run
    if storage.voice_exists("dup_test"):
        storage.delete_voice("dup_test")

    clone_voice_from_audio(
        ref_audio_path=str(sample_wav),
        ref_text="first",
        voice_name="dup_test",
    )
    try:
        with pytest.raises(ValueError, match="already exists"):
            clone_voice_from_audio(
                ref_audio_path=str(sample_wav),
                ref_text="second",
                voice_name="dup_test",
            )
    finally:
        storage.delete_voice("dup_test")


def test_instruct_validation():
    """Invalid instruct keywords are rejected by synthesize_speech."""
    from mcp_voice_studio.tools.synthesize import _validate_instruct

    # Valid
    assert _validate_instruct("whisper, female, low pitch") == "whisper, female, low pitch"
    # Empty / None pass through
    assert _validate_instruct(None) is None
    assert _validate_instruct("") is None
    # Invalid keyword raises
    with pytest.raises(ValueError, match="invalid instruct keywords"):
        _validate_instruct("whisper, super-soft")
