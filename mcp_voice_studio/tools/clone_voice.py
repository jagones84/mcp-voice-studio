"""Voice cloning tool: save a voice profile from a reference audio file OR from a URL clip.

Two input modes:
  1. Direct file: pass ref_audio_path (existing local file).
  2. URL clip: pass audio_url + (optional) ts, tf to download a [ts..tf] clip
     from a YouTube (or any yt-dlp-supported) URL on the fly.
"""
from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

from ..core.models import VoiceProfile
from ..core.storage import load_voice, save_voice, voice_exists
from ..core.youtube import extract_youtube_clip, suggest_duration_seconds

log = logging.getLogger(__name__)

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
    ref_audio_path: Optional[str] = None,
    ref_text: str = "",
    voice_name: str = "",
    description: Optional[str] = None,
    language: str = "auto",
    overwrite: bool = False,
    audio_url: Optional[str] = None,
    ts: float = 0.0,
    tf: Optional[float] = None,
    workdir: Optional[str] = None,
) -> dict:
    """Save a voice profile from a reference audio (5-30s WAV ideal).

    Two input modes (one of ref_audio_path / audio_url MUST be provided):
      A) ref_audio_path: local file path (WAV/MP3/etc., 5-30s ideal).
      B) audio_url: YouTube (or yt-dlp-supported) URL. Combined with ts/tf
         it downloads + slices a clip into a temp file and uses that as the
         reference.

    Args:
        ref_audio_path: Path to the reference audio file (mode A).
        ref_text: Transcript of what is spoken in the reference audio (required).
        voice_name: Unique identifier for this voice (will be slugified).
        description: Optional human-readable description.
        language: Language of the reference audio (default: 'auto').
        overwrite: If True, replace an existing profile with the same name.
        audio_url: YouTube/yt-dlp URL to source the reference from (mode B).
        ts: Start second for the URL clip (default: 0).
        tf: End second for the URL clip (default: end of stream).
        workdir: Optional work directory for the download (default: temp dir).

    Returns:
        Dict with status, voice_name, profile_path, ref_audio_path, source.

    Raises:
        ValueError: if neither ref_audio_path nor audio_url is provided, or
                    if both are provided (ambiguous), or validation fails.
        FileNotFoundError: if ref_audio_path is provided but missing.
        RuntimeError: if yt-dlp/ffmpeg fails to produce the clip.
    """
    # --- Validate mode selection ---
    if not ref_audio_path and not audio_url:
        raise ValueError(
            "Provide exactly one of: ref_audio_path (local file) OR audio_url "
            "(YouTube/yt-dlp URL with optional ts/tf)."
        )
    if ref_audio_path and audio_url:
        raise ValueError(
            "Provide exactly one of ref_audio_path OR audio_url, not both."
        )
    if not ref_text or not ref_text.strip():
        raise ValueError("ref_text is required and must not be empty")
    if not voice_name or not voice_name.strip():
        raise ValueError("voice_name is required")

    name = _slug(voice_name)
    if voice_exists(name) and not overwrite:
        raise ValueError(
            f"voice '{name}' already exists. Pass overwrite=True to replace it, "
            f"or use a different voice_name."
        )

    # --- Mode B: download from URL ---
    source_info: dict = {}
    if audio_url:
        min_ideal, max_ideal, min_abs, max_abs = suggest_duration_seconds()
        if tf is not None and (tf - ts) < min_abs:
            raise ValueError(
                f"URL clip duration is too short ({(tf - ts):.1f}s < {min_abs}s). "
                f"Voice cloning needs at least {min_abs}s; ideal is {min_ideal}-{max_ideal}s."
            )
        if tf is not None and (tf - ts) > max_abs:
            log.warning(
                "URL clip duration %.1fs exceeds the recommended maximum of %ds. "
                "The clone will still work but generation is slower.",
                (tf - ts), max_abs,
            )

        wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="voice_studio_yt_"))
        wd.mkdir(parents=True, exist_ok=True)
        out_wav = wd / "ref_clip.wav"
        log.info("Extracting URL clip: %s [%.1fs..%ss] -> %s", audio_url, ts, tf, out_wav)
        try:
            source_info = extract_youtube_clip(
                audio_url, out_wav, ts=ts, tf=tf, workdir=wd, keep_intermediate=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to extract URL clip: {e}") from e
        src = out_wav
        duration_s = source_info.get("duration_s", 0.0)
        if min_ideal <= duration_s <= max_ideal:
            log.info("URL clip duration %.1fs is in the ideal range.", duration_s)
        elif duration_s < min_ideal:
            log.warning(
                "URL clip duration %.1fs is shorter than the ideal minimum %ds. "
                "Consider using a longer ts..tf window.",
                duration_s, min_ideal,
            )
    else:
        # --- Mode A: local file ---
        src = Path(ref_audio_path).expanduser().resolve()  # type: ignore[arg-type]
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

    out = {
        "status": "ok",
        "voice_name": name,
        "profile_path": str(vd / "profile.json"),
        "ref_audio_path": str(vd / "ref_audio.wav"),
        "ref_text": ref_text.strip(),
        "language": language,
        "description": description,
        "overwritten": voice_exists(name),
    }
    if audio_url:
        out["source"] = "url_clip"
        out["source_url"] = audio_url
        out["ts"] = ts
        out["tf"] = tf
        out["clip_duration_s"] = source_info.get("duration_s")
        out["clip_sample_rate"] = source_info.get("sample_rate")
        out["clip_channels"] = source_info.get("channels")
    else:
        out["source"] = "local_file"
    return out
