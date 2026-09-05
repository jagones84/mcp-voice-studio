"""YouTube (and yt-dlp-supported) audio clip extraction.

Used by `clone_voice_from_audio` to source a reference audio from a URL
without the user having to download + slice manually.

Pipeline (runs on the local machine; here: DGX Spark):
  1. yt-dlp -f 'bestaudio[ext=m4a]/bestaudio' -o <tmp>  URL
  2. ffmpeg  -ss <ts> -to <tf> -i <tmp>  -ar 24000 -ac 1 -sample_fmt s16 <out>

Public API:
    extract_youtube_clip(url, output_wav, *, ts, tf, sample_rate, channels,
                          workdir, keep_intermediate) -> dict
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _which_or_die(tool: str) -> str:
    """Find a binary. Check PATH first, then common install locations for
    `uv tool install` (typically `~/.local/bin/<tool>`) which are NOT in
    the default PATH of SSH non-interactive shells.

    Returns the absolute path. Raises RuntimeError if not found anywhere.
    """
    import os
    candidates: list[str] = []
    found = shutil.which(tool)
    if found:
        return found
    home = os.path.expanduser("~")
    candidates.extend([
        os.path.join(home, ".local", "bin", tool),
        os.path.join(home, ".cargo", "bin", tool),
        f"/usr/local/bin/{tool}",
        f"/opt/homebrew/bin/{tool}",
    ])
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise RuntimeError(
        f"'{tool}' not found in PATH or common install locations. "
        f"Install it first (e.g. `uv tool install {tool}` or `apt install ffmpeg`)."
    )


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess; raise on failure with full command visible."""
    res = subprocess.run(cmd, capture_output=True, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {res.returncode}): {' '.join(cmd)}\n"
            f"--- STDOUT ---\n{res.stdout}\n--- STDERR ---\n{res.stderr}"
        )
    return res


def extract_youtube_clip(
    url: str,
    output_wav: str | Path,
    *,
    ts: float = 0.0,
    tf: Optional[float] = None,
    sample_rate: int = 24000,
    channels: int = 1,
    workdir: Optional[Path] = None,
    keep_intermediate: bool = False,
) -> dict:
    """Download `url` (yt-dlp), slice [ts, tf] seconds, convert to WAV.

    Returns a dict with: output_path, duration_s, sample_rate, channels,
    codec_name, output_size_bytes, source_size_bytes, intermediate_dir,
    source_url, ts, tf, yt_dlp_path, ffmpeg_path.
    """
    if ts < 0:
        raise ValueError(f"ts must be >= 0, got {ts}")
    if tf is not None and tf <= ts:
        raise ValueError(f"tf ({tf}) must be > ts ({ts})")

    yt_dlp = _which_or_die("yt-dlp")
    ffmpeg = _which_or_die("ffmpeg")
    ffprobe = _which_or_die("ffprobe")

    out_path = Path(output_wav).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(workdir) if workdir else out_path.parent / "_yt_clip_work"
    # SAFETY: never let work resolve to the same directory as out_path.
    # Otherwise the cleanup `shutil.rmtree(work)` below would delete the
    # output WAV we just produced.
    try:
        if work.resolve() == out_path.parent.resolve():
            work = work / "_yt_clip_work"
    except (OSError, RuntimeError):
        # If resolve() fails (e.g. broken symlink) fall back to lexical compare.
        if str(work).rstrip("/") == str(out_path.parent).rstrip("/"):
            work = work / "_yt_clip_work"
    work.mkdir(parents=True, exist_ok=True)

    # 1) yt-dlp
    raw_template = work / "raw.%(ext)s"
    _run([
        yt_dlp,
        "--no-playlist",
        "--no-warnings",
        "-f", "bestaudio[ext=m4a]/bestaudio/best",
        "-o", str(raw_template),
        url,
    ])
    candidates = sorted(work.glob("raw.*"))
    if not candidates:
        raise RuntimeError(f"yt-dlp did not produce any file in {work}")
    raw_path = candidates[-1]
    raw_size = raw_path.stat().st_size

    # 2) ffmpeg slice + convert
    duration: Optional[float] = (tf - ts) if tf is not None else None
    ffmpeg_cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if ts > 0:
        ffmpeg_cmd += ["-ss", str(ts)]
    ffmpeg_cmd += ["-i", str(raw_path)]
    if tf is not None:
        ffmpeg_cmd += ["-to", str(tf - ts)]
    ffmpeg_cmd += [
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-sample_fmt", "s16",
        "-vn",
        str(out_path),
    ]
    _run(ffmpeg_cmd)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"output WAV not produced or empty: {out_path}")

    # 3) ffprobe verify
    probe = _run([
        ffprobe, "-v", "error",
        "-show_entries", "stream=duration,sample_rate,channels,codec_name",
        "-of", "json",
        str(out_path),
    ])
    info = json.loads(probe.stdout)
    streams = info.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe could not parse output: {out_path}")
    s = streams[0]
    actual = {
        "duration_s": float(s.get("duration", 0.0)),
        "sample_rate": int(s.get("sample_rate", 0)),
        "channels": int(s.get("channels", 0)),
        "codec_name": s.get("codec_name"),
    }
    if actual["duration_s"] <= 0:
        raise RuntimeError(f"produced WAV has zero/negative duration: {actual}")
    if actual["sample_rate"] != sample_rate:
        print(
            f"  WARNING: requested {sample_rate} Hz, got {actual['sample_rate']} Hz",
            file=sys.stderr,
        )
    if actual["channels"] != channels:
        print(
            f"  WARNING: requested {channels} ch, got {actual['channels']} ch",
            file=sys.stderr,
        )

    if not keep_intermediate:
        shutil.rmtree(work, ignore_errors=True)

    return {
        "output_path": str(out_path),
        "source_url": url,
        "ts": ts,
        "tf": tf,
        "duration_s": actual["duration_s"],
        "sample_rate": actual["sample_rate"],
        "channels": actual["channels"],
        "codec_name": actual["codec_name"],
        "output_size_bytes": out_path.stat().st_size,
        "source_size_bytes": raw_size,
        "intermediate_dir": str(work) if keep_intermediate else None,
        "yt_dlp_path": yt_dlp,
        "ffmpeg_path": ffmpeg,
    }


def suggest_duration_seconds() -> tuple[int, int, int, int]:
    """Suggested duration window for voice cloning references.

    Returns (min_ideal_s, max_ideal_s, min_absolute_s, max_absolute_s).
    """
    return (5, 30, 3, 60)
