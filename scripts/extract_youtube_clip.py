"""CLI wrapper around `mcp_voice_studio.core.youtube.extract_youtube_clip`.

Usage:
    python scripts/extract_youtube_clip.py URL OUTPUT.wav [--ts 30] [--tf 40]

Example:
    python scripts/extract_youtube_clip.py \\
        "https://www.youtube.com/watch?v=aqz-KE-bpKQ" /tmp/bbb_30_40.wav \\
        --ts 30 --tf 40

Suggested duration: 5-30 seconds is the sweet spot for voice cloning.
Below 3 s: too little acoustic data; above 60 s: unnecessary slowness.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add repo root to sys.path so `mcp_voice_studio` is importable
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from mcp_voice_studio.core.youtube import extract_youtube_clip  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract a [ts, tf] audio clip from a YouTube URL and convert to WAV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("url", help="YouTube (or yt-dlp-supported) URL")
    p.add_argument("output", help="Output WAV path (e.g. /tmp/clip.wav)")
    p.add_argument("--ts", type=float, default=0.0, help="Start second (default: 0)")
    p.add_argument("--tf", type=float, default=None, help="End second (default: end of stream)")
    p.add_argument("--sample-rate", type=int, default=24000, help="Target sample rate (default: 24000)")
    p.add_argument("--channels", type=int, default=1, help="Target channels (default: 1 = mono)")
    p.add_argument("--keep-intermediate", action="store_true", help="Keep raw downloaded file for debugging")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    result = extract_youtube_clip(
        args.url,
        args.output,
        ts=args.ts,
        tf=args.tf,
        sample_rate=args.sample_rate,
        channels=args.channels,
        keep_intermediate=args.keep_intermediate,
    )
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
