"""End-to-end: clone a voice (or design) + synthesize + apply ASMR effects.

Usage:
    # Cloned voice + ASMR
    python scripts/clone_and_speak.py --voice claudia_asmr --text "..." --out out.wav \\
        --stereo-pan "L<->R" --silence-padding-ms 600 --reverb small_room

    # Voice design (no clone) + ASMR
    python scripts/clone_and_speak.py --instruct "whisper, female, low pitch" \\
        --text "Hello world" --out out.wav --reverb small_room
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from mcp_voice_studio.tools.synthesize import synthesize_speech  # noqa: E402


_VALID_PAN = {"center", "L", "R", "L<->R", "L->R", "R->L"}
_VALID_REVERB = {"none", "small_room", "large_room"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E2E voice cloning/design + ASMR DSP effects (calls MCP tool function directly).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--text", required=True, help="Text to speak")
    p.add_argument("--voice", default=None, help="Cloned voice name (from clone_voice_from_audio)")
    p.add_argument("--instruct", default=None, help="Voice design keywords (e.g. 'whisper, female, low pitch')")
    p.add_argument("--language", default="Italian", help="Target language")
    p.add_argument("--num-step", type=int, default=32, help="Diffusion steps (4-100)")
    p.add_argument("--guidance-scale", type=float, default=2.0, help="CFG scale (0-10)")
    p.add_argument("--speed", type=float, default=1.0, help="Speed factor (0.25-4)")
    p.add_argument("--out", type=Path, required=True, help="Output WAV path")
    # ASMR
    p.add_argument("--stereo-pan", choices=sorted(_VALID_PAN), default=None)
    p.add_argument("--period-s", type=float, default=2.0)
    p.add_argument("--silence-padding-ms", type=int, default=0)
    p.add_argument("--reverb", choices=sorted(_VALID_REVERB), default=None)
    p.add_argument("--binaural-beat-hz", type=float, default=0.0)
    p.add_argument("--lowpass-cutoff-hz", type=float, default=0.0)
    return p.parse_args()


def main() -> int:
    if sys.version_info >= (3, 14):
        # The MCP tool wrappers may not yet support 3.14 - warn early.
        print("WARN: running on Python 3.14+, some deps may need rebuild", file=sys.stderr)

    args = _parse_args()
    if not args.voice and not args.instruct:
        print("ERROR: pass --voice <name> OR --instruct <keywords>", file=sys.stderr)
        return 2

    print(f"Synthesizing text={len(args.text)} chars voice={args.voice!r} instruct={args.instruct!r}")
    result = synthesize_speech(
        text=args.text,
        voice_name=args.voice,
        language=args.language,
        instruct=args.instruct,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        speed=args.speed,
        output_path=str(args.out),
        stereo_pan=args.stereo_pan,
        silence_padding_ms=args.silence_padding_ms,
        reverb=args.reverb,
        binaural_beat_hz=args.binaural_beat_hz,
        lowpass_cutoff_hz=args.lowpass_cutoff_hz,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
