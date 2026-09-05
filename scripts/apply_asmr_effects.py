"""Apply ASMR DSP effects to an existing WAV file (no TTS, no GPU required).

Standalone wrapper around `mcp_voice_studio.core.asmr.apply_asmr_pipeline`.

Usage:
    python scripts/apply_asmr_effects.py INPUT.wav OUTPUT.wav --text "..." [options]

Example:
    python scripts/apply_asmr_effects.py voice.wav out.wav \\
        --text "Benvenuto. Chiudi gli occhi. Respira." \\
        --stereo-pan "L<->R" --period-s 2.5 \\
        --silence-padding-ms 600 \\
        --reverb small_room \\
        --lowpass-cutoff-hz 6500.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add repo root to sys.path so `mcp_voice_studio` is importable
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from mcp_voice_studio.core.asmr import (  # noqa: E402
    apply_asmr_pipeline,
    read_wav,
    write_wav,
)


_VALID_PAN = {"center", "L", "R", "L<->R", "L->R", "R->L"}
_VALID_REVERB = {"none", "small_room", "large_room"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply ASMR DSP effects (highpass, lowpass, pan, reverb, binaural, padding) to a WAV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=Path, help="Input WAV file (mono or stereo)")
    p.add_argument("output", type=Path, help="Output WAV file")
    p.add_argument("--text", default="",
                   help="Original text (used to weight silence-padding positions; pass '.' for no padding weighting)")
    # Pan
    p.add_argument("--stereo-pan", choices=sorted(_VALID_PAN), default=None,
                   help="Stereo panning mode (None = passthrough mono)")
    p.add_argument("--period-s", type=float, default=2.0,
                   help="L<->R/L->R/R->L period in seconds (default 2.0)")
    # Padding
    p.add_argument("--silence-padding-ms", type=int, default=0,
                   help="Insert ms of silence at sentence boundaries (0-5000)")
    # Reverb
    p.add_argument("--reverb", choices=sorted(_VALID_REVERB), default=None,
                   help="Reverb preset")
    p.add_argument("--reverb-damping", type=float, default=0.5,
                   help="Reverb HF damping 0..1 (0=Schroeder classico, 0.5=morbido)")
    # Binaural
    p.add_argument("--binaural-beat-hz", type=float, default=0.0,
                   help="Binaural beat frequency in Hz (0=off, 4-8 = theta-alpha)")
    p.add_argument("--binaural-amplitude", type=float, default=0.0005,
                   help="Binaural carrier peak amplitude (default 0.0005 = -66dBFS sub-audible)")
    # Lowpass
    p.add_argument("--lowpass-cutoff-hz", type=float, default=0.0,
                   help="Lowpass cutoff in Hz (0=off, 5000-7000 sweet spot)")
    # Highpass (always on by default; pass 0 to disable)
    p.add_argument("--highpass-cutoff-hz", type=float, default=60.0,
                   help="Highpass cutoff in Hz (0=off, default 60 = DC/sub-bass cleanup)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    # Read input for stats display
    samples_in, sr_in = read_wav(args.input)
    n_in = samples_in.shape[0]
    ch_in = 1 if samples_in.ndim == 1 else samples_in.shape[1]
    print(f"Input:  {args.input}  sr={sr_in}  channels={ch_in}  samples={n_in}  ({n_in/sr_in:.2f}s)")

    # The pipeline does its own read/write; we need a non-empty text for padding weighting.
    text = args.text if args.text else "."

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sr_out, ch_out = apply_asmr_pipeline(
        input_wav=args.input,
        output_wav=args.output,
        text=text,
        sample_rate=sr_in,
        stereo_pan=args.stereo_pan,
        period_s=args.period_s,
        silence_padding_ms=args.silence_padding_ms,
        reverb=args.reverb,
        reverb_damping=args.reverb_damping,
        binaural_beat_hz=args.binaural_beat_hz,
        binaural_amplitude=args.binaural_amplitude,
        lowpass_cutoff_hz=args.lowpass_cutoff_hz,
        highpass_cutoff_hz=args.highpass_cutoff_hz,
    )

    # Re-read to print duration
    samples_out, _ = read_wav(args.output)
    n_out = samples_out.shape[0]
    print(f"Output: {args.output}  sr={sr_out}  channels={ch_out}  samples={n_out}  ({n_out/sr_out:.2f}s)")
    applied = []
    if args.highpass_cutoff_hz > 0:
        applied.append(f"highpass({args.highpass_cutoff_hz}Hz)")
    if args.lowpass_cutoff_hz > 0:
        applied.append(f"lowpass({args.lowpass_cutoff_hz}Hz)")
    if args.stereo_pan and args.stereo_pan != "center":
        applied.append(f"stereo_pan({args.stereo_pan}, period={args.period_s}s)")
    if args.reverb and args.reverb != "none":
        applied.append(f"reverb({args.reverb}, damping={args.reverb_damping})")
    if args.binaural_beat_hz > 0:
        applied.append(f"binaural_beat({args.binaural_beat_hz}Hz, amp={args.binaural_amplitude})")
    if args.silence_padding_ms > 0:
        applied.append(f"silence_padding({args.silence_padding_ms}ms)")
    print(f"Effects applied: {json.dumps(applied)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
