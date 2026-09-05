"""Synthesis tool: generate speech with a cloned or designed voice."""

from __future__ import annotations

import re
import time
import wave
from pathlib import Path

from ..core import engine
from ..core.asmr import apply_asmr_pipeline
from ..core.config import outputs_dir
from ..core.models import SynthRequest, SynthResult
from ..core.storage import voice_exists


# Valid OmniVoice English instruct keywords (subset, full set in README-voice-studio-dgx.md §3)
_VALID_INSTRUCT = {
    "american accent", "australian accent", "british accent", "canadian accent",
    "child", "chinese accent", "elderly", "female", "high pitch", "indian accent",
    "japanese accent", "korean accent", "low pitch", "male", "middle-aged",
    "moderate pitch", "portuguese accent", "russian accent", "teenager",
    "very high pitch", "very low pitch", "whisper", "young adult",
}

_VALID_PAN = {"center", "L", "R", "L<->R", "L->R", "R->L"}
_VALID_REVERB = {"none", "small_room", "large_room"}


def _validate_instruct(instruct: str | None) -> str | None:
    if not instruct:
        return None
    items = [i.strip() for i in instruct.split(",") if i.strip()]
    bad = [i for i in items if i not in _VALID_INSTRUCT]
    if bad:
        raise ValueError(
            f"invalid instruct keywords: {bad}. "
            f"Valid: {sorted(_VALID_INSTRUCT)}"
        )
    return ", ".join(items)


def _validate_asmr(req: SynthRequest) -> None:
    if req.stereo_pan is not None and req.stereo_pan not in _VALID_PAN:
        raise ValueError(
            f"invalid stereo_pan: {req.stereo_pan!r}. Valid: {sorted(_VALID_PAN)}"
        )
    if req.reverb is not None and req.reverb not in _VALID_REVERB:
        raise ValueError(
            f"invalid reverb: {req.reverb!r}. Valid: {sorted(_VALID_REVERB)}"
        )


def _wav_info(path: Path) -> tuple[float, int, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        n = w.getnframes()
    return n / sr, sr, ch


def _has_asmr_effects(req: SynthRequest) -> bool:
    return any([
        req.stereo_pan is not None,
        req.silence_padding_ms > 0,
        req.reverb is not None and req.reverb != "none",
        req.binaural_beat_hz > 0,
        req.lowpass_cutoff_hz > 0,
    ])


def synthesize_speech(
    text: str,
    voice_name: str | None = None,
    language: str = "Italian",
    instruct: str | None = None,
    num_step: int = 32,
    guidance_scale: float = 2.0,
    speed: float = 1.0,
    output_path: str | None = None,
    stereo_pan: str | None = None,
    period_s: float = 2.0,
    silence_padding_ms: int = 0,
    reverb: str | None = None,
    reverb_damping: float = 0.5,
    binaural_beat_hz: float = 0.0,
    binaural_amplitude: float = 0.0005,
    lowpass_cutoff_hz: float = 0.0,
    highpass_cutoff_hz: float = 60.0,
) -> dict:
    """Generate speech audio with a cloned voice (from voice_name) or voice design (instruct only).

    Args:
        text: The text to speak.
        voice_name: Saved voice name (from clone_voice_from_audio). Optional.
        language: Target language (default: Italian). Examples: 'English', 'Italian', 'French'.
        instruct: Voice design keywords, e.g. 'whisper, female, low pitch' (English only).
                  See README §3 for valid keywords. Optional if voice_name is set.
        num_step: Diffusion steps (4-100, default 32).
        guidance_scale: CFG scale (0-10, default 2.0).
        speed: Speed factor (0.25-4, default 1.0; <1 slower, >1 faster).
        output_path: Where to write the WAV. Default: data/outputs/synth_<ts>.wav.

        # ASMR enhancements (all optional, applied as a post-synth pipeline):
        stereo_pan: 'center' | 'L' | 'R' | 'L<->R' (alternating) | 'L->R' (sweep) | 'R->L'.
        period_s: Period in seconds for L<->R/L->R/R->L modes (default 2.0).
        silence_padding_ms: ms of silence between sentences (split on . ? !), 0 = off.
        reverb: 'small_room' | 'large_room' (subtle mix), None/'none' = off.
        reverb_damping: 0..1, HF damping in reverb tail (default 0.5, recommended).
        binaural_beat_hz: 0 = off (default). ASMR relax: 4-8 Hz (theta-alpha).
        binaural_amplitude: carrier amplitude 0..1, default 0.0005 (-66dBFS, sub-audible).
        lowpass_cutoff_hz: 0 = off. ASMR warmth: 5000-7000 Hz.
        highpass_cutoff_hz: default 60 Hz (always on). 0 = off.

    Returns:
        Dict with output_path, duration_s, sample_rate, channels, voice_name, generation_time_s,
        asmr_applied (list of effect names actually applied).
    """
    if not text or not text.strip():
        raise ValueError("text is required and must not be empty")
    if voice_name:
        if not voice_exists(voice_name):
            raise ValueError(
                f"voice '{voice_name}' not found. Use list_voices to see available voices."
            )
    elif not instruct:
        raise ValueError(
            "either voice_name or instruct must be provided "
            "(you can also provide both for voice design applied to a cloned voice)"
        )

    instruct = _validate_instruct(instruct)

    if output_path:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        ts = int(time.time())
        out = outputs_dir() / f"synth_{ts}.wav"

    req = SynthRequest(
        text=text.strip(),
        voice_name=voice_name,
        language=language,
        instruct=instruct,
        num_step=num_step,
        guidance_scale=guidance_scale,
        speed=speed,
        output_path=str(out),
        stereo_pan=stereo_pan,
        period_s=period_s,
        silence_padding_ms=silence_padding_ms,
        reverb=reverb,
        reverb_damping=reverb_damping,
        binaural_beat_hz=binaural_beat_hz,
        binaural_amplitude=binaural_amplitude,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
        highpass_cutoff_hz=highpass_cutoff_hz,
    )
    _validate_asmr(req)

    t0 = time.time()
    wall_s, model_id = engine.synth(req, out)

    asmr_applied: list[str] = []
    if _has_asmr_effects(req):
        # Synthesize raw to a temp file, then run ASMR pipeline to the final path
        raw = out.with_suffix(".raw.wav")
        out.rename(raw)
        try:
            sr_final, ch_final = apply_asmr_pipeline(
                input_wav=raw,
                output_wav=out,
                text=req.text,
                sample_rate=24000,
                stereo_pan=req.stereo_pan,
                period_s=req.period_s,
                silence_padding_ms=req.silence_padding_ms,
                reverb=req.reverb,
                reverb_damping=req.reverb_damping,
                binaural_beat_hz=req.binaural_beat_hz,
                binaural_amplitude=req.binaural_amplitude,
                lowpass_cutoff_hz=req.lowpass_cutoff_hz,
                highpass_cutoff_hz=req.highpass_cutoff_hz,
            )
            if req.highpass_cutoff_hz > 0:
                asmr_applied.append(f"highpass({req.highpass_cutoff_hz:g}Hz)")
            if req.lowpass_cutoff_hz > 0:
                asmr_applied.append(f"lowpass({req.lowpass_cutoff_hz:g}Hz)")
            if req.stereo_pan is not None:
                asmr_applied.append(f"stereo_pan({req.stereo_pan})")
            if req.reverb is not None and req.reverb != "none":
                asmr_applied.append(f"reverb({req.reverb}, damping={req.reverb_damping:g})")
            if req.binaural_beat_hz > 0:
                asmr_applied.append(f"binaural_beat({req.binaural_beat_hz:g}Hz)")
            if req.silence_padding_ms > 0:
                asmr_applied.append(f"silence_padding({req.silence_padding_ms}ms)")
        finally:
            if raw.exists():
                raw.unlink()

    dur, sr, ch = _wav_info(out)
    res = SynthResult(
        output_path=str(out),
        duration_s=dur,
        sample_rate=sr,
        channels=ch,
        model=model_id,
        voice_name=voice_name,
        generation_time_s=wall_s,
    )
    out_dict = res.model_dump()
    out_dict["asmr_applied"] = asmr_applied
    return out_dict


def design_voice(
    text: str,
    instruct: str,
    language: str = "Italian",
    num_step: int = 32,
    guidance_scale: float = 2.0,
    speed: float = 1.0,
    output_path: str | None = None,
    stereo_pan: str | None = None,
    period_s: float = 2.0,
    silence_padding_ms: int = 0,
    reverb: str | None = None,
    reverb_damping: float = 0.5,
    binaural_beat_hz: float = 0.0,
    binaural_amplitude: float = 0.0005,
    lowpass_cutoff_hz: float = 0.0,
    highpass_cutoff_hz: float = 60.0,
) -> dict:
    """Generate speech using ONLY voice design (no cloned voice).

    Same parameters as synthesize_speech, including all 6 ASMR enhancements.
    """
    return synthesize_speech(
        text=text,
        voice_name=None,
        language=language,
        instruct=instruct,
        num_step=num_step,
        guidance_scale=guidance_scale,
        speed=speed,
        output_path=output_path,
        stereo_pan=stereo_pan,
        period_s=period_s,
        silence_padding_ms=silence_padding_ms,
        reverb=reverb,
        reverb_damping=reverb_damping,
        binaural_beat_hz=binaural_beat_hz,
        binaural_amplitude=binaural_amplitude,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
        highpass_cutoff_hz=highpass_cutoff_hz,
    )
