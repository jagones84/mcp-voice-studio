"""Pydantic models: voice profile + synthesis request."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class VoiceProfile(BaseModel):
    """A saved voice profile (one cloned voice)."""

    name: str = Field(..., description="Unique voice identifier (slug)")
    description: Optional[str] = Field(None, description="Human-readable description")
    ref_audio_path: str = Field(..., description="Path to reference WAV (5-30s ideal)")
    ref_text: str = Field(..., description="Transcript of the reference audio")
    language: str = Field("auto", description="Language of ref audio, e.g. 'Italian'")
    created_at: datetime = Field(default_factory=datetime.now)
    source: str = Field("cloned", description="'cloned' (from audio) or 'designed' (from instruct)")


class SynthRequest(BaseModel):
    """A speech synthesis request."""

    text: str = Field(..., min_length=1, description="Text to synthesize")
    voice_name: Optional[str] = Field(None, description="Saved voice name (from clone_voice_from_audio)")
    language: str = Field("Italian", description="Target language, e.g. 'Italian'")
    instruct: Optional[str] = Field(None, description="Voice design keywords (English or Chinese, comma+space separated)")
    num_step: int = Field(32, ge=4, le=100)
    guidance_scale: float = Field(2.0, ge=0.0, le=10.0)
    speed: float = Field(1.0, ge=0.25, le=4.0)
    output_path: Optional[str] = Field(None, description="Output WAV path; default = data/outputs/<ts>.wav")

    # ASMR post-processing (optional, all default to passthrough / off)
    stereo_pan: Optional[str] = Field(
        None,
        description=(
            "Stereo panning mode for ASMR. One of: 'center' (mono->stereo equal L/R), "
            "'L' (hard left), 'R' (hard right), 'L<->R' (alternating per sentence), "
            "'L->R' (slow sweep L to R), 'R->L' (slow sweep R to L). None = no panning."
        ),
    )
    silence_padding_ms: int = Field(
        0,
        ge=0,
        le=5000,
        description="Milliseconds of silence inserted between sentences (split on . ? !). 0 = off.",
    )
    reverb: Optional[str] = Field(
        None,
        description="Reverb mode: 'none', 'small_room' (ASMR-tight), 'large_room' (spacious). None = off.",
    )
    binaural_beat_hz: float = Field(
        0.0,
        ge=0.0,
        le=40.0,
        description="Binaural beat frequency in Hz (L = carrier, R = carrier + beat). 0 = off. ASMR relax: 4-8 Hz.",
    )
    lowpass_cutoff_hz: float = Field(
        0.0,
        ge=0.0,
        le=20000.0,
        description="Lowpass cutoff in Hz for warmth/intimacy. 0 = off. ASMR sweet spot: 5000-7000.",
    )


class SynthResult(BaseModel):
    """Result of a synthesis."""

    output_path: str
    duration_s: float
    sample_rate: int
    channels: int
    model: str
    voice_name: Optional[str] = None
    generation_time_s: float
