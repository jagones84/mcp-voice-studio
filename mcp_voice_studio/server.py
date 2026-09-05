"""MCP server entry point.

Registers all tools from mcp_voice_studio.tools with FastMCP and runs the stdio server.

Run with:
    uv run python -m mcp_voice_studio.server
"""

from __future__ import annotations

import logging
import sys

from mcp.server.fastmcp import FastMCP

from .tools.clone_voice import clone_voice_from_audio
from .tools.manage import delete_voice, get_voice_info, list_voices_tool
from .tools.synthesize import design_voice, synthesize_speech


log = logging.getLogger("mcp_voice_studio")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")


mcp = FastMCP(
    name="voice-studio",
    instructions=(
        "MCP server for VoiceStudio (OmniVoice engine). Tools:\n"
        "- clone_voice_from_audio: save a voice profile from a reference audio (5-30s WAV).\n"
        "- synthesize_speech: generate speech with a cloned or designed voice.\n"
        "- design_voice: generate speech with voice design keywords (no cloned voice).\n"
        "- list_voices: list all saved voice profiles.\n"
        "- get_voice_info: get metadata of a specific voice profile.\n"
        "- delete_voice: delete a voice profile and its reference audio.\n"
    ),
)

# --- Tool registration ---

@mcp.tool()
def tool_clone_voice_from_audio(
    ref_audio_path: str,
    ref_text: str,
    voice_name: str,
    description: str | None = None,
    language: str = "auto",
    overwrite: bool = False,
) -> dict:
    """Save a voice profile from a reference audio file. Requires 5-30s of clear speech and the exact transcript."""
    return clone_voice_from_audio(ref_audio_path, ref_text, voice_name, description, language, overwrite)


@mcp.tool()
def tool_synthesize_speech(
    text: str,
    voice_name: str | None = None,
    language: str = "Italian",
    instruct: str | None = None,
    num_step: int = 32,
    guidance_scale: float = 2.0,
    speed: float = 1.0,
    output_path: str | None = None,
    stereo_pan: str | None = None,
    silence_padding_ms: int = 0,
    reverb: str | None = None,
    binaural_beat_hz: float = 0.0,
    lowpass_cutoff_hz: float = 0.0,
) -> dict:
    """Generate speech audio. Provide voice_name (cloned) OR instruct (voice design). Both OK too.

    ASMR enhancements (all optional, applied as a post-synth pipeline):
      stereo_pan: 'center' (mono->stereo equal L/R), 'L', 'R', 'L<->R' (alternating),
                  'L->R' (slow sweep), 'R->L'.
      silence_padding_ms: ms of silence between sentences (split on . ? !), 0 = off.
      reverb: 'small_room' (ASMR-tight), 'large_room' (spacious), None = off.
      binaural_beat_hz: 0 = off. ASMR relax: 4-8 Hz (theta-alpha entrainment).
      lowpass_cutoff_hz: 0 = off. ASMR warmth: 5000-7000 Hz.
    """
    return synthesize_speech(
        text, voice_name, language, instruct, num_step, guidance_scale, speed,
        output_path, stereo_pan, silence_padding_ms, reverb, binaural_beat_hz, lowpass_cutoff_hz,
    )


@mcp.tool()
def tool_design_voice(
    text: str,
    instruct: str,
    language: str = "Italian",
    num_step: int = 32,
    guidance_scale: float = 2.0,
    speed: float = 1.0,
    output_path: str | None = None,
    stereo_pan: str | None = None,
    silence_padding_ms: int = 0,
    reverb: str | None = None,
    binaural_beat_hz: float = 0.0,
    lowpass_cutoff_hz: float = 0.0,
) -> dict:
    """Generate speech using voice design keywords only (no cloned voice). Same ASMR params as synthesize_speech."""
    return design_voice(
        text, instruct, language, num_step, guidance_scale, speed,
        output_path, stereo_pan, silence_padding_ms, reverb, binaural_beat_hz, lowpass_cutoff_hz,
    )


@mcp.tool()
def tool_list_voices() -> list[dict]:
    """List all saved voice profiles."""
    return list_voices_tool()


@mcp.tool()
def tool_get_voice_info(voice_name: str) -> dict:
    """Get metadata of a single saved voice profile."""
    return get_voice_info(voice_name)


@mcp.tool()
def tool_delete_voice(voice_name: str) -> dict:
    """Delete a voice profile and its reference audio."""
    return delete_voice(voice_name)


def main() -> None:
    log.info("starting voice-studio MCP server (stdio)")
    mcp.run()


if __name__ == "__main__":
    main()
