"""MCP server entry point.

Registers all tools from mcp_voice_studio.tools with FastMCP and runs the stdio server.

Run with:
    uv run python -m mcp_voice_studio.server

Each tool parameter is annotated with `Annotated[T, Field(description=...)]` so
the MCP `list_tools()` response includes a per-parameter description that
LLM-based MCP clients (Trae IDE, Claude Desktop, etc.) can use to understand
how to call the tool. Without this, the client only sees the parameter name
and type, but not what it means.
"""

from __future__ import annotations

import logging
from typing import Annotated, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .tools.clone_voice import clone_voice_from_audio
from .tools.manage import delete_voice, get_voice_info, list_voices_tool
from .tools.synthesize import design_voice, synthesize_speech


log = logging.getLogger("mcp_voice_studio")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")


# Reusable type aliases for cleaner signatures -------------------------------

# ASMR pipeline order: highpass (always on, 60Hz) -> lowpass -> stereo_pan ->
#                     reverb (Schroeder, HF-damped) -> binaural_beat -> silence_padding.
# All ASMR parameters are OPTIONAL. None / 0 = passthrough (off). The pipeline
# auto-promotes mono to stereo if a stereo effect is requested.

_PAN = Annotated[
    Optional[str],
    Field(
        description=(
            "Stereo panning mode for ASMR. One of: "
            "'center' (mono -> stereo, equal L/R), "
            "'L' (hard left, R muted), 'R' (hard right, L muted), "
            "'L<->R' (alternating L/R, ASMR 'whisper in each ear' effect, sweet spot period_s=2-3s), "
            "'L->R' (slow sweep L to R then back, sawtooth), "
            "'R->L' (slow sweep R to L then back, sawtooth). "
            "None = no panning (passthrough)."
        ),
    ),
]
_PADDING_MS = Annotated[
    int,
    Field(
        ge=0,
        le=5000,
        description=(
            "Milliseconds of silence inserted between sentences (split on . ? !). "
            "Position is weighted by sentence length. 0 = off. ASMR sweet spot: 400-800 ms."
        ),
    ),
]
_REVERB = Annotated[
    Optional[str],
    Field(
        description=(
            "Reverb mode: 'none' (off), 'small_room' (ASMR-tight, ~18% mix, HF-damped), "
            "'large_room' (spacious, longer tail). None = off. Default = off."
        ),
    ),
]
_BINAURAL_HZ = Annotated[
    float,
    Field(
        ge=0.0,
        le=40.0,
        description=(
            "Binaural beat frequency in Hz (L channel = 200Hz carrier, R channel = 200Hz+beat). "
            "0 = off (default, RECOMMENDED for clean output). "
            "Perceived as brainwave entrainment: 4-8 Hz = theta-alpha (sleep/relax), "
            "10-15 Hz = alpha-beta (focus), 15-40 Hz = beta (alert). "
            "Carrier amplitude is fixed at 0.0005 (-66dBFS, sub-audible)."
        ),
    ),
]
_LOWPASS_HZ = Annotated[
    float,
    Field(
        ge=0.0,
        le=20000.0,
        description=(
            "Lowpass cutoff in Hz for warmth/intimacy. 0 = off. "
            "ASMR sweet spot: 5000-7000 Hz (cuts above 7kHz for 'headphones' feel)."
        ),
    ),
]
_HIGHPASS_HZ = Annotated[
    float,
    Field(
        ge=0.0,
        le=1000.0,
        description=(
            "Highpass cutoff in Hz for DC/sub-bass cleanup. Default = 60 Hz (always on). "
            "0 = off. ASMR standard: 60-80 Hz to remove room rumble without affecting voice."
        ),
    ),
]


mcp = FastMCP(
    name="voice-studio",
    instructions=(
        "MCP server for VoiceStudio (OmniVoice engine). Local voice cloning + TTS on GPU.\n"
        "\n"
        "WORKFLOW:\n"
        "  1. clone_voice_from_audio(ref_audio_path | audio_url, ref_text, voice_name) -> saves a voice profile\n"
        "     - Mode A: ref_audio_path = local file path (WAV, 5-30s ideal)\n"
        "     - Mode B: audio_url = YouTube/yt-dlp URL + optional ts, tf for clip extraction\n"
        "  2. synthesize_speech(text, voice_name=..., [ASMR params]) -> generates speech audio\n"
        "  OR\n"
        "  design_voice(text, instruct='whisper, female, low pitch', [ASMR params]) -> voice design\n"
        "  3. list_voices / get_voice_info / delete_voice for management.\n"
        "\n"
        "URL CLIP EXTRACTION (clone_voice_from_audio Mode B):\n"
        "  - audio_url: full YouTube (or yt-dlp-supported) URL.\n"
        "  - ts: start second (default 0).\n"
        "  - tf: end second (default: end of stream).\n"
        "  - SUGGESTED DURATION: 5-30 seconds ideal. < 3s: too little acoustic data. > 60s: slow.\n"
        "  - Pipeline: yt-dlp download -> ffmpeg slice -> 24kHz mono 16-bit PCM WAV -> clone.\n"
        "\n"
        "ASMR EFFECTS (optional, post-synth DSP pipeline, all stereo-capable):\n"
        "  - stereo_pan: 'L<->R' for ear-to-ear whisper, 'L->R'/'R->L' for slow sweep\n"
        "  - silence_padding_ms: 400-800 ms between sentences (whisper pauses)\n"
        "  - reverb: 'small_room' (ASMR tight) or 'large_room' (spacious)\n"
        "  - binaural_beat_hz: 4-8 Hz (sleep/relax, OFF by default)\n"
        "  - lowpass_cutoff_hz: 5000-7000 (warmth)\n"
        "\n"
        "All ASMR params have safe defaults (off / passthrough). Output is stereo whenever "
        "any stereo effect is active. The response includes 'asmr_applied' listing what ran."
    ),
)


# --- Tool registration ------------------------------------------------------


@mcp.tool()
def tool_clone_voice_from_audio(
    ref_text: Annotated[
        str,
        Field(
            description=(
                "Exact transcript of what is spoken in the reference audio, including punctuation. "
                "OmniVoice uses this for prosody matching. Wrong transcripts degrade clone quality. "
                "Required in BOTH input modes (local file or URL clip)."
            ),
        ),
    ],
    voice_name: Annotated[
        str,
        Field(
            description=(
                "Unique identifier for this voice (alphanumeric + underscores, 1-64 chars). "
                "Will be slugified. Use a memorable name like 'claudia_asmr' or 'nasa_male_v1'. "
                "Used in subsequent synthesize_speech(voice_name=...) calls."
            ),
        ),
    ],
    ref_audio_path: Annotated[
        Optional[str],
        Field(
            description=(
                "Mode A: absolute path to the reference audio file (WAV, 16-bit PCM, 24kHz ideal). "
                "5-30 seconds of clear, single-speaker speech works best. "
                "Example: '/home/jagones/Repositories/VoiceStudio/inputs/asmr_sample.wav'. "
                "Use EITHER this OR audio_url, not both. Omit both to get a validation error."
            ),
        ),
    ] = None,
    audio_url: Annotated[
        Optional[str],
        Field(
            description=(
                "Mode B: YouTube (or any yt-dlp-supported) URL. Combined with ts/tf it downloads "
                "+ slices a clip into a temp file and uses that as the reference. "
                "Example: 'https://www.youtube.com/watch?v=21X5lGlDOfg'. "
                "SUGGESTED DURATION: 5-30 seconds ideal for voice cloning. "
                "Pipeline: yt-dlp download -> ffmpeg slice [ts, tf] -> 24kHz mono 16-bit PCM WAV. "
                "Use EITHER this OR ref_audio_path, not both."
            ),
        ),
    ] = None,
    ts: Annotated[
        float,
        Field(
            ge=0.0,
            description=(
                "Mode B only: start second of the URL clip (default 0). "
                "Example: ts=30 starts at 30s into the video. "
                "Combined with tf, defines the [ts, tf] clip window."
            ),
        ),
    ] = 0.0,
    tf: Annotated[
        Optional[float],
        Field(
            ge=0.0,
            description=(
                "Mode B only: end second of the URL clip (default: end of stream). "
                "Example: ts=30, tf=45 yields a 15-second clip from 30s to 45s. "
                "Voice cloning needs >=3s; 5-30s is the ideal range. "
                "Ignored if audio_url is not provided."
            ),
        ),
    ] = None,
    description: Annotated[
        Optional[str],
        Field(
            description="Optional human-readable description (e.g. 'ASMR whisper, female, CC0 from archive.org' or 'NASA mission audio, male, PD US Gov')."
        ),
    ] = None,
    language: Annotated[
        str,
        Field(
            description=(
                "Language of the reference audio. 'auto' (default, OmniVoice detects), "
                "or one of: 'English', 'Italian', 'French', 'German', 'Spanish', etc."
            ),
        ),
    ] = "auto",
    overwrite: Annotated[
        bool,
        Field(description="If True, replace an existing profile with the same voice_name. Default: False."),
    ] = False,
) -> dict:
    """Save a voice profile from a reference audio (5-30s WAV ideal).

    Two input modes (EITHER ref_audio_path OR audio_url must be provided, not both):
      A) Local file: pass ref_audio_path='/path/to/sample.wav'.
      B) URL clip:   pass audio_url='https://youtu.be/...', ts=30, tf=45.
                     yt-dlp downloads the audio and ffmpeg slices [ts, tf] into
                     a 24kHz mono 16-bit PCM WAV, which is then used as the reference.

    Workflow:
      1. Pick a clean 5-30s speech sample (local file OR URL clip).
      2. Write the exact transcript in ref_text.
      3. Pick a memorable voice_name (used later as voice_name= in synthesize_speech).
      4. Returns a dict with 'status', 'voice_name', 'profile_path', 'ref_audio_path',
         and (for URL mode) 'source_url', 'ts', 'tf', 'clip_duration_s'.

    Common errors:
      - both ref_audio_path and audio_url provided: pick one.
      - voice_exists: pass overwrite=True or pick a different voice_name.
      - bad local path: check the path exists and is readable.
      - bad URL: yt-dlp/ffmpeg failure will be reported in the raised RuntimeError.
    """
    return clone_voice_from_audio(
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        voice_name=voice_name,
        description=description,
        language=language,
        overwrite=overwrite,
        audio_url=audio_url,
        ts=ts,
        tf=tf,
    )


@mcp.tool()
def tool_synthesize_speech(
    text: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Text to speak. Use punctuation (. ? !) for natural pauses; combine with "
                "silence_padding_ms for ASMR-style long pauses between sentences."
            ),
        ),
    ],
    voice_name: Annotated[
        Optional[str],
        Field(
            description=(
                "Saved voice name from a prior clone_voice_from_audio call (e.g. 'claudia_asmr'). "
                "Optional: omit to use only voice design via 'instruct'."
            ),
        ),
    ] = None,
    language: Annotated[
        str,
        Field(
            description=(
                "Target language for synthesis. Default: 'Italian'. "
                "OmniVoice supports 600+ languages. Common: 'English', 'Italian', 'French', "
                "'German', 'Spanish', 'Japanese', 'Chinese', 'Korean'."
            ),
        ),
    ] = "Italian",
    instruct: Annotated[
        Optional[str],
        Field(
            description=(
                "Voice design keywords (OmniVoice specific, case-sensitive, English OR Chinese, "
                "comma+space separated, NEVER mix). Examples:\n"
                "  English: 'whisper, female, low pitch' (ASMR sweet spot)\n"
                "  English: 'male, young adult, british accent'\n"
                "  Chinese (full-width comma ,): '女，低音调，耳语'\n"
                "See README §'Voice design keywords' for the full list of 22+ accepted keywords. "
                "Combine with voice_name to add style to a cloned voice."
            ),
        ),
    ] = None,
    num_step: Annotated[
        int,
        Field(
            ge=4,
            le=100,
            description=(
                "Diffusion steps. Higher = better quality, slower. 32 (default) is a good balance. "
                "16 for fast drafts, 64 for production quality."
            ),
        ),
    ] = 32,
    guidance_scale: Annotated[
        float,
        Field(
            ge=0.0,
            le=10.0,
            description="Classifier-free guidance scale. 2.0 default. Higher = more prompt-faithful.",
        ),
    ] = 2.0,
    speed: Annotated[
        float,
        Field(
            ge=0.25,
            le=4.0,
            description="Speech rate factor. 1.0 = normal. <1 = slower (ASMR-like), >1 = faster.",
        ),
    ] = 1.0,
    output_path: Annotated[
        Optional[str],
        Field(
            description=(
                "Output WAV path. Default: 'mcp_voice_studio/data/outputs/synth_<timestamp>.wav'. "
                "For long ASMR tests use a stable path like '/home/.../outputs/asmr_test.wav'."
            ),
        ),
    ] = None,
    # ASMR pipeline params (all optional, all default to OFF)
    stereo_pan: _PAN = None,
    period_s: Annotated[
        float,
        Field(
            ge=0.1,
            le=20.0,
            description=(
                "Period in seconds for L<->R/L->R/R->L panning modes. "
                "ASMR sweet spot: 2.0-3.0s. Ignored if stereo_pan is None or 'center'/'L'/'R'."
            ),
        ),
    ] = 2.0,
    silence_padding_ms: _PADDING_MS = 0,
    reverb: _REVERB = None,
    reverb_damping: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "Reverb HF damping 0..1. 0 = classic Schroeder (metallic ring at high reverb). "
                "0.5 (default) = soft HF rolloff in reverb tail, recommended. "
                "1.0 = heavy damping (dark tail). Ignored if reverb is None or 'none'."
            ),
        ),
    ] = 0.5,
    binaural_beat_hz: _BINAURAL_HZ = 0.0,
    lowpass_cutoff_hz: _LOWPASS_HZ = 0.0,
    highpass_cutoff_hz: _HIGHPASS_HZ = 60.0,
) -> dict:
    """Generate speech audio with a cloned voice (from voice_name) OR voice design (from instruct), or both.

    Three usage modes:
      1. Cloned voice: pass voice_name='claudia_asmr', omit instruct.
      2. Voice design: pass instruct='whisper, female, low pitch', omit voice_name.
      3. Hybrid: pass both voice_name AND instruct to add style to a cloned voice.

    Returns dict with: output_path, duration_s, sample_rate, channels, model,
    voice_name, generation_time_s, and (if any ASMR effect ran) asmr_applied.

    ASMR pipeline: highpass(60Hz) -> lowpass -> stereo_pan -> reverb -> binaural -> padding.
    All ASMR params default to OFF (passthrough). Output is stereo whenever any
    stereo-capable effect (stereo_pan, reverb, binaural_beat_hz) is active.
    """
    return synthesize_speech(
        text=text,
        voice_name=voice_name,
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
        lowpass_cutoff_hz=lowpass_cutoff_hz,
        highpass_cutoff_hz=highpass_cutoff_hz,
    )


@mcp.tool()
def tool_design_voice(
    text: Annotated[
        str,
        Field(min_length=1, description="Text to speak (same as synthesize_speech)."),
    ],
    instruct: Annotated[
        str,
        Field(
            description=(
                "Voice design keywords (REQUIRED for this tool). English OR Chinese, "
                "comma+space separated. Examples: 'whisper, female, low pitch' (ASMR), "
                "'male, young adult, british accent', '女，低音调' (Chinese)."
            ),
        ),
    ],
    language: Annotated[str, Field(description="Target language. Default: 'Italian'.")] = "Italian",
    num_step: Annotated[int, Field(ge=4, le=100, description="Diffusion steps. 32 = default.")] = 32,
    guidance_scale: Annotated[float, Field(ge=0.0, le=10.0, description="CFG scale. 2.0 default.")] = 2.0,
    speed: Annotated[float, Field(ge=0.25, le=4.0, description="Speech rate. 1.0 = normal.")] = 1.0,
    output_path: Annotated[Optional[str], Field(description="Output WAV path. Default: data/outputs/<ts>.wav.")] = None,
    # ASMR params (same as synthesize_speech)
    stereo_pan: _PAN = None,
    period_s: Annotated[float, Field(ge=0.1, le=20.0, description="L<->R period in seconds. ASMR sweet spot: 2-3s.")] = 2.0,
    silence_padding_ms: _PADDING_MS = 0,
    reverb: _REVERB = None,
    reverb_damping: Annotated[float, Field(ge=0.0, le=1.0, description="Reverb HF damping. 0.5 default.")] = 0.5,
    binaural_beat_hz: _BINAURAL_HZ = 0.0,
    lowpass_cutoff_hz: _LOWPASS_HZ = 0.0,
    highpass_cutoff_hz: _HIGHPASS_HZ = 60.0,
) -> dict:
    """Generate speech using voice design keywords only (no cloned voice).

    Same ASMR params as synthesize_speech. Use this when you want a one-off voice
    without persisting a profile. For reusable voices, clone first with
    clone_voice_from_audio then call synthesize_speech with voice_name.
    """
    return design_voice(
        text=text,
        instruct=instruct,
        language=language,
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
        lowpass_cutoff_hz=lowpass_cutoff_hz,
        highpass_cutoff_hz=highpass_cutoff_hz,
    )


@mcp.tool()
def tool_list_voices() -> list[dict]:
    """List all saved voice profiles.

    Returns: list of dicts, each with 'name', 'description', 'language', 'source', 'created_at'.
    Use this to discover available voice_name values before calling synthesize_speech.
    """
    return list_voices_tool()


@mcp.tool()
def tool_get_voice_info(
    voice_name: Annotated[
        str,
        Field(description="Voice name to inspect (from list_voices output)."),
    ],
) -> dict:
    """Get full metadata of a single saved voice profile: name, description, language, ref_audio_path, ref_text, created_at.

    Use this to verify a profile exists and inspect its reference audio path
    before calling synthesize_speech with that voice_name.
    """
    return get_voice_info(voice_name)


@mcp.tool()
def tool_delete_voice(
    voice_name: Annotated[
        str,
        Field(description="Voice name to delete. IRREVERSIBLE - removes profile + reference audio."),
    ],
) -> dict:
    """Delete a voice profile and its reference audio file. Cannot be undone.

    Use this to free disk space or clean up test voices. The voice will no longer
    appear in list_voices and synthesize_speech with this voice_name will fail.
    """
    return delete_voice(voice_name)


def main() -> None:
    log.info("starting voice-studio MCP server (stdio)")
    mcp.run()


if __name__ == "__main__":
    main()
