# mcp-voice-studio

**MCP server wrapper for [VoiceStudio (debpalash)](https://github.com/debpalash/VoiceStudio) — local voice cloning & TTS via the [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) engine.**

Exposes 6 tools to any MCP client (Trae IDE, Claude Desktop, etc.):

| Tool | What it does |
|---|---|
| `clone_voice_from_audio` | Save a voice profile from 5-30s of reference audio + transcript |
| `synthesize_speech` | Generate speech with a cloned voice OR voice design keywords (or both) |
| `design_voice` | Generate speech using only voice design (no cloned voice) |
| `list_voices` | List all saved voice profiles |
| `get_voice_info` | Get metadata of a single voice profile |
| `delete_voice` | Delete a voice profile and its ref audio |

**Free & local.** No API costs, all inference runs on your GPU (tested on NVIDIA GB10 sm_120 / DGX Spark). Uses [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (Apache 2.0) + [higgs-audio-v2-tokenizer](https://huggingface.co/eustlb/higgs-audio-v2-tokenizer) (MIT).

---

## Why this exists

VoiceStudio is a great local ElevenLabs alternative (AGPL-3.0, 1.3k+ stars) but it ships as a CLI + Gradio UI. This wrapper exposes it as an **MCP server** so you can:

- Call voice cloning / TTS from any MCP-compatible agent
- Programmatically manage voice profiles (create, list, delete)
- Reuse a single VoiceStudio venv across multiple tools without spawning Gradio

---

## Install (DGX Spark / aarch64+CUDA)

### 1. Clone this repo + VoiceStudio (sibling)

```bash
mkdir -p ~/Repositories
cd ~/Repositories
git clone https://github.com/debpalash/VoiceStudio.git
git clone <this-repo>   mcp-voice-studio
```

### 2. Build VoiceStudio venv (one-time, ~5min)

Follow the procedure in `.agent/README-voice-studio-dgx.md` (L24-L28):

```bash
cd VoiceStudio
# patch pyproject (remove aarch64 marker, pin torch 2.11+cu128, etc.)
bash /home/jagones/Repositories/trash/patch_pyproject.sh
bash /home/jagones/Repositories/trash/fix_torchvision_pin.sh
# build venv
export PATH="$HOME/.local/bin:$PATH"
rm -rf .venv uv.lock
uv lock && uv sync --python 3.12
# install torchcodec + cu12 NPP
source .venv/bin/activate
bash /home/jagones/Repositories/trash/install_torchcodec.sh
bash /home/jagones/Repositories/trash/install_nvidia_cu12.sh
bash /home/jagones/Repositories/trash/upgrade_nvidia_cu128.sh
# verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA GB10
```

### 3. Build this MCP venv (lightweight, ~30s)

```bash
cd ../mcp-voice-studio
cp .env.template .env
# edit .env: set HF_TOKEN (free at https://huggingface.co/settings/tokens)
export PATH="$HOME/.local/bin:$PATH"
uv sync
```

The MCP venv reuses the heavy torch+CUDA libs from VoiceStudio's venv via the engine wrapper (`mcp_voice_studio/core/engine.py`). The MCP venv only needs `mcp` + `pydantic`.

### 4. Test

```bash
uv run python -c "from mcp_voice_studio.server import mcp; print('tools:', [t.name for t in mcp._tool_manager._tools.values()])"
```

Expected:
```
tools: ['tool_clone_voice_from_audio', 'tool_synthesize_speech', 'tool_design_voice', 'tool_list_voices', 'tool_get_voice_info', 'tool_delete_voice']
```

---

## Register in MCP clients

### Trae IDE (Windows)

Add to `.mcp.json` (project root or `~/.trae/mcp.json`):

```json
{
  "mcpServers": {
    "voice-studio": {
      "command": "ssh",
      "args": [
        "dgx",
        "cd /home/jagones/Repositories/mcp-voice-studio && /home/jagones/.local/bin/uv run --no-sync python -m mcp_voice_studio"
      ],
      "env": {
        "HF_TOKEN": "hf_xxx",
        "VOICESTUDIO_VENV": "/home/jagones/Repositories/VoiceStudio/.venv"
      }
    }
  }
}
```

Replace `dgx` with your SSH host alias, and `hf_xxx` with your real HF token (free).

### Claude Desktop

`claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "voice-studio": {
      "command": "ssh",
      "args": ["dgx", "cd /home/jagones/Repositories/mcp-voice-studio && /home/jagones/.local/bin/uv run --no-sync python -m mcp_voice_studio"]
    }
  }
}
```

---

## Usage examples (from an MCP client)

### Clone a voice from a 6-second sample

```
> Use clone_voice_from_audio to save a voice called "claudia_asmr"
> from /home/jagones/Repositories/VoiceStudio/inputs/asmr_sample.wav
> with ref_text "Hola, soy Claudia. Esta es una muestra de mi voz en estilo ASMR."
```

Tool response:
```json
{
  "status": "ok",
  "voice_name": "claudia_asmr",
  "profile_path": "/home/jagones/Repositories/mcp-voice-studio/mcp_voice_studio/data/voices/claudia_asmr/profile.json",
  "ref_audio_path": ".../data/voices/claudia_asmr/ref_audio.wav"
}
```

### Generate ASMR speech with the cloned voice

```
> Use synthesize_speech with voice_name="claudia_asmr", text="Benvenuto, chiudi gli occhi, fai un respiro profondo..."
```

Tool response:
```json
{
  "output_path": "/home/jagones/Repositories/mcp-voice-studio/mcp_voice_studio/data/outputs/synth_1757062500.wav",
  "duration_s": 24.04,
  "sample_rate": 24000,
  "channels": 1,
  "model": "k2-fsa/OmniVoice",
  "voice_name": "claudia_asmr",
  "generation_time_s": 47.3
}
```

### ASMR enhancements (post-synth DSP pipeline)

`synthesize_speech` and `design_voice` accept 5 optional ASMR parameters. All are applied as a deterministic post-processing pipeline on the generated mono WAV (numpy + scipy, no extra GPU). The output is **stereo** whenever any pan/reverb effect is active.

The pipeline runs in this order: **highpass (60Hz) → lowpass → stereo_pan → reverb (with HF damping) → binaural_beat → silence_padding**. The highpass is always on by default to remove DC offset and sub-bass rumble that synthetic voices can carry.

| Param | Type | Effect | ASMR sweet spot |
|---|---|---|---|
| `stereo_pan` | `center` \| `L` \| `R` \| `L<->R` \| `L->R` \| `R->L` | Stereo panning law (constant-power). `L<->R` = alternating L/R per `period_s`; `L->R`/`R->L` = sawtooth sweep. | `L<->R` with `period_s=2.0`–`3.0` |
| `silence_padding_ms` | int (0–5000) | Inserts ms of silence at every `.` `?` `!` boundary (position weighted by sentence length). | 400–800 ms |
| `reverb` | `none` \| `small_room` \| `large_room` | Schroeder reverb (4 comb + 2 allpass filters, 18 % mix) with HF damping filter (damping=0.5 default) on each comb to avoid the classic "metallic" ring. | `small_room` |
| `binaural_beat_hz` | float (0–40) | Adds a sine wave to L (200 Hz) and a slightly-detuned sine to R. Perceived as a brainwave entrainment tone. Amplitude is fixed at 0.0005 (-66dBFS, true sub-audible carrier) — only the 4–8 Hz pulsation is heard, never the 200 Hz tone itself. **OFF by default** to keep the output clean; pass a positive value to enable. | 4–8 Hz (theta-alpha) |
| `lowpass_cutoff_hz` | float (0–20000) | 2nd-order Butterworth lowpass for "warmth" / intimacy. | 5000–7000 Hz |

Example — full ASMR stack:

```
> Use synthesize_speech with:
    voice_name="claudia_asmr"
    text="Ascolta il mio respiro. Lascia andare ogni tensione. Sei al sicuro."
    stereo_pan="L<->R"
    silence_padding_ms=600
    reverb="small_room"
    lowpass_cutoff_hz=6500.0
```

Response:
```json
{
  "output_path": ".../synth_1757064500.wav",
  "duration_s": 8.83,
  "sample_rate": 24000,
  "channels": 2,
  "asmr_applied": [
    "lowpass(6500Hz)", "stereo_pan(L<->R)", "reverb(small_room)",
    "binaural_beat(6Hz)", "silence_padding(600ms)"
  ],
  "generation_time_s": 17.6
}
```

The `asmr_applied` list reports exactly which effects ran (so you can distinguish "nothing applied" from "applied but no audible effect"). Order in the pipeline: **lowpass → pan → reverb → binaural → silence padding**.

### Voice design without cloning

```
> Use design_voice with instruct="whisper, female, low pitch", text="Hello world"
```

### Combine cloning + design

```
> Use synthesize_speech with voice_name="claudia_asmr", instruct="whisper", text="..."
```

(cloned voice + extra style instruction)

### List / inspect / delete

```
> list_voices
> get_voice_info(voice_name="claudia_asmr")
> delete_voice(voice_name="claudia_asmr")
```

---

## Voice design keywords (OmniVoice)

Only these are accepted by OmniVoice's `--instruct` (case-sensitive, comma+space separated, English OR Chinese, never mix):

**English:** `american accent, australian accent, british accent, canadian accent, child, chinese accent, elderly, female, high pitch, indian accent, japanese accent, korean accent, low pitch, male, middle-aged, moderate pitch, portuguese accent, russian accent, teenager, very high pitch, very low pitch, whisper, young adult`

**Chinese (full-width comma ,):** `东北话，中年，中音调，云南话，低音调，儿童，四川话，女，宁夏话，少年，极低音调，极高音调，桂林话，河南话，济南话，甘肃话，男，石家庄话，老年，耳语，贵州话，陕西话，青岛话，青年，高音调`

For ASMR whisper: `whisper, female, low pitch`

---

## Architecture

```
mcp-voice-studio/
├── pyproject.toml                 # mcp + pydantic + numpy + scipy (no torch!)
├── mcp_voice_studio/
│   ├── server.py                  # FastMCP entry, registers 6 tools
│   ├── core/
│   │   ├── config.py              # paths, env (HF_TOKEN, VOICESTUDIO_VENV, CUDA_VISIBLE_DEVICES)
│   │   ├── models.py              # Pydantic: VoiceProfile, SynthRequest, SynthResult
│   │   ├── storage.py             # JSON+ref_audio persistence per voice profile
│   │   ├── engine.py              # auto-fallback: subprocess omnivoice-infer → inproc import
│   │   └── asmr.py                # DSP post-processor: pan, padding, reverb, binaural, lowpass
│   ├── tools/
│   │   ├── clone_voice.py         # clone_voice_from_audio
│   │   ├── synthesize.py          # synthesize_speech, design_voice
│   │   └── manage.py              # list_voices, get_voice_info, delete_voice
│   └── data/                      # gitignored runtime data
│       ├── voices/<name>/         # per-voice: ref_audio.wav, ref_text.txt, profile.json
│       ├── outputs/               # generated WAVs
│       ├── logs/                  # synthesis logs
│       └── inputs/                # default reference audio
├── tests/                         # pytest
├── examples/                      # usage examples + mcp_config.json
└── docs/                          # architecture, API
```

### Engine wrapper: auto-fallback

`mcp_voice_studio/core/engine.py` tries two execution modes:

1. **Subprocess** (default): spawns `uv run --no-sync omnivoice-infer ...` from VoiceStudio's venv.
   - Pro: isolates GPU state, most robust, no version coupling
   - Con: ~1s spawn overhead per call
2. **In-process** (fallback): adds VoiceStudio's site-packages to `sys.path` and imports `omnivoice` directly.
   - Pro: faster (no spawn)
   - Con: requires VoiceStudio venv to be importable in this venv

Auto-fallback: if subprocess fails because the binary is missing, switches to in-process.

### LD_LIBRARY_PATH for cu12 NPP

torchcodec (used by torchaudio 2.11) loads `libnppicc.so.12`. DGX Spark only has CUDA 13 system libs. The engine wrapper sets `LD_LIBRARY_PATH` to point at the **pip-installed** `nvidia-npp-cu12==12.4.1.87` (from VoiceStudio venv) BEFORE the system CUDA 13 path. This avoids the TLS clash caused by symlinks (see `.agent/README-voice-studio-dgx.md` L26).

---

## Standalone scripts (no MCP client needed)

Two scripts under `scripts/` let you run the cloning + ASMR pipeline from a terminal (useful for batch jobs or quick testing).

### `apply_asmr_effects.py` — DSP-only on an existing WAV (no GPU, runs anywhere)

```bash
python scripts/apply_asmr_effects.py INPUT.wav OUTPUT.wav --text "..." [options]
```

| Flag | Default | Effect |
|---|---|---|
| `--stereo-pan` | off | `center`/`L`/`R`/`L<->R`/`L->R`/`R->L` |
| `--period-s` | 2.0 | L<->R/L->R period in seconds |
| `--silence-padding-ms` | 0 | silence padding at sentence boundaries (0-5000) |
| `--reverb` | off | `none`/`small_room`/`large_room` |
| `--reverb-damping` | 0.5 | HF damping 0..1 (0=classic Schroeder) |
| `--binaural-beat-hz` | 0.0 | 0=off, 4-8=theta-alpha |
| `--binaural-amplitude` | 0.0005 | carrier peak (default -66dBFS sub-audible) |
| `--lowpass-cutoff-hz` | 0 | 0=off, 5000-7000 sweet spot |
| `--highpass-cutoff-hz` | 60.0 | 0=off, 60Hz = DC/sub-bass cleanup |

**Example:**
```bash
python scripts/apply_asmr_effects.py voice.wav out.wav \
    --text "Benvenuto. Chiudi gli occhi. Respira." \
    --stereo-pan "L<->R" --period-s 2.5 \
    --silence-padding-ms 600 \
    --reverb small_room \
    --lowpass-cutoff-hz 6500.0
```

### `clone_and_speak.py` — E2E: clone/design + synthesize + ASMR (needs DGX/GPU)

```bash
# Cloned voice
python scripts/clone_and_speak.py --voice claudia_asmr \
    --text "Ascolta il mio respiro. Sei al sicuro." \
    --out out.wav \
    --stereo-pan "L<->R" --silence-padding-ms 600 --reverb small_room

# Voice design (no clone)
python scripts/clone_and_speak.py --instruct "whisper, female, low pitch" \
    --text "Hello world" --out out.wav --reverb small_room
```

Both scripts call the same code paths the MCP tools use (`synthesize_speech` for TTS, `apply_asmr_pipeline` for DSP), so output is identical to what you'd get from an MCP client.

---

## License

MIT. See [LICENSE](./LICENSE) and [THIRD_PARTY_LICENSES.md](./THIRD_PARTY_LICENSES.md) for the full picture.

**TL;DR for publishing on GitHub:**
- This wrapper (this repo) is **MIT** — you can publish, fork, modify freely.
- **VoiceStudio is AGPL-3.0** — installed separately via `git clone`, NOT bundled here. Subprocess invocation does not extend AGPL to this wrapper (mere aggregation per FSF interpretation).
- **OmniVoice is Apache 2.0**, **higgs-audio-v2-tokenizer is MIT** — both downloaded from Hugging Face at runtime, NOT bundled.
- If you offer VoiceStudio's functionality as a network service, AGPL section 13 requires you to make the VoiceStudio source available to your users (irrelevant for local/personal use).
