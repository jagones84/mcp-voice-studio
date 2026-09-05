# Clone + Speak: end-to-end example

## 1. Clone a voice

From an MCP client:

> "Use `clone_voice_from_audio` to save a voice called `claudia_asmr` from `/home/jagones/Repositories/VoiceStudio/inputs/asmr_sample.wav`. The transcript is: 'Hola, soy Claudia. Esta es una muestra de mi voz en estilo ASMR.'"

The tool copies the audio to `data/voices/claudia_asmr/ref_audio.wav` and writes `profile.json`.

## 2. Generate ASMR speech in Italian

> "Use `synthesize_speech` with `voice_name='claudia_asmr'`, `text='Benvenuto. Chiudi gli occhi e rilassati.'`, `language='Italian'`."

The tool returns the WAV path and metadata (duration, sample rate, generation time).

## 3. Optional: layer voice design on top of the clone

> "Use `synthesize_speech` with `voice_name='claudia_asmr'`, `instruct='whisper, low pitch'`, `text='...'`."

You can combine a cloned voice (timbre) with voice design (style) by passing both `voice_name` and `instruct`.

## 4. List / inspect

> "List all saved voices with `list_voices`."

> "Get details of `claudia_asmr` with `get_voice_info`."

## 5. Cleanup

> "Delete the voice profile with `delete_voice(voice_name='claudia_asmr')`."
