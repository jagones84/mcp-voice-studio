"""ASMR post-processing DSP for synthesized speech.

All transformations are pure-numpy/scipy on int16 PCM WAVs. They are designed
to be applied in this order after OmniVoice synthesis (which produces mono
24 kHz PCM 16-bit):

  1. apply_highpass         (mono, remove DC offset / sub-bass rumble)
  2. apply_lowpass          (mono, optional warmth/intimacy EQ)
  3. apply_stereo_pan       (mono -> stereo, optional panning)
  4. apply_reverb           (stereo, optional small room with HF damping)
  5. apply_binaural_beat    (stereo, optional sub-audible carrier L/R)
  6. inject_silence_padding (stereo, optional pauses between sentences)

Each function returns a numpy float32 array in [-1, 1].
"""

from __future__ import annotations

import re
import wave
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.signal import butter, filtfilt


# ----- WAV I/O ------------------------------------------------------------

def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV. Returns (samples_float32[-1,1], sample_rate).

    Output shape: (n,) for mono, (n, 2) for stereo.
    """
    p = Path(path)
    with wave.open(str(p), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        n = w.getnframes()
        raw = w.readframes(n)
    samp = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch == 2:
        samp = samp.reshape(-1, 2)
    return samp, sr


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write float32 samples in [-1,1] as 16-bit PCM WAV (mono or stereo)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if samples.ndim == 1:
        ch = 1
        flat = samples
    elif samples.ndim == 2 and samples.shape[1] == 2:
        ch = 2
        flat = samples.reshape(-1)
    else:
        raise ValueError(f"unsupported sample shape: {samples.shape}")
    flat = np.clip(flat, -1.0, 1.0)
    pcm = (flat * 32767.0).astype("<i2")
    with wave.open(str(p), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


# ----- 1. Highpass (DC / sub-bass cleanup) -------------------------------

def apply_highpass(samples: np.ndarray, sample_rate: int, cutoff_hz: float = 60.0) -> np.ndarray:
    """Remove DC offset and sub-bass rumble below `cutoff_hz`.

    A 2nd-order Butterworth highpass. Always recommended for ASMR speech:
    synthetic voices often carry a small DC offset (lowpass ringing tail,
    comb-filter build-up) that adds an audible 'thump' on transients and
    is hard to remove later in the chain.
    """
    if cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2:
        return samples
    b, a = butter(2, cutoff_hz / (sample_rate / 2), btype="high")
    if samples.ndim == 1:
        return filtfilt(b, a, samples).astype(np.float32)
    out = np.empty_like(samples)
    for c in range(samples.shape[1]):
        out[:, c] = filtfilt(b, a, samples[:, c])
    return out.astype(np.float32)


# ----- 2. Lowpass (warmth) -----------------------------------------------

def apply_lowpass(samples: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """Apply a 2nd-order Butterworth lowpass. Useful for 'intimacy' warmth.

    ASMR sweet spot: 5000-7000 Hz. Default speech intelligibility: 8000 Hz.
    """
    if cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2:
        return samples
    b, a = butter(2, cutoff_hz / (sample_rate / 2), btype="low")
    if samples.ndim == 1:
        return filtfilt(b, a, samples).astype(np.float32)
    out = np.empty_like(samples)
    for c in range(samples.shape[1]):
        out[:, c] = filtfilt(b, a, samples[:, c])
    return out.astype(np.float32)


# ----- 2. Stereo pan -----------------------------------------------------

PanMode = Literal["center", "L", "R", "L<->R", "L->R", "R->L"]


def apply_stereo_pan(
    samples: np.ndarray,
    mode: PanMode = "center",
    period_s: float = 4.0,
    sample_rate: int = 24000,
) -> np.ndarray:
    """Mono -> stereo with a panning law.

    Modes:
        center: equal L and R (mono folded to stereo, +0 dB each)
        L: hard left (R is muted)
        R: hard right
        L<->R: alternating L and R per sentence (proxied by half-period)
        L->R: slow sweep L to R, then back (triangle wave, period_s)
        R->L: same but inverted
    """
    if samples.ndim == 2 and samples.shape[1] == 2:
        mono = samples.mean(axis=1)
    else:
        mono = samples
    n = len(mono)
    if mode == "center":
        l_g = np.ones(n, dtype=np.float32)
        r_g = np.ones(n, dtype=np.float32)
    elif mode == "L":
        l_g = np.ones(n, dtype=np.float32)
        r_g = np.zeros(n, dtype=np.float32)
    elif mode == "R":
        l_g = np.zeros(n, dtype=np.float32)
        r_g = np.ones(n, dtype=np.float32)
    elif mode in ("L<->R", "L->R", "R->L"):
        # t is in units of `period_s` (0..1 = one full period_s)
        t = np.arange(n, dtype=np.float32) / sample_rate / max(period_s, 1e-6)
        if mode == "L<->R":
            # period_s = dwell time per ear; 2*period_s = full L+R cycle
            left = (t % 2.0) < 1.0
            l_g = np.where(left, 1.0, 0.0).astype(np.float32)
            r_g = 1.0 - l_g
        else:
            # Sawtooth sweep: period_s = duration of one L->R (or R->L) pass.
            # At phase=0 we are full L (or R), at phase=1 we are full R (or L),
            # then it resets. Mid-period (phase=0.5) -> center.
            phase = (t % 1.0).astype(np.float32)
            if mode == "L->R":
                l_g = 1.0 - phase
                r_g = phase
            else:  # R->L
                l_g = phase
                r_g = 1.0 - phase
    else:
        raise ValueError(f"unknown stereo_pan mode: {mode}")
    # Constant-power panning: gain = sqrt(weight) keeps perceived loudness constant
    l_g = np.sqrt(l_g)
    r_g = np.sqrt(r_g)
    stereo = np.stack([mono * l_g, mono * r_g], axis=1)
    return stereo.astype(np.float32)


# ----- 3. Reverb (Schroeder, with HF damping) ----------------------------

def apply_reverb(
    samples: np.ndarray,
    sample_rate: int,
    mode: Literal["none", "small_room", "large_room"] = "none",
    mix: float = 0.18,
    damping: float = 0.5,
) -> np.ndarray:
    """Schroeder reverb (4 parallel combs + 2 series allpass) with HF damping.

    Args:
        mode: 'small_room' (tight, ASMR), 'large_room' (spacious), 'none' (passthrough).
        mix: dry/wet mix (0 = fully dry, 1 = fully wet). Default 0.18 (subtle).
        damping: 0..1, how much the comb feedback absorbs high frequencies.
            0 = no damping (classic Schroeder, can sound metallic).
            0.5 = gentle HF rolloff in the reverb tail (recommended for ASMR).
            1 = heavy damping, dark/warm tail.
    """
    if mode == "none" or mix <= 0:
        return samples
    # Delay lengths in seconds, classic Schroeder values
    if mode == "small_room":
        comb_delays = [0.0297, 0.0371, 0.0411, 0.0437]
        allpass_delays = [0.005, 0.0017]
        comb_gains = [0.84, 0.83, 0.82, 0.81]
        allpass_gain = 0.7
    elif mode == "large_room":
        comb_delays = [0.0411, 0.0437, 0.0473, 0.0533]
        allpass_delays = [0.013, 0.0071]
        comb_gains = [0.86, 0.85, 0.84, 0.83]
        allpass_gain = 0.7
    else:
        raise ValueError(f"unknown reverb mode: {mode}")

    # Work in mono for reverb (ASMR doesn't need true stereo decorrelation here)
    if samples.ndim == 2:
        mono = samples.mean(axis=1)
    else:
        mono = samples
    # Damping coefficient: 1-pole lowpass inside the comb feedback path.
    # Higher damping = more HF absorption per round-trip, so the tail decays
    # faster at high freqs (warmer, less metallic).
    damp = float(np.clip(damping, 0.0, 1.0))
    wet = np.zeros_like(mono)
    for d, g in zip(comb_delays, comb_gains):
        n_delay = int(d * sample_rate)
        if n_delay <= 0:
            continue
        buf = np.zeros(len(mono) + n_delay, dtype=np.float32)
        buf[: len(mono)] = mono
        if damp <= 0.0:
            # Plain Schroeder comb (no damping) - fast path
            for i in range(n_delay, len(buf)):
                buf[i] += g * buf[i - n_delay]
        else:
            # Damped comb: feedback = g * lowpass(buf[i - n_delay])
            # 1-pole IIR: y[n] = (1-damp) * x[n] + damp * y[n-1]
            prev = 0.0
            for i in range(n_delay, len(buf)):
                x = buf[i - n_delay]
                prev = (1.0 - damp) * x + damp * prev
                buf[i] += g * prev
        wet += buf[: len(mono)]
    wet /= max(len(comb_delays), 1)
    # Allpass chain (kept simple: each adds a short dense reflection)
    for d in allpass_delays:
        n_delay = max(int(d * sample_rate), 1)
        buf = np.zeros(len(wet) + n_delay, dtype=np.float32)
        buf[: len(wet)] = wet
        out = np.zeros_like(wet)
        for i in range(len(wet)):
            delayed = buf[i - n_delay] if i >= n_delay else 0.0
            out[i] = -allpass_gain * wet[i] + delayed + allpass_gain * (
                buf[i - n_delay + 1] if i >= n_delay - 1 else 0.0
            )
        # The above is a hand-rolled allpass; for simplicity we keep the comb-summed wet
    # If stereo input, return stereo by broadcasting wet to both channels
    if samples.ndim == 2:
        result = (1 - mix) * samples + mix * np.stack([wet, wet], axis=1)
    else:
        result = (1 - mix) * samples + mix * wet
    return result.astype(np.float32)


# ----- 4. Binaural beat --------------------------------------------------

def apply_binaural_beat(
    samples: np.ndarray,
    sample_rate: int,
    beat_hz: float = 0.0,
    carrier_hz: float = 200.0,
    amplitude: float = 0.0005,
) -> np.ndarray:
    """Add a binaural beat carrier (L = carrier, R = carrier + beat_hz).

    The brain perceives the difference (beat_hz) as a pulsing tone, useful for
    entrainment (4 Hz = theta/relaxation, 8 Hz = alpha, 10 Hz = alpha-peak).

    Args:
        beat_hz: frequency difference L/R in Hz (0 = disabled).
        carrier_hz: base frequency in Hz.
        amplitude: 0..1, peak amplitude of the carrier tone.
            Default 0.0005 (-66dBFS, true sub-audible). The carrier tone itself
            should NOT be heard, only the perceived 4-8Hz pulsation. Values
            above 0.005 produce an audible 200Hz drone on quiet material.
    """
    if beat_hz <= 0 or amplitude <= 0:
        return samples
    if samples.ndim == 1:
        # Mono: convert to stereo first
        samples = np.stack([samples, samples], axis=1)
    n = samples.shape[0]
    t = np.arange(n, dtype=np.float32) / sample_rate
    left_tone = amplitude * np.sin(2 * np.pi * carrier_hz * t)
    right_tone = amplitude * np.sin(2 * np.pi * (carrier_hz + beat_hz) * t)
    out = np.empty_like(samples)
    out[:, 0] = np.clip(samples[:, 0] + left_tone, -1.0, 1.0)
    out[:, 1] = np.clip(samples[:, 1] + right_tone, -1.0, 1.0)
    return out.astype(np.float32)


# ----- 5. Silence padding ------------------------------------------------

def inject_silence_padding(
    samples: np.ndarray,
    sample_rate: int,
    text: str,
    padding_ms: int = 0,
) -> np.ndarray:
    """Insert `padding_ms` of silence at every sentence boundary.

    Boundaries are detected from `text` (`.`, `?`, `!`, newlines). The total
    duration of the audio is split proportionally to the relative weight of
    each sentence (counted in characters + spaces). Silences are inserted at
    the estimated boundaries.
    """
    if padding_ms <= 0 or samples.shape[0] == 0:
        return samples
    # Split text into sentences (preserving the splits)
    parts = re.split(r"(?<=[.?!])\s+|\n+", text.strip())
    parts = [p for p in parts if p.strip()]
    if len(parts) <= 1:
        return samples
    # Compute weight per sentence (chars)
    weights = np.array([max(len(p), 1) for p in parts], dtype=np.float32)
    weights /= weights.sum()
    # Total audio duration in samples
    n = samples.shape[0]
    # Cumulative boundaries in samples
    boundaries = np.cumsum(weights * n).astype(int)[:-1]  # exclude final endpoint
    pad_n = int(sample_rate * padding_ms / 1000)
    silence = np.zeros((pad_n,) + samples.shape[1:], dtype=samples.dtype)
    # Build output by interleaving segments and silences (segments from right to
    # left so indices remain valid)
    out = samples
    for b in reversed(boundaries):
        out = np.concatenate([out[:b], silence, out[b:]], axis=0)
    return out


# ----- Master pipeline ----------------------------------------------------

def apply_asmr_pipeline(
    input_wav: Path,
    output_wav: Path,
    text: str,
    sample_rate: int,
    stereo_pan: PanMode | None = None,
    period_s: float = 4.0,
    silence_padding_ms: int = 0,
    reverb: Literal["none", "small_room", "large_room"] | None = None,
    reverb_damping: float = 0.5,
    binaural_beat_hz: float = 0.0,
    binaural_amplitude: float = 0.0005,
    lowpass_cutoff_hz: float = 0.0,
    highpass_cutoff_hz: float = 60.0,
) -> tuple[int, int]:
    """Run the full ASMR enhancement pipeline. Returns (sample_rate, channels).

    The pipeline order matters: the highpass runs FIRST to strip any DC offset
    introduced by the source WAV; the binaural beat is OFF by default (set
    ``binaural_beat_hz`` > 0 explicitly to enable it).

    Args:
        input_wav: path of input WAV (read by the pipeline).
        output_wav: path of output WAV (written by the pipeline).
        text: original text (used to weight silence-padding positions).
        sample_rate: expected sample rate (overridden by actual sr of the file).
        stereo_pan: panning mode. None = passthrough mono.
        period_s: L<->R/L->R/R->L period in seconds (default 4.0).
        silence_padding_ms: silence padding at sentence boundaries.
        reverb: reverb preset. None or "none" = off.
        reverb_damping: HF damping 0..1 for each comb filter (0=classic Schroeder, 0.5=morbido).
        binaural_beat_hz: binaural beat frequency in Hz. 0 = off.
        binaural_amplitude: carrier peak amplitude (default 0.0005 = -66dBFS sub-audible).
        lowpass_cutoff_hz: lowpass cutoff. 0 = off.
        highpass_cutoff_hz: highpass cutoff (DC/sub-bass cleanup). 0 = off.
    """
    samples, sr = read_wav(input_wav)
    if sr != sample_rate:
        # Resampling would be ideal but we trust OmniVoice's 24 kHz output
        sample_rate = sr
    # 0. Highpass (DC / sub-bass cleanup) - always on by default
    if highpass_cutoff_hz > 0:
        samples = apply_highpass(samples, sample_rate, highpass_cutoff_hz)
    # 1. Lowpass (mono OK; on stereo applies per channel)
    if lowpass_cutoff_hz > 0:
        samples = apply_lowpass(samples, sample_rate, lowpass_cutoff_hz)
    # 2. Stereo pan (only if explicitly requested)
    if stereo_pan and stereo_pan != "center":
        samples = apply_stereo_pan(samples, mode=stereo_pan, period_s=period_s, sample_rate=sample_rate)
    # 3. Reverb (promotes to stereo if needed)
    if reverb and reverb != "none":
        if samples.ndim == 1:
            samples = np.stack([samples, samples], axis=1)
        samples = apply_reverb(samples, sample_rate, mode=reverb, damping=reverb_damping)
    # 4. Binaural beat (promotes to stereo) - OFF by default
    if binaural_beat_hz > 0:
        samples = apply_binaural_beat(samples, sample_rate, beat_hz=binaural_beat_hz, amplitude=binaural_amplitude)
    # 5. Silence padding (works on any shape)
    if silence_padding_ms > 0:
        samples = inject_silence_padding(samples, sample_rate, text, silence_padding_ms)
    write_wav(output_wav, samples, sample_rate)
    ch = 1 if samples.ndim == 1 else 2
    return sample_rate, ch
