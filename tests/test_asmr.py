"""Tests for ASMR DSP pipeline (core/asmr.py)."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from mcp_voice_studio.core.asmr import (
    apply_asmr_pipeline,
    apply_binaural_beat,
    apply_highpass,
    apply_lowpass,
    apply_reverb,
    apply_stereo_pan,
    inject_silence_padding,
    read_wav,
    write_wav,
)


# ----- Helpers ------------------------------------------------------------

def _write_silent_wav(path: Path, duration_s: float = 1.0, sr: int = 24000, ch: int = 1) -> None:
    """Write a 16-bit PCM WAV filled with a 440 Hz sine wave (audible)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    sig = 0.3 * np.sin(2 * np.pi * 440 * t)
    if ch == 2:
        sig = np.stack([sig, sig], axis=1).reshape(-1)
    pcm = (np.clip(sig, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ----- WAV I/O ------------------------------------------------------------

def test_read_write_wav_mono(tmp_path: Path) -> None:
    p = tmp_path / "mono.wav"
    _write_silent_wav(p, duration_s=0.5, ch=1)
    samples, sr = read_wav(p)
    assert sr == 24000
    assert samples.ndim == 1
    assert len(samples) == 24000 // 2
    out = tmp_path / "out_mono.wav"
    write_wav(out, samples, sr)
    assert out.exists()
    s2, sr2 = read_wav(out)
    np.testing.assert_allclose(samples, s2, atol=1e-4)
    assert sr == sr2


def test_read_write_wav_stereo(tmp_path: Path) -> None:
    p = tmp_path / "stereo.wav"
    _write_silent_wav(p, duration_s=0.5, ch=2)
    samples, sr = read_wav(p)
    assert samples.ndim == 2 and samples.shape[1] == 2
    out = tmp_path / "out_stereo.wav"
    write_wav(out, samples, sr)
    s2, _ = read_wav(out)
    np.testing.assert_allclose(samples, s2, atol=1e-4)


# ----- Lowpass ------------------------------------------------------------

def test_lowpass_cuts_high_freq(tmp_path: Path) -> None:
    # White noise
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(24000).astype(np.float32) * 0.3
    # 200 Hz lowpass should kill most energy above 200 Hz
    out = apply_lowpass(noise, 24000, cutoff_hz=200.0)
    assert out.shape == noise.shape
    # High-freq content: FFT energy above 1000 Hz should be much smaller
    spec_in = np.abs(np.fft.rfft(noise))
    spec_out = np.abs(np.fft.rfft(out))
    high_in = spec_in[1000:].sum()
    high_out = spec_out[1000:].sum()
    assert high_out < high_in * 0.1  # at least 90% reduction


def test_lowpass_zero_cutoff_is_passthrough() -> None:
    sig = np.ones(100, dtype=np.float32)
    out = apply_lowpass(sig, 24000, cutoff_hz=0)
    np.testing.assert_array_equal(sig, out)


# ----- Highpass (DC / sub-bass cleanup) ---------------------------------

def test_highpass_removes_dc_offset() -> None:
    """A signal with a DC offset must come out centered on 0 after highpass."""
    sig = np.ones(24000, dtype=np.float32) * 0.5  # constant 0.5
    out = apply_highpass(sig, 24000, cutoff_hz=60.0)
    assert float(np.mean(out)) < 0.01, f"DC not removed: mean={float(np.mean(out)):.4f}"
    # Output amplitude should be near zero (DC was the only thing in the signal)
    assert float(np.max(np.abs(out))) < 0.05


def test_highpass_preserves_speech_band() -> None:
    """A 440Hz tone must pass through mostly intact (filtfilt has edge transients)."""
    t = np.arange(24000, dtype=np.float32) / 24000
    sig = 0.3 * np.sin(2 * np.pi * 440 * t)
    out = apply_highpass(sig, 24000, cutoff_hz=60.0)
    # Check the middle portion (skip 2000 samples at each end - filtfilt edges)
    np.testing.assert_allclose(out[2000:22000], sig[2000:22000], atol=0.01)
    # Overall RMS should be close (highpass does not destroy the signal)
    rms_in = float(np.sqrt(np.mean(sig[2000:22000] ** 2)))
    rms_out = float(np.sqrt(np.mean(out[2000:22000] ** 2)))
    assert abs(rms_out - rms_in) < 0.01, f"RMS mismatch: in={rms_in:.4f} out={rms_out:.4f}"


def test_highpass_zero_cutoff_is_passthrough() -> None:
    sig = np.ones(100, dtype=np.float32)
    out = apply_highpass(sig, 24000, cutoff_hz=0)
    np.testing.assert_array_equal(sig, out)


# ----- Stereo pan ---------------------------------------------------------

def test_pan_center_promotes_mono_to_stereo() -> None:
    mono = np.ones(1000, dtype=np.float32)
    out = apply_stereo_pan(mono, mode="center")
    assert out.shape == (1000, 2)
    np.testing.assert_array_equal(out[:, 0], out[:, 1])


def test_pan_hard_left() -> None:
    mono = np.ones(1000, dtype=np.float32)
    out = apply_stereo_pan(mono, mode="L")
    assert out.shape == (1000, 2)
    assert np.all(out[:, 0] > 0.5)  # L has signal
    assert np.allclose(out[:, 1], 0)  # R is silent


def test_pan_hard_right() -> None:
    mono = np.ones(1000, dtype=np.float32)
    out = apply_stereo_pan(mono, mode="R")
    assert np.allclose(out[:, 0], 0)
    assert np.all(out[:, 1] > 0.5)


def test_pan_alternating_l_r() -> None:
    mono = np.ones(24000, dtype=np.float32)  # 1 second
    out = apply_stereo_pan(mono, mode="L<->R", period_s=0.5)
    assert out.shape == (24000, 2)
    # First half: L should be active, R silent
    first_half_l = np.abs(out[:12000, 0]).mean()
    first_half_r = np.abs(out[:12000, 1]).mean()
    assert first_half_l > first_half_r * 10
    # Second half: R should be active, L silent
    second_half_l = np.abs(out[12000:, 0]).mean()
    second_half_r = np.abs(out[12000:, 1]).mean()
    assert second_half_r > second_half_l * 10


def test_pan_sweep_l_to_r() -> None:
    mono = np.ones(24000, dtype=np.float32)
    out = apply_stereo_pan(mono, mode="L->R", period_s=1.0)
    # Middle of first period (t=0.5) should be roughly equal L and R
    mid_l = np.abs(out[12000 - 100:12000 + 100, 0]).mean()
    mid_r = np.abs(out[12000 - 100:12000 + 100, 1]).mean()
    assert 0.4 < mid_l / (mid_l + mid_r) < 0.6


# ----- Reverb -------------------------------------------------------------

def test_reverb_none_is_passthrough() -> None:
    sig = np.random.default_rng(0).standard_normal(1000).astype(np.float32) * 0.3
    out = apply_reverb(sig, 24000, mode="none")
    np.testing.assert_array_equal(sig, out)


def test_reverb_small_room_adds_tail() -> None:
    # Impulse response (1 sample click)
    sig = np.zeros(24000, dtype=np.float32)
    sig[0] = 1.0
    out = apply_reverb(sig, 24000, mode="small_room", mix=1.0)
    # Tail should be non-zero
    assert np.abs(out[1000:]).sum() > 0


def test_reverb_damping_reduces_high_freq_energy() -> None:
    """Damping must attenuate high frequencies in the reverb tail (L38).

    Without damping (damping=0) the Schroeder reverb tail keeps its original
    spectral balance and can sound metallic. With damping>0 the HF energy in
    the tail is much lower.
    """
    # A short click excites all frequencies
    sig = np.zeros(24000, dtype=np.float32)
    sig[0] = 1.0
    # Capture only the tail (after the click)
    out_dry = apply_reverb(sig, 24000, mode="small_room", mix=1.0, damping=0.0)[1000:]
    out_damp = apply_reverb(sig, 24000, mode="small_room", mix=1.0, damping=0.7)[1000:]
    # FFT of the tail
    spec_dry = np.abs(np.fft.rfft(out_dry))
    spec_damp = np.abs(np.fft.rfft(out_damp))
    # HF energy ratio (bins > 4000Hz at sr=24000 -> bin > 4000)
    hf_dry = spec_dry[4000:].sum()
    hf_damp = spec_damp[4000:].sum()
    assert hf_damp < hf_dry * 0.5, (
        f"damping did not reduce HF: damp={hf_damp:.2f} vs dry={hf_dry:.2f}"
    )


# ----- Binaural beat ------------------------------------------------------

def test_binaural_beat_off_is_passthrough() -> None:
    sig = np.random.default_rng(0).standard_normal((1000, 2)).astype(np.float32) * 0.3
    out = apply_binaural_beat(sig, 24000, beat_hz=0)
    np.testing.assert_array_equal(sig, out)


def test_binaural_beat_adds_tones() -> None:
    sig = np.zeros((24000, 2), dtype=np.float32)  # 1s silence
    out = apply_binaural_beat(sig, 24000, beat_hz=4.0, carrier_hz=200.0, amplitude=0.1)
    # Both channels should have energy
    assert np.abs(out[:, 0]).sum() > 1
    assert np.abs(out[:, 1]).sum() > 1
    # FFT: L should have a peak near 200 Hz, R near 204 Hz
    fft_l = np.abs(np.fft.rfft(out[:, 0]))
    fft_r = np.abs(np.fft.rfft(out[:, 1]))
    # Bin resolution = sr/n = 1 Hz, so bin 200 is 200 Hz
    assert fft_l[200] > fft_l.mean() * 5
    assert fft_r[204] > fft_r.mean() * 5


def test_binaural_beat_promotes_mono_to_stereo() -> None:
    sig = np.zeros(1000, dtype=np.float32)
    out = apply_binaural_beat(sig, 24000, beat_hz=4.0)
    assert out.shape == (1000, 2)


def test_binaural_beat_default_amplitude_is_subaudible() -> None:
    """Regression: default amplitude must stay sub-audible.

    With amplitude=0.04 the 200Hz carrier becomes a clearly audible drone
    (12% of total energy, L37). With amplitude=0.005 the carrier is at -46dBFS
    but still perceptible on quiet material. Default 0.0005 is true sub-audible
    (-66dBFS).
    """
    sig = np.zeros((24000, 2), dtype=np.float32)  # 1s silence
    out = apply_binaural_beat(sig, 24000, beat_hz=8.0, carrier_hz=200.0)
    # Peak amplitude must stay well below perception threshold
    assert float(np.max(np.abs(out))) < 0.002, (
        f"default binaural amplitude too loud: peak={float(np.max(np.abs(out))):.4f}"
    )
    # FFT: 200Hz bin should exist (function works) but be tiny
    fft_l = np.abs(np.fft.rfft(out[:, 0]))
    fft_r = np.abs(np.fft.rfft(out[:, 1]))
    # 200Hz bin must be < 0.1% of dynamic range
    assert fft_l[200] < 100, f"200Hz L bin too loud: {fft_l[200]:.1f}"
    assert fft_r[208] < 100, f"208Hz R bin too loud: {fft_r[208]:.1f}"


# ----- Silence padding ----------------------------------------------------

def test_padding_zero_is_passthrough() -> None:
    sig = np.ones((1000, 2), dtype=np.float32)
    out = inject_silence_padding(sig, 24000, "Ciao. Come stai.", padding_ms=0)
    np.testing.assert_array_equal(sig, out)


def test_padding_inserts_silence() -> None:
    # 1 second stereo, then padded
    sig = np.ones((24000, 2), dtype=np.float32) * 0.5
    text = "Prima frase. Seconda frase. Terza frase."  # 12 + 14 + 12 = 38 chars
    out = inject_silence_padding(sig, 24000, text, padding_ms=1000)
    # Expected: 24k + 2*24k = 72k samples (2 boundaries x 1000ms @ 24kHz)
    assert out.shape[0] == 24000 + 2 * 24000
    # Silence regions (weighted boundaries processed right-to-left, see asmr.py):
    #   weights = [12/38, 14/38, 12/38]
    #   boundary 2 = round(26/38 * 24000) = 16421 -> silence B at [16421:40421] in 48k
    #   boundary 1 = round(12/38 * 24000) = 7578  -> silence A at [7578:31578] in 72k
    # After iter 2, silence B is shifted by +24000 (size of silence A): [40421:64421]
    assert np.allclose(out[7578:31578, :], 0), "silence A missing at weighted boundary 1"
    assert np.allclose(out[40421:64421, :], 0), "silence B missing at weighted boundary 2"
    # Signal regions preserved
    assert np.allclose(out[:7578, :], 0.5), "head signal lost"
    assert np.allclose(out[31578:40421, :], 0.5), "mid signal lost"
    assert np.allclose(out[64421:72000, :], 0.5), "tail signal lost"


def test_padding_single_sentence_no_change() -> None:
    sig = np.ones((24000, 2), dtype=np.float32)
    out = inject_silence_padding(sig, 24000, "Una sola frase.", padding_ms=500)
    np.testing.assert_array_equal(sig, out)


# ----- Full pipeline ------------------------------------------------------

def test_full_pipeline_mono_to_stereo(tmp_path: Path) -> None:
    src = tmp_path / "src.wav"
    _write_silent_wav(src, duration_s=2.0, ch=1)
    dst = tmp_path / "out.wav"
    text = "Ciao. Come stai. Tutto bene."
    sr, ch = apply_asmr_pipeline(
        input_wav=src, output_wav=dst, text=text, sample_rate=24000,
        stereo_pan="L<->R", silence_padding_ms=500, reverb="small_room",
        binaural_beat_hz=4.0, lowpass_cutoff_hz=6000.0,
    )
    assert dst.exists()
    assert sr == 24000
    assert ch == 2  # stereo due to panning
    samples, _ = read_wav(dst)
    assert samples.ndim == 2 and samples.shape[1] == 2


def test_full_pipeline_no_asmr_is_passthrough_mono(tmp_path: Path) -> None:
    src = tmp_path / "src.wav"
    _write_silent_wav(src, duration_s=0.5, ch=1)
    dst = tmp_path / "out.wav"
    sr, ch = apply_asmr_pipeline(
        input_wav=src, output_wav=dst, text="Ciao.", sample_rate=24000,
    )
    assert ch == 1  # stays mono when no panning
