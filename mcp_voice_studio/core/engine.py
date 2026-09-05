"""Engine wrapper for Omnivoice with auto-fallback: subprocess CLI -> Python import.

Two execution modes:
  - subprocess: spawn `uv run --no-sync omnivoice-infer` from VoiceStudio venv
                (slow spawn ~1s, but isolates GPU state, most robust)
  - inproc: import omnivoice directly, add VoiceStudio .venv to sys.path
            (faster, no spawn, but couples MCP venv to VoiceStudio torch versions)

The wrapper tries subprocess first. If it fails because the binary is missing,
falls back to inproc.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

from .config import (
    cuda_device,
    hf_token,
    logs_dir,
    voicestudio_root,
    voicestudio_venv,
)
from .models import SynthRequest, SynthResult


# ----- Mode detection ----------------------------------------------------

def _has_subprocess() -> bool:
    """Can we run omnivoice-infer as a subprocess (direct binary invocation)?"""
    bin_path = voicestudio_venv() / "bin" / "omnivoice-infer"
    return bin_path.exists()


def _has_inproc() -> bool:
    """Can we import omnivoice in-process?"""
    lib = voicestudio_venv() / "lib"
    if not lib.exists():
        return False
    for py_dir in lib.glob("python*/site-packages"):
        return True
    return False


# ----- LD_LIBRARY_PATH for NPP 12 (cu12 pip libs) -----------------------

def _npp_env() -> dict[str, str]:
    """Build env with LD_LIBRARY_PATH pointing to pip-installed cu12 libs.

    torchcodec 0.11+cu128 requires libnppicc.so.12 (CUDA 12 NPP). DGX Spark
    has only CUDA 13 system libs, so we need the pip nvidia-npp-cu12 package
    to be loaded FIRST.

    Discovers the actual python X.Y site-packages in VoiceStudio venv (which
    may be a different Python version than this MCP venv).
    """
    env = os.environ.copy()
    venv = voicestudio_venv()
    # Find lib/pythonX.Y/site-packages inside the venv (any version present)
    lib = venv / "lib"
    if lib.exists():
        for py_dir in lib.glob("python*/site-packages"):
            nvidia = py_dir / "nvidia"
            if nvidia.exists():
                paths = [
                    str(nvidia / "npp" / "lib"),
                    str(nvidia / "cuda_runtime" / "lib"),
                    str(nvidia / "cublas" / "lib"),
                    str(nvidia / "cuda_cuxxfilt" / "lib"),
                ]
                existing = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = ":".join(paths + ([existing] if existing else []))
                break
    if hf_token():
        env["HF_TOKEN"] = hf_token()
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token()
    env["HF_HOME"] = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    env["CUDA_VISIBLE_DEVICES"] = cuda_device()
    env["PYTHONUNBUFFERED"] = "1"
    return env


# ----- Subprocess backend ------------------------------------------------

def synth_subprocess(req: SynthRequest, output: Path) -> tuple[float, str]:
    """Run omnivoice-infer as subprocess (direct binary, no uv). Returns (wall_time_s, model_id)."""
    bin_path = voicestudio_venv() / "bin" / "omnivoice-infer"
    cmd = [
        str(bin_path),
        "--model", "k2-fsa/OmniVoice",
        "--text", req.text,
        "--output", str(output),
        "--language", req.language,
        "--num_step", str(req.num_step),
        "--guidance_scale", str(req.guidance_scale),
        "--speed", str(req.speed),
        "--device", "cuda",
    ]
    if req.voice_name:
        # Use the saved voice profile
        from .storage import load_voice
        prof = load_voice(req.voice_name)
        cmd.extend(["--ref_audio", prof.ref_audio_path, "--ref_text", prof.ref_text])
    if req.instruct:
        cmd.extend(["--instruct", req.instruct])

    log_path = logs_dir() / f"synth_{int(time.time())}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = _npp_env()
    # Run from VoiceStudio root so uv finds pyproject
    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=str(voicestudio_root()),
        env=env, capture_output=True, text=True, timeout=600,
    )
    dt = time.time() - t0
    log_path.write_text(
        f"# cmd\n{' '.join(cmd)}\n\n# stdout\n{proc.stdout}\n\n# stderr\n{proc.stderr}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"omnivoice-infer failed (exit={proc.returncode}). Log: {log_path}"
        )
    return dt, "k2-fsa/OmniVoice"


# ----- In-process backend (fallback) -------------------------------------

_INPROC_OK = False


def _ensure_inproc() -> bool:
    """Try to set up sys.path for in-process omnivoice import."""
    global _INPROC_OK
    if _INPROC_OK:
        return True
    lib = voicestudio_venv() / "lib"
    if not lib.exists():
        return False
    # Find any python X.Y site-packages
    candidates = sorted(lib.glob("python*/site-packages"))
    for site in candidates:
        sp = str(site)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    try:
        import omnivoice  # noqa: F401
        _INPROC_OK = True
        return True
    except Exception:
        return False


def synth_inproc(req: SynthRequest, output: Path) -> tuple[float, str]:
    """Run omnivoice in-process. Returns (wall_time_s, model_id)."""
    if not _ensure_inproc():
        raise RuntimeError("in-process omnivoice import not available")
    # Set NPP env for any native loaders triggered later
    for k, v in _npp_env().items():
        if k.startswith("LD_LIBRARY_PATH") or k in ("HF_HOME", "CUDA_VISIBLE_DEVICES"):
            os.environ[k] = v
    import omnivoice
    model = omnivoice.OmniVoice("k2-fsa/OmniVoice", device="cuda")
    kwargs: dict = {"text": req.text, "output": str(output), "language": req.language,
                    "num_step": req.num_step, "guidance_scale": req.guidance_scale,
                    "speed": req.speed}
    if req.voice_name:
        from .storage import load_voice
        prof = load_voice(req.voice_name)
        kwargs["ref_audio"] = prof.ref_audio_path
        kwargs["ref_text"] = prof.ref_text
    if req.instruct:
        kwargs["instruct"] = req.instruct
    t0 = time.time()
    model.synthesize(**kwargs)
    dt = time.time() - t0
    return dt, "k2-fsa/OmniVoice"


# ----- Public API --------------------------------------------------------

def synth(req: SynthRequest, output: Path) -> tuple[float, str]:
    """Synthesize speech. Tries subprocess first, falls back to in-process.

    Returns (wall_time_s, model_id).
    """
    if _has_subprocess():
        return synth_subprocess(req, output)
    if _has_inproc():
        return synth_inproc(req, output)
    raise RuntimeError(
        f"Neither subprocess omnivoice-infer nor in-process omnivoice is available. "
        f"Check VOICESTUDIO_VENV={voicestudio_venv()}"
    )
