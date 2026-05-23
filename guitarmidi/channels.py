from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def split_channels(stereo_wav: Path, output_dir: Path) -> dict[str, Path]:
    """Split a stereo WAV into L / R / mid / side mono WAVs.

    If the input is mono, write a single `guitar_mono.wav` and return
    {"mono": path} so callers can treat it uniformly.
    """
    stereo_wav = Path(stereo_wav)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(str(stereo_wav), always_2d=False)

    # Write as 32-bit float WAV so channel math (esp. (L-R)/2) isn't quantized
    # by PCM_16, which costs ~3e-5 absolute error per sample and shows up as
    # ghost noise in `side` for nearly-mono inputs.
    if audio.ndim == 1 or (audio.ndim == 2 and audio.shape[1] == 1):
        signal = audio if audio.ndim == 1 else audio[:, 0]
        p = output_dir / "guitar_mono.wav"
        sf.write(str(p), signal.astype(np.float32, copy=False), sr, subtype="FLOAT")
        return {"mono": p}

    L = audio[:, 0].astype(np.float32, copy=False)
    R = audio[:, 1].astype(np.float32, copy=False)
    variants = {
        "L": L,
        "R": R,
        "mid": (L + R) * 0.5,
        "side": (L - R) * 0.5,
    }

    paths: dict[str, Path] = {}
    for name, signal in variants.items():
        p = output_dir / f"guitar_{name}.wav"
        sf.write(str(p), signal.astype(np.float32, copy=False), sr, subtype="FLOAT")
        paths[name] = p
    return paths
