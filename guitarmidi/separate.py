from __future__ import annotations

from pathlib import Path

from .utils import detect_device

STEM_NAMES = ("vocals", "drums", "bass", "guitar", "piano", "other")


def separate_guitar(
    input_wav: Path,
    output_dir: Path,
    device: str = "auto",
    model_name: str = "htdemucs_6s",
) -> dict[str, Path]:
    """Run demucs and write each stem as a WAV to output_dir.

    Returns a dict mapping stem name to path. If every expected stem is
    already present on disk, the model is not invoked.
    """
    input_wav = Path(input_wav)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cached = {name: output_dir / f"{name}.wav" for name in STEM_NAMES}
    if all(p.exists() and p.stat().st_size > 0 for p in cached.values()):
        return cached

    if device == "auto":
        device = detect_device()

    # Lazy imports: demucs/torch are heavy and importing them eagerly makes
    # `--help` slow and breaks `pytest` in light envs. demucs 4.0.1 (current
    # PyPI) exposes pretrained.get_model + apply.apply_model; the
    # `demucs.api.Separator` wrapper used in newer commits isn't released yet.
    # torchaudio 2.11 dropped its built-in audio backend and now requires
    # torchcodec; we use soundfile for I/O instead.
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    from demucs.pretrained import get_model

    model = get_model(model_name)
    model.to(device)
    model.eval()

    # Read input. soundfile returns (time, channels) or (time,) for mono;
    # demucs expects (channels, time).
    audio, sr = sf.read(str(input_wav), always_2d=True, dtype="float32")
    wav = torch.from_numpy(audio.T.copy())  # (channels, time)
    wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)
    # apply_model expects a batched input: (batch, channels, time).
    mix = wav.unsqueeze(0).to(device)

    with torch.no_grad():
        # Returns (batch, sources, channels, time). split=True chunks long
        # audio so memory stays bounded on a 10GB RTX 3080.
        out = apply_model(model, mix, split=True, overlap=0.25, progress=True)
    stems_tensor = out.squeeze(0).cpu()  # (sources, channels, time)

    paths: dict[str, Path] = {}
    for idx, source_name in enumerate(model.sources):
        tensor = stems_tensor[idx]
        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"Demucs produced non-finite samples for stem '{source_name}' "
                f"on device '{device}'. Try --device cpu."
            )
        out_path = output_dir / f"{source_name}.wav"
        # soundfile expects (frames, channels), so transpose back.
        sf.write(
            str(out_path),
            tensor.numpy().T.astype(np.float32, copy=False),
            model.samplerate,
            subtype="FLOAT",
        )
        paths[source_name] = out_path

    if "guitar" not in paths:
        raise RuntimeError(
            f"Model '{model_name}' did not produce a 'guitar' stem. "
            f"Got: {sorted(paths)}"
        )
    return paths
