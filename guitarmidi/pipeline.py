from __future__ import annotations

import shutil
from pathlib import Path

from .channels import split_channels
from .download import download_audio
from .separate import separate_guitar
from .transcribe import transcribe
from .utils import is_url, slugify


def run_pipeline(
    input_source: str | Path,
    output_root: Path,
    skip_channel_split: bool = False,
    device: str = "auto",
    transcribe_kwargs: dict | None = None,
) -> dict[str, Path]:
    """Run the full pipeline. Returns {variant_name: midi_path}.

    Steps:
      1. Resolve input → ``<song_dir>/original.wav`` + song_id
      2. demucs → ``<song_dir>/stems/``
      3. (optional) channel split → ``<song_dir>/variants/``
      4. transcribe → ``<song_dir>/midi/``
    """
    transcribe_kwargs = transcribe_kwargs or {}
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    src = str(input_source)
    if is_url(src):
        download_dir = output_root / "_downloads"
        wav_path, song_id = download_audio(src, download_dir)
    else:
        wav_path = Path(input_source)
        if not wav_path.exists():
            raise FileNotFoundError(f"Input audio not found: {wav_path}")
        song_id = wav_path.stem

    song_dir = output_root / slugify(song_id)
    song_dir.mkdir(parents=True, exist_ok=True)

    original_path = song_dir / "original.wav"
    if not original_path.exists():
        # Copy rather than symlink: Windows symlinks need elevation and the
        # original.wav anchors the song folder for later resumes.
        shutil.copy2(wav_path, original_path)

    stems_dir = song_dir / "stems"
    stems = separate_guitar(original_path, stems_dir, device=device)
    guitar_stem = stems["guitar"]

    if skip_channel_split:
        variant_paths = {"combined": guitar_stem}
    else:
        variants_dir = song_dir / "variants"
        variant_paths = split_channels(guitar_stem, variants_dir)

    midi_dir = song_dir / "midi"
    midi_paths: dict[str, Path] = {}
    for name, wav in variant_paths.items():
        midi_paths[name] = transcribe(wav, midi_dir, **transcribe_kwargs)

    return midi_paths
