from __future__ import annotations

from pathlib import Path


def transcribe(
    input_wav: Path,
    output_dir: Path,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_length: int = 58,
    min_freq: float = 80.0,
    max_freq: float = 1500.0,
) -> Path:
    """Run basic-pitch on input_wav. Returns the path to the output MIDI.

    Output filename follows basic-pitch's convention: ``<stem>_basic_pitch.mid``.
    If that file already exists, this function returns it without re-running.
    """
    input_wav = Path(input_wav)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_path = output_dir / f"{input_wav.stem}_basic_pitch.mid"
    if midi_path.exists() and midi_path.stat().st_size > 0:
        return midi_path

    # Lazy import: basic-pitch pulls in TensorFlow, which is expensive.
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict_and_save

    predict_and_save(
        audio_path_list=[str(input_wav)],
        output_directory=str(output_dir),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_length,
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
    )

    if not midi_path.exists():
        raise RuntimeError(
            f"basic-pitch ran but no MIDI at {midi_path}. "
            f"Files in {output_dir}: {sorted(p.name for p in output_dir.iterdir())}"
        )
    return midi_path
