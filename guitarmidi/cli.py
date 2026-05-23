from __future__ import annotations

import sys
from pathlib import Path

import click

from .pipeline import run_pipeline


def _ensure_utf8_console() -> None:
    """basic-pitch prints emoji in its progress/result messages, which crash
    Windows' default cp1252 stdout. Force UTF-8 (with replacement) before any
    library does printing."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


@click.command()
@click.argument("input_source")
@click.option(
    "--output-dir",
    "-o",
    default="./output",
    type=click.Path(file_okay=False),
    help="Root output directory.",
)
@click.option(
    "--no-channel-split",
    is_flag=True,
    help="Skip L/R/mid/side split; transcribe the combined guitar stem only.",
)
@click.option(
    "--device",
    default="auto",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    help="Inference device for demucs.",
)
@click.option("--onset-threshold", default=0.5, type=float, show_default=True)
@click.option("--frame-threshold", default=0.3, type=float, show_default=True)
@click.option(
    "--min-note-length",
    default=58,
    type=int,
    show_default=True,
    help="Minimum note length in ms. Higher = more ghost-note filtering.",
)
@click.option(
    "--min-freq",
    default=80.0,
    type=float,
    show_default=True,
    help="Minimum frequency considered by basic-pitch (Hz).",
)
@click.option(
    "--max-freq",
    default=1500.0,
    type=float,
    show_default=True,
    help="Maximum frequency considered by basic-pitch (Hz).",
)
def main(
    input_source: str,
    output_dir: str,
    no_channel_split: bool,
    device: str,
    onset_threshold: float,
    frame_threshold: float,
    min_note_length: int,
    min_freq: float,
    max_freq: float,
) -> None:
    """Generate MIDI from a YouTube URL or local audio file.

    INPUT_SOURCE: YouTube URL or path to a local audio file.
    """
    _ensure_utf8_console()
    transcribe_kwargs = {
        "onset_threshold": onset_threshold,
        "frame_threshold": frame_threshold,
        "min_note_length": min_note_length,
        "min_freq": min_freq,
        "max_freq": max_freq,
    }
    midi_paths = run_pipeline(
        input_source=input_source,
        output_root=Path(output_dir),
        skip_channel_split=no_channel_split,
        device=device,
        transcribe_kwargs=transcribe_kwargs,
    )
    click.echo("\nGenerated MIDI files:")
    for name, path in midi_paths.items():
        click.echo(f"  [{name}] {path}")


if __name__ == "__main__":
    main()
