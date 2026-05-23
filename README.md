# guitarmidi

Audio-to-MIDI pipeline for hand-tabbing guitar parts.

```
YouTube URL → audio download → guitar stem separation →
L/R/mid/side channel split → audio-to-MIDI per variant →
folder of MIDI files ready for DAW import
```

The MIDI is meant for hand-tabbing in Guitar Pro / MuseScore. No automated
string/fret assignment is attempted.

## Install

This project uses [uv](https://github.com/astral-sh/uv). Python 3.11 is
required — `basic-pitch` 0.4 pins TensorFlow <2.15.1, which only has wheels
for cp311.

```bash
uv sync
```

That's it — `uv sync` reads `uv.lock` and installs the exact set that's been
verified to work (including the cu128 PyTorch wheels, the win32-only
`tensorflow-intel` shim dep, and the dev tools).

You also need `ffmpeg` on PATH:

- macOS: `brew install ffmpeg`
- Windows (winget): `winget install Gyan.FFmpeg`
- Linux: `apt install ffmpeg`

Note: `basic-pitch` pulls in TensorFlow, so the first install is ~500MB.

## Usage

```bash
# YouTube URL (youtube.com and music.youtube.com both work)
uv run guitarmidi "https://music.youtube.com/watch?v=Sm40N0ScfHI"

# Local audio file
uv run guitarmidi /path/to/song.mp3

# Skip channel split — one MIDI from the combined stereo stem
uv run guitarmidi <input> --no-channel-split

# Force GPU / CPU
uv run guitarmidi <input> --device cuda
uv run guitarmidi <input> --device cpu

# Tune transcription (lower onset threshold = more sensitive)
uv run guitarmidi <input> --onset-threshold 0.3 --min-note-length 40
```

## Tests

```bash
uv run pytest tests/
```

Outputs land in `./output/<song_id>/`:

```
output/<song_id>/
├── original.wav
├── stems/         (vocals, drums, bass, guitar, piano, other)
├── variants/      (guitar_L, guitar_R, guitar_mid, guitar_side)
└── midi/          (one .mid per variant)
```

## Troubleshooting

- **YouTube download fails**: bump yt-dlp in `pyproject.toml` and re-run
  `uv lock && uv sync`. YouTube tends to break yt-dlp every few weeks.
- **`ffmpeg not found` after `winget install Gyan.FFmpeg`**: winget updates the
  user PATH but already-open shells won't see it. Open a new shell or set
  `PATH` for the current session.
- **Partial output cached as done**: delete the `output/<song_id>/` directory
  to force a clean re-run.
- **Out of memory on long videos**: htdemucs_6s with default segments needs
  ~4GB RAM. Try `--device cpu` and/or shorter inputs.
- **CUDA on Windows**: only the demucs separation step uses CUDA. TensorFlow
  on Windows is CPU-only after 2.10, so basic-pitch transcription runs on
  CPU regardless of `--device`. The cu128 torch wheel is sourced via
  `[tool.uv.sources]` in pyproject.toml.
