from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from guitarmidi.channels import split_channels
from guitarmidi.utils import is_url, slugify


def test_is_url_recognises_youtube():
    assert is_url("https://www.youtube.com/watch?v=abc123")
    assert is_url("https://music.youtube.com/watch?v=abc123")
    assert is_url("http://example.com/foo.mp3")


def test_is_url_rejects_paths():
    assert not is_url("/path/to/song.mp3")
    assert not is_url("song.mp3")
    assert not is_url("C:\\Users\\me\\song.mp3")
    assert not is_url("")


def test_slugify_keeps_safe_chars():
    assert slugify("hello") == "hello"
    assert slugify("abc-123_v2.0") == "abc-123_v2.0"


def test_slugify_replaces_unsafe_chars():
    assert slugify("hello world!") == "hello_world"
    assert slugify("a/b\\c:d?e*f") == "a_b_c_d_e_f"


def test_slugify_handles_empty_and_dotfiles():
    assert slugify("") == "untitled"
    assert slugify("...") == "untitled"
    assert slugify("___") == "untitled"


def test_split_channels_stereo(tmp_path: Path):
    sr = 22050
    n = sr  # 1 second
    L = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    R = np.linspace(0.5, -0.5, n, dtype=np.float32)
    stereo = np.stack([L, R], axis=1)
    src = tmp_path / "stereo.wav"
    sf.write(str(src), stereo, sr)

    out = split_channels(src, tmp_path / "variants")
    assert set(out) == {"L", "R", "mid", "side"}
    for p in out.values():
        assert p.exists() and p.stat().st_size > 0

    L_read, sr_l = sf.read(str(out["L"]))
    R_read, sr_r = sf.read(str(out["R"]))
    mid_read, _ = sf.read(str(out["mid"]))
    side_read, _ = sf.read(str(out["side"]))
    assert sr_l == sr_r == sr
    assert L_read.ndim == 1
    np.testing.assert_allclose(mid_read, (L_read + R_read) / 2, atol=1e-5)
    np.testing.assert_allclose(side_read, (L_read - R_read) / 2, atol=1e-5)


def test_split_channels_mono(tmp_path: Path):
    sr = 22050
    signal = np.linspace(-0.5, 0.5, sr, dtype=np.float32)
    src = tmp_path / "mono.wav"
    sf.write(str(src), signal, sr)

    out = split_channels(src, tmp_path / "variants")
    assert set(out) == {"mono"}
    assert out["mono"].exists()
    read, sr_r = sf.read(str(out["mono"]))
    assert sr_r == sr
    assert read.ndim == 1
