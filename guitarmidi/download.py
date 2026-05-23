from __future__ import annotations

from pathlib import Path


def download_audio(url: str, output_dir: Path, cache: bool = True) -> tuple[Path, str]:
    """Download YouTube audio as WAV.

    Returns (wav_path, video_id). Works for youtube.com and music.youtube.com.
    If cache=True and `<video_id>.wav` already exists in output_dir, the
    download is skipped — yt-dlp's own caching is unreliable.
    """
    from yt_dlp import YoutubeDL

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cache:
        cached = _find_cached_wav(url, output_dir)
        if cached is not None:
            return cached

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "quiet": False,
        "noprogress": False,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_id = info["id"]
    wav_path = output_dir / f"{video_id}.wav"
    if not wav_path.exists():
        raise RuntimeError(
            f"yt-dlp finished but no WAV at {wav_path}. "
            "Is ffmpeg installed and on PATH?"
        )
    return wav_path, video_id


def _find_cached_wav(url: str, output_dir: Path) -> tuple[Path, str] | None:
    """Resolve the video ID without downloading; return (path, id) if cached.

    Uses yt-dlp's extract_info(..., download=False) which only hits the
    metadata API. Cheap relative to a full download.
    """
    from yt_dlp import YoutubeDL

    try:
        with YoutubeDL({"quiet": True, "skip_download": True, "noprogress": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return None

    video_id = info.get("id")
    if not video_id:
        return None
    wav_path = output_dir / f"{video_id}.wav"
    if wav_path.exists() and wav_path.stat().st_size > 0:
        return wav_path, video_id
    return None
