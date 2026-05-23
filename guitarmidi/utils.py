from __future__ import annotations

import re
from urllib.parse import urlparse


def detect_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' based on availability."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_url(s: str) -> bool:
    """True if s looks like an http(s) URL."""
    try:
        parsed = urlparse(s)
    except (ValueError, AttributeError):
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


_SLUG_BAD = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(name: str) -> str:
    """Filesystem-safe version of a string.

    Keeps alphanumerics, dot, underscore, hyphen; collapses everything else
    to a single underscore. Strips leading/trailing dots and underscores so
    the result is portable across Windows and POSIX.
    """
    s = _SLUG_BAD.sub("_", name).strip("._")
    return s or "untitled"
