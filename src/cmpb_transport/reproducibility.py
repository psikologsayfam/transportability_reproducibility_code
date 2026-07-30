"""Seeds, hashes, and non-overwriting run directories."""

from __future__ import annotations
import hashlib
import os
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch/CUDA when installed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
    except ModuleNotFoundError:
        pass


def sha256(path: Path) -> str:
    """Return SHA-256 of an existing file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def new_run_directory(root: Path, command: str) -> Path:
    """Create a timestamped run directory and never overwrite existing output."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = root / f"{stamp}_{command}"
    candidate = base
    index = 1
    while candidate.exists():
        candidate = root / f"{base.name}_{index:02d}"
        index += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate
