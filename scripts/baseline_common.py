"""Content fingerprints and run stamps for reproducible baseline drivers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def content_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def source_sha256(root: Path) -> str:
    paths = [root / "train_colmap3d.py"]
    paths.extend((root / "drawingwithgaussians").glob("*.py"))
    paths.extend((root / "scripts").glob("*.py"))
    return content_sha256(paths, root)


def experiment_sha256(config_sha256: str, source_hash: str) -> str:
    return hashlib.sha256(f"{config_sha256}:{source_hash}".encode()).hexdigest()


def write_run_stamp(output: Path, **payload: object) -> None:
    (output / "baseline_run.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def matching_run_stamp(output: Path, **expected: object) -> bool:
    path = output / "baseline_run.json"
    if not path.is_file():
        return False
    actual = json.loads(path.read_text())
    return all(actual.get(key) == value for key, value in expected.items())
