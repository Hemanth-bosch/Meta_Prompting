from __future__ import annotations

from pathlib import Path


def resolve_default_split_file(repo_root: Path, split_seed: int = 42) -> Path:
    """Return the preferred default split manifest path for evaluation scripts."""
    cache_dir = repo_root / ".cache"
    legacy_path = cache_dir / f"dca_split_seed{split_seed}.json"
    if legacy_path.exists():
        return legacy_path

    fingerprinted = []
    for search_dir in (cache_dir, cache_dir / "splits"):
        if not search_dir.exists():
            continue
        fingerprinted.extend(search_dir.glob(f"dca_split_seed{split_seed}_*.json"))

    fingerprinted = sorted(
        fingerprinted,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if fingerprinted:
        return fingerprinted[0]

    return legacy_path


def is_split_manifest_file(path: Path) -> bool:
    return path.suffix == ".json" and path.name.startswith("dca_split_seed")