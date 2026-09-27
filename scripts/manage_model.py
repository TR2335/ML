"""Promote, verify and roll back model artifacts.

The registry is intentionally filesystem based so the workflow is runnable in
the demo environment. A hospital deployment can replace the copy operation
with an approved artifact store while preserving the manifest shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from app.synthetic import FEATURES

TARGETS = {"deterioration_24h", "readmission_30d"}
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_artifact(path: Path, target_id: str, version: str | None = None) -> dict[str, Any]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read model artifact: {path}") from exc
    if not isinstance(artifact, dict) or artifact.get("feature_names") != FEATURES:
        raise ValueError("model feature signature is invalid")
    if artifact.get("target_id") != target_id:
        raise ValueError("model target signature is invalid")
    artifact_version = str(artifact.get("model_version", ""))
    if not VERSION_PATTERN.fullmatch(artifact_version) or (version and artifact_version != version):
        raise ValueError("model version does not match requested version")
    return artifact


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "models": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("models", {}), dict):
        raise TypeError("registry manifest is invalid")
    return value


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def promote(args: argparse.Namespace) -> None:
    artifact_path = Path(args.artifact)
    artifact = load_artifact(artifact_path, args.target_id, args.version)
    version = str(artifact["model_version"])
    registry = Path(args.registry_dir)
    destination = registry / args.target_id / f"{version}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(artifact_path, destination)
    file_hash = digest(destination)
    manifest_path = registry / "manifest.json"
    manifest = read_manifest(manifest_path)
    manifest.setdefault("models", {})[args.target_id] = {
        "active_version": version,
        "artifact": destination.relative_to(registry).as_posix(),
        "sha256": file_hash,
    }
    atomic_write(manifest_path, manifest)
    if args.serving_path:
        serving_path = Path(args.serving_path)
        serving_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(destination, serving_path)
    print(
        json.dumps(
            {"target_id": args.target_id, "version": version, "sha256": file_hash},
            ensure_ascii=False,
        )
    )


def rollback(args: argparse.Namespace) -> None:
    if not VERSION_PATTERN.fullmatch(args.version):
        raise ValueError("model version is invalid")
    registry = Path(args.registry_dir)
    source = registry / args.target_id / f"{args.version}.json"
    artifact = load_artifact(source, args.target_id, args.version)
    file_hash = digest(source)
    manifest_path = registry / "manifest.json"
    manifest = read_manifest(manifest_path)
    manifest.setdefault("models", {})[args.target_id] = {
        "active_version": args.version,
        "artifact": source.relative_to(registry).as_posix(),
        "sha256": file_hash,
    }
    atomic_write(manifest_path, manifest)
    if args.serving_path:
        serving_path = Path(args.serving_path)
        serving_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, serving_path)
    print(
        json.dumps(
            {
                "target_id": args.target_id,
                "version": artifact["model_version"],
                "sha256": file_hash,
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", default="model-registry")
    subparsers = parser.add_subparsers(dest="command", required=True)
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--artifact", required=True)
    promote_parser.add_argument("--target-id", choices=sorted(TARGETS), required=True)
    promote_parser.add_argument("--version")
    promote_parser.add_argument("--serving-path")
    promote_parser.set_defaults(handler=promote)
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--target-id", choices=sorted(TARGETS), required=True)
    rollback_parser.add_argument("--version", required=True)
    rollback_parser.add_argument("--serving-path")
    rollback_parser.set_defaults(handler=rollback)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
