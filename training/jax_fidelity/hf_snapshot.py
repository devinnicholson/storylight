"""Download the exact Gemma revision into one immutable local snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .configuration import load_config
from .integrity import artifact_manifest, canonical_json_bytes, sha256_file
from .runtime import approval_token, require_approval

_TOKENIZER_NAMES = {
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
}


class SnapshotError(RuntimeError):
    """The base-model snapshot did not match the frozen revision and access mode."""


def _write_once(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())


def _harden(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise SnapshotError(f"snapshot contains a symbolic link: {path}")
        os.chmod(path, 0o500 if path.is_dir() else 0o400)
    os.chmod(root, 0o500)


def _tokenizer_package(snapshot: Path, destination: Path) -> dict[str, object]:
    if destination.exists() or destination.is_symlink():
        raise SnapshotError("tokenizer output already exists")
    destination.mkdir(parents=True, mode=0o700)
    copied = 0
    for source in sorted(snapshot.iterdir()):
        if source.is_file() and (
            source.name in _TOKENIZER_NAMES
            or source.name.startswith("tokenizer.")
            or source.name.startswith("tokenizer_")
        ):
            shutil.copyfile(source, destination / source.name, follow_symlinks=False)
            copied += 1
    if copied < 2 or not (destination / "tokenizer_config.json").is_file():
        raise SnapshotError("downloaded snapshot has no complete tokenizer package")
    _harden(destination)
    return artifact_manifest(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument(
        "--access-mode",
        choices=("secret-token", "public-anonymous"),
        default="secret-token",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    outputs = (
        args.snapshot,
        args.tokenizer,
        args.snapshot_manifest,
        args.tokenizer_manifest,
        args.completion,
    )
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise SnapshotError("snapshot outputs are write-once")

    config = load_config(args.config)
    model_id = config.production["model_id"]
    revision = config.production["model_revision"]
    binding = hashlib.sha256(
        f"{model_id}@{revision}@{args.access_mode}".encode()
    ).hexdigest()
    run_id = f"hf-snapshot-{config.sha256[:12]}-{revision[:12]}"
    token = approval_token(
        stage="hf-snapshot",
        run_id=run_id,
        config_sha256=config.sha256,
        input_sha256=binding,
    )
    plan = {
        "schema_version": "1.0",
        "run_id": run_id,
        "model_id": model_id,
        "model_revision": revision,
        "access_mode": args.access_mode,
        "config_sha256": config.sha256,
        "approval_token": token,
        "network_download": True,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.execute:
        return
    require_approval(token)
    hf_token: str | bool
    if args.access_mode == "secret-token":
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if not hf_token:
            raise SnapshotError("HF_TOKEN must be supplied by a secret provider")
    else:
        hf_token = False
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise SnapshotError("huggingface_hub is required for the snapshot") from error

    info = HfApi(token=hf_token).model_info(model_id, revision=revision)
    if info.sha != revision:
        raise SnapshotError("Hugging Face resolved a different model revision")
    if args.access_mode == "public-anonymous" and (info.private or info.gated):
        raise SnapshotError("model no longer permits anonymous public access")
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.snapshot.name}.partial-",
        dir=args.snapshot.parent,
    ) as raw:
        temporary = Path(raw) / "snapshot"
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            token=hf_token,
            local_dir=temporary,
        )
        for metadata in (temporary / ".cache",):
            if metadata.exists():
                shutil.rmtree(metadata)
        _harden(temporary)
        os.replace(temporary, args.snapshot)

    snapshot_manifest = artifact_manifest(args.snapshot)
    tokenizer_manifest = _tokenizer_package(args.snapshot, args.tokenizer)
    _write_once(args.snapshot_manifest, snapshot_manifest)
    _write_once(args.tokenizer_manifest, tokenizer_manifest)
    completion = {
        **plan,
        "approval_token": None,
        "status": "succeeded",
        "resolved_revision": info.sha,
        "snapshot_manifest_sha256": sha256_file(args.snapshot_manifest),
        "tokenizer_manifest_sha256": sha256_file(args.tokenizer_manifest),
        "snapshot_content_sha256": snapshot_manifest["content_sha256"],
        "tokenizer_content_sha256": tokenizer_manifest["content_sha256"],
    }
    _write_once(args.completion, completion)


if __name__ == "__main__":
    main()
