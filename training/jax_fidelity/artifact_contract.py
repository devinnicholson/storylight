"""Create immutable named-artifact manifests for conversion and cloud staging."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from .integrity import artifact_manifest, canonical_json_bytes, sha256_file

_NAME = re.compile(r"[a-z][a-z0-9_]{1,63}\Z")


def create_artifact_contract(artifacts: list[str]) -> dict[str, object]:
    declarations: dict[str, object] = {}
    for value in artifacts:
        name, separator, raw_path = value.partition("=")
        if not separator or _NAME.fullmatch(name) is None or name in declarations:
            raise ValueError(f"invalid or duplicate artifact binding: {value!r}")
        root = Path(raw_path).expanduser().resolve()
        declarations[name] = artifact_manifest(root)
    if not declarations:
        raise ValueError("at least one named artifact is required")
    return {"schema_version": "1.0", "artifacts": declarations}


def _write_once(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="bind a named artifact directory; repeat for each required input",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = create_artifact_contract(args.artifact)
    _write_once(args.output, document)
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
