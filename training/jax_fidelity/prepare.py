"""Materialize exact production chat records from a verified fidelity split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .formatting import format_training_record
from .integrity import canonical_json_bytes, sha256_file, validate_dataset_manifest


def prepare_training_jsonl(source: Path | str, destination: Path | str) -> dict[str, Any]:
    source_path = Path(source)
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    count = 0
    try:
        with (
            source_path.open("r", encoding="utf-8") as input_stream,
            os.fdopen(descriptor, "wb") as output_stream,
        ):
            for line_number, line in enumerate(input_stream, start=1):
                try:
                    record = json.loads(line)
                    prepared = format_training_record(record)
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"could not prepare {source_path}:{line_number}: {error}"
                    ) from error
                output_stream.write(canonical_json_bytes(prepared))
                count += 1
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    if count == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError("training split contained no records")
    return {
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "records": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = validate_dataset_manifest(
        args.dataset_manifest,
        expected_manifest_sha256=args.dataset_manifest_sha256,
    )
    train = next((split_ for split_ in dataset.splits if split_.name == "train"), None)
    if train is None:
        raise SystemExit("verified dataset does not expose a train split")
    result = prepare_training_jsonl(train.path, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
