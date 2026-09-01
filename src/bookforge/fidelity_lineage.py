"""Shared content-derived identifiers for the Story Fidelity pipeline."""

from __future__ import annotations

import hashlib
import json


def stable_run_id(*, stage: str, config_sha256: str, dataset_manifest_sha256: str) -> str:
    payload = json.dumps(
        {
            "stage": stage,
            "config_sha256": config_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(f"{payload}\n".encode()).hexdigest()[:20]
    return f"{stage}-{digest}"
