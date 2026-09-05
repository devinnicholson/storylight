import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bookforge.fidelity_dataset import DatasetSplit
from bookforge.fidelity_manifest import (
    FidelityDatasetManifest,
    record_schema_sha256,
    validate_manifest,
)

DATASET_ROOT = Path("datasets/story-fidelity-v1")


def test_committed_manifest_validates_public_content_and_schema() -> None:
    manifest = validate_manifest(DATASET_ROOT / "manifest.json")

    assert manifest.total_records == 5120
    assert manifest.record_schema_sha256 == record_schema_sha256()
    assert manifest.splits[DatasetSplit.TRAIN].records == 4096
    assert manifest.splits[DatasetSplit.DEVELOPMENT].records == 512
    assert manifest.splits[DatasetSplit.HIDDEN].records == 512
    assert manifest.splits[DatasetSplit.HIDDEN].path is None
    assert manifest.splits[DatasetSplit.HIDDEN].public is False


def test_manifest_rejects_published_hidden_split_and_wrong_counts() -> None:
    payload = json.loads((DATASET_ROOT / "manifest.json").read_text())
    published = copy.deepcopy(payload)
    published["splits"]["hidden"]["path"] = "hidden.jsonl"
    published["splits"]["hidden"]["public"] = True
    with pytest.raises(ValidationError, match="hidden records may not be published"):
        FidelityDatasetManifest.model_validate(published)

    wrong_count = copy.deepcopy(payload)
    wrong_count["splits"]["train"]["records"] = 4095
    with pytest.raises(ValidationError, match="total_records|incorrect train count"):
        FidelityDatasetManifest.model_validate(wrong_count)


def test_manifest_rejects_unsafe_public_path() -> None:
    payload = json.loads((DATASET_ROOT / "manifest.json").read_text())
    payload["splits"]["train"]["path"] = "../train.jsonl"
    with pytest.raises(ValidationError, match="safe relative path"):
        FidelityDatasetManifest.model_validate(payload)
