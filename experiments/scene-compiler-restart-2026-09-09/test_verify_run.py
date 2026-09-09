"""Reject seed and helper-name confounds before accepting restart comparisons."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "seeded_audit", Path(__file__).with_name("verify-run.py")
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_startup_checks_seed_and_stable_import_names():
    startup = dict(
        arm="cold-0",
        python_hash_seed="0",
        hash_randomization=0,
        pid=23,
        python_version="3.12",
        cpu_proof_sha256=audit.CPU_SHA,
        helper_modules=dict(
            compiled="seeded_compiled_profile",
            merge="seeded_merge_helper",
            trainer="seeded_training_helper",
            runner="seeded_runner",
            parent="seeded_parent",
        ),
        layer_type_hashes=dict(full_attention=10, sliding_attention=20),
    )
    audit.check_startup(startup, "cold-0")
    for arm, seed, randomization in [("reuse-0", "0", 0), ("reuse-1", "1", 1)]:
        audit.check_startup(
            {**startup, "arm": arm, "python_hash_seed": seed, "hash_randomization": randomization},
            arm,
        )
    with pytest.raises(ValueError, match="startup seed"):
        audit.check_startup({**startup, "python_hash_seed": "1"}, "cold-0")
    altered = copy.deepcopy(startup)
    altered["helper_modules"]["compiled"] = "different_process_import"
    with pytest.raises(ValueError, match="stable helper"):
        audit.check_startup(altered, "cold-0")
