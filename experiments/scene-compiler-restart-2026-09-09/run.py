"""Seed-controlled fresh-process compiler-cache experiment with a stable wrapper."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

PROFILE_SHA = "b308c7e236ef81b30915fe416639a6b8032e7a5faa2b94a422754082d62a4939"
CPU_PROOF_SHA = "f871a13638172227673d571b380bb1918010d23b72c377a7286ad1931fb2a92c"
EXPECTED_ROOT = Path("/tmp/storylight-v5-seeded-compiler-cache-01")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, sha, name):
    require(digest(path) == sha, "source changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(root):
    require(root.is_dir() and not root.is_symlink(), "compiler root missing or symlink")
    files = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "compiler symlink")
        if path.is_file():
            files[str(path.relative_to(root))] = dict(
                bytes=path.stat().st_size, sha256=digest(path)
            )
        else:
            require(path.is_dir(), "unexpected compiler entry")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return dict(
        root=str(root),
        files=files,
        file_count=len(files),
        total_bytes=sum(row["bytes"] for row in files.values()),
        inventory_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "compiled-profile",
        "compiler-cache-root",
        "bridge-helper",
        "v5-trainer",
        "v5-runner",
        "v2-trainer",
        "v3-runner",
        "merge-helper",
        "model-dir",
        "model-manifest",
        "training-run",
        "data-dir",
        "grammar",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    for name in ("completed", "model-manifest"):
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--max-runtime-seconds", type=int, default=900)
    parser.add_argument("--arm", choices=("cold-0", "reuse-0", "reuse-1"), required=True)
    parser.add_argument("--prior-run", type=Path)
    parser.add_argument("--prior-completed-sha256")
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), "output already exists")
    require(1 <= args.max_runtime_seconds <= 900, "finite bound")
    require(args.compiler_cache_root == EXPECTED_ROOT, "unexpected compiler root")
    expected_seed = "1" if args.arm == "reuse-1" else "0"
    require(os.environ.get("PYTHONHASHSEED") == expected_seed, "launch with expected hash seed")
    startup = dict(
        pid=os.getpid(),
        python_version=sys.version,
        python_hash_seed=os.environ["PYTHONHASHSEED"],
        hash_randomization=sys.flags.hash_randomization,
        hash_info={
            name: getattr(sys.hash_info, name)
            for name in ("width", "modulus", "algorithm", "hash_bits", "seed_bits", "cutoff")
        },
        layer_type_hashes={name: hash(name) for name in ("full_attention", "sliding_attention")},
        arm=args.arm,
        cpu_proof_sha256=CPU_PROOF_SHA,
    )
    locations = {
        "TORCHINDUCTOR_CACHE_DIR": args.compiler_cache_root / "inductor",
        "TRITON_CACHE_DIR": args.compiler_cache_root / "triton",
    }
    prior, prior_rows = None, None
    if args.arm == "cold-0":
        require(args.prior_run is None and args.prior_completed_sha256 is None, "unexpected prior")
        require(
            not args.compiler_cache_root.exists() and not args.compiler_cache_root.is_symlink(),
            "cold cache root must be new",
        )
        args.compiler_cache_root.mkdir(exist_ok=False)
        for path in locations.values():
            path.mkdir(exist_ok=False)
    else:
        require(
            args.prior_run is not None and args.prior_completed_sha256 is not None, "prior required"
        )
        require(
            digest(args.prior_run / "completed.json") == args.prior_completed_sha256,
            "prior changed",
        )
        prior = json.loads((args.prior_run / "completed.json").read_text())
        require(
            prior["calls"] == 54
            and prior["all_calls_token_identical"]
            and (args.arm != "reuse-1" or prior["all_54_prior_tokens_identical"]),
            "prior parity",
        )
        require(
            digest(args.prior_run / "protocol.json") == prior["files"]["protocol.json"],
            "prior protocol",
        )
        protocol = json.loads((args.prior_run / "protocol.json").read_text())
        require(
            protocol["source_sha256"] == PROFILE_SHA
            and protocol["confirmation_wrapper_sha256"] == digest(Path(__file__))
            and protocol["completed_sha256"] == args.completed_sha256
            and protocol["model_manifest_sha256"] == args.model_manifest_sha256
            and protocol["compiler_cache_preparation"]["directories"]
            == {k: str(v) for k, v in locations.items()}
            and protocol["startup"]["arm"] == ("cold-0" if args.arm == "reuse-0" else "reuse-0"),
            "prior experiment mismatch",
        )
        require(
            digest(args.prior_run / "raw.jsonl") == prior["files"]["raw.jsonl"], "prior raw changed"
        )
        prior_rows = [
            json.loads(line) for line in (args.prior_run / "raw.jsonl").read_text().splitlines()
        ]
        require(len(prior_rows) == 54, "prior raw count")
    before = inventory(args.compiler_cache_root)
    if prior is not None:
        previous_inventory_path = args.prior_run / "compiler-cache-after.json"
        require(
            digest(previous_inventory_path) == prior["files"]["compiler-cache-after.json"],
            "prior cache receipt changed",
        )
        previous_inventory = json.loads(previous_inventory_path.read_text())
        require(before["files"] == previous_inventory["files"], "cache changed between arms")
    require((before["file_count"] == 0) == (args.arm == "cold-0"), "cache state mismatch")
    for name, path in locations.items():
        require(path.is_dir() and not path.is_symlink(), "compiler directory missing")
        os.environ[name] = str(path)
    args.compiler_cache_preparation = dict(
        mode="new local compiler directories"
        if args.arm == "cold-0"
        else "existing local compiler artifacts",
        directories={k: str(v) for k, v in locations.items()},
        prior_completed_sha256=args.prior_completed_sha256,
        file_count=before["file_count"],
        total_bytes=before["total_bytes"],
        inventory_sha256=before["inventory_sha256"],
        limitation="Fresh Python process; no claim of cold driver, GPU, OS or filesystem caches.",
    )
    compiled = load(args.compiled_profile, PROFILE_SHA, "seeded_compiled_profile")
    args.profile = load(args.merge_helper, compiled.PROFILE_SHA, "seeded_merge_helper")
    args.wrapper = load(args.v5_trainer, args.profile.TRAINER_SHA, "seeded_training_helper")
    args.runner = args.wrapper.load_module(args.v5_runner, args.profile.RUNNER_SHA, "seeded_runner")
    helper = args.wrapper.load_module(args.v2_trainer, args.wrapper.V2_SHA256, "seeded_parent")
    startup["helper_modules"] = {
        "compiled": compiled.__name__,
        "merge": args.profile.__name__,
        "trainer": args.wrapper.__name__,
        "runner": args.runner.__name__,
        "parent": helper.__name__,
    }
    original_write = helper.write_json

    def write_with_provenance(path, value):
        if path == args.output / "protocol.json":
            value = {
                **value,
                "confirmation_wrapper_sha256": digest(Path(__file__)),
                "prior_completed_sha256": args.prior_completed_sha256,
                "startup": startup,
            }
        original_write(path, value)

    original_merge = args.profile.checked_merge

    def merge_with_order(model, torch, parent):
        merged, hashes = original_merge(model, torch, parent)
        text_models = [
            module for module in merged.modules() if type(module).__name__ == "Gemma4TextModel"
        ]
        require(len(text_models) == 1, "text model scope")
        text = text_models[0]
        order = list(text.unique_layer_types)
        expected_order = (
            ["full_attention", "sliding_attention"]
            if expected_seed == "0"
            else ["sliding_attention", "full_attention"]
        )
        parent.write_json(
            args.output / "model-order.json",
            dict(
                class_module=type(text).__module__,
                actual_unique_layer_types=order,
                rotary_buffer_registration_order=[
                    name for name, _ in text.rotary_emb.named_buffers()
                ],
                rotary_frequency_shapes={
                    name: list(getattr(text.rotary_emb, name + "_inv_freq").shape) for name in order
                },
                expected_order=expected_order,
                matches_cpu_order=order == expected_order,
            ),
        )
        require(order == expected_order, "GPU runtime hash order differs from CPU proof")
        return merged, hashes

    def execute(arguments, parent, state):
        parent.write_json(arguments.output / "compiler-cache-before.json", before)
        result = compiled.execute(arguments, parent, state)
        parent.write_json(
            arguments.output / "compiler-cache-after.json", inventory(arguments.compiler_cache_root)
        )
        if prior_rows is not None:
            current_rows = [
                json.loads(line)
                for line in (arguments.output / "raw.jsonl").read_text().splitlines()
            ]
            require(len(current_rows) == 54, "current raw count")
            keys = ("dispatch_ordinal", "mode", "id", "case_index", "repetition", "warmup")
            comparisons = []
            for old, new in zip(prior_rows, current_rows, strict=True):
                require(all(old[key] == new[key] for key in keys), "prior schedule mismatch")
                comparisons.append(
                    {
                        **{key: new[key] for key in keys},
                        "tokens_identical": old["token_ids"] == new["token_ids"],
                    }
                )
            prior_comparison = dict(
                prior_completed_sha256=args.prior_completed_sha256,
                prior_raw_sha256=prior["files"]["raw.jsonl"],
                rows=comparisons,
                all_54_tokens_identical=all(row["tokens_identical"] for row in comparisons),
            )
            parent.write_json(arguments.output / "prior-comparison.json", prior_comparison)
            return {
                **result,
                "all_54_prior_tokens_identical": prior_comparison["all_54_tokens_identical"],
            }

        return result

    args.profile.checked_merge = merge_with_order
    helper.write_json = write_with_provenance
    try:
        args.wrapper.bounded(args, helper, execute, 900)
    finally:
        helper.write_json = original_write
        args.profile.checked_merge = original_merge


if __name__ == "__main__":
    main()
