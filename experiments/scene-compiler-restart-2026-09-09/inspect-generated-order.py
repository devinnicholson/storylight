"""Compare generated decoder ASTs from the controlled restart; never execute cache artifacts."""

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

CPU_HELPER_SHA = "d1e04e0e8316dd441946b602824cc5fce02c28410ce87e5f72c9257dbf06ac3d"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, sha):
    require(digest(path) == sha, "receipt changed")
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for arm in ("cold", "same", "changed"):
        parser.add_argument(f"--{arm}-completed-sha256", required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "fresh output required")
    here = Path(__file__).resolve().parent
    source = here.parent / "scene-compile-order-2026-09-09/cpu-proof.py"
    require(digest(source) == CPU_HELPER_SHA, "AST helper changed")
    spec = importlib.util.spec_from_file_location("controlled_order_ast", source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    arms = {}
    for arm, sha in (
        ("cold-0", args.cold_completed_sha256),
        ("reuse-0", args.same_completed_sha256),
        ("reuse-1", args.changed_completed_sha256),
    ):
        directory = here / "results/gpu" / f"seeded-{arm}"
        completed = pinned(directory / "completed.json", sha)
        arms[arm] = {
            name: pinned(directory / name, completed["files"][name])
            for name in (
                "compiler-cache-before.json",
                "compiler-cache-after.json",
                "model-order.json",
                "protocol.json",
            )
        }
    old_files = arms["cold-0"]["compiler-cache-after.json"]["files"]
    same_files = arms["reuse-0"]["compiler-cache-after.json"]["files"]
    new_files = arms["reuse-1"]["compiler-cache-after.json"]["files"]
    require(
        old_files == arms["reuse-0"]["compiler-cache-before.json"]["files"] == same_files,
        "same-seed cache changed",
    )
    require(
        same_files == arms["reuse-1"]["compiler-cache-before.json"]["files"],
        "control inventory mismatch",
    )
    require(
        all(new_files.get(name) == row for name, row in same_files.items()),
        "existing artifact changed",
    )
    old_names = [
        name for name, row in old_files.items() if name.endswith(".py") and row["bytes"] > 1_000_000
    ]
    new_names = [
        name
        for name, row in new_files.items()
        if name not in old_files and name.endswith(".py") and row["bytes"] > 1_000_000
    ]
    require(len(old_names) == len(new_names) == 2, "expected two decoder partitions")
    parsed = {}
    for name in old_names + new_names:
        path = args.cache_root / name
        require(
            not path.is_symlink() and digest(path) == new_files[name]["sha256"],
            "generated Python changed",
        )
        parsed[name] = helper.partition(path.read_text())
    comparisons, used = [], set()
    for old_name in old_names:
        matches = []
        for new_name in new_names:
            old = copy.deepcopy(parsed[old_name])
            if ast.dump(helper.SwapRotaryArguments().visit(old)) == ast.dump(parsed[new_name]):
                matches.append(new_name)
        require(len(matches) == 1 and matches[0] not in used, "ambiguous graph correspondence")
        new_name = matches[0]
        used.add(new_name)
        comparisons.append(
            dict(
                old_file=old_name,
                changed_seed_file=new_name,
                old_sha256=new_files[old_name]["sha256"],
                new_sha256=new_files[new_name]["sha256"],
                original_partition_ast_equal=ast.dump(parsed[old_name])
                == ast.dump(parsed[new_name]),
                partition_ast_equal_after_arg8_arg9_swap=True,
                old_argument_shapes=helper.rotary_argument_shapes(parsed[old_name]),
                changed_seed_argument_shapes=helper.rotary_argument_shapes(parsed[new_name]),
            )
        )
    result = dict(
        source_sha256=digest(Path(__file__)),
        ast_helper_sha256=CPU_HELPER_SHA,
        completed_sha256={
            "cold-0": args.cold_completed_sha256,
            "reuse-0": args.same_completed_sha256,
            "reuse-1": args.changed_completed_sha256,
        },
        model_order={arm: values["model-order.json"] for arm, values in arms.items()},
        helper_modules={
            arm: values["protocol.json"]["startup"]["helper_modules"]
            for arm, values in arms.items()
        },
        same_seed_entire_inventory_unchanged=True,
        changed_seed_added_files=sorted(new_files.keys() - same_files.keys()),
        comparisons=comparisons,
        downloaded_code_executed=False,
        pickle_loaded=False,
        scope=(
            "Static code evidence supports the rotary-order explanation; "
            "hash-seed intervention also changes other hash-sensitive behavior."
        ),
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
