"""Verify retained V5 training bytes and selection offline; never read quality labels."""

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINER_SHA = "c77f744bd70196babc4a5edb84c11ffe4f5b785382bc460c6e8f72d809479355"
CPU_SHA = "b01b265a556bd31f868fb1dd44eff984ff3cd5994627ffc9631333236db34e17"
SELECTOR_SHA = "378cf2906c2123e900d8a1f591e4f4bd7b6706d09e850a68c5b7da524cdcd090"
TARGET_PROOF_SHA = "04b97da7e94e9933428e929252d7cc828a3b4e2e6b5d23532204acc61e7c0feb"
SHA = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def pinned(path, sha):
    require(isinstance(sha, str) and SHA.fullmatch(sha) and digest(path) == sha, "external pin")
    return read(path)


def same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
        b, sort_keys=True, allow_nan=False
    )


def finite(value, positive=False):
    return (
        type(value) in (float, int)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def accumulated(values):
    # Match the GPU runner's += order; Python's sum may use compensated arithmetic.
    total = 0.0
    for value in values:
        total += value
    return total


def inventory(directory, files, exclude=()):
    require(isinstance(files, dict) and files, "missing inventory")
    for name, sha in files.items():
        relative = Path(name)
        require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and str(relative) == name
            and name not in exclude,
            "unsafe inventory path",
        )
        path = directory / relative
        require(
            path.resolve().is_relative_to(directory.resolve())
            and not path.is_symlink()
            and isinstance(sha, str)
            and SHA.fullmatch(sha)
            and digest(path) == sha,
            "inventory bytes",
        )
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    require(actual == set(files) | set(exclude), "unlisted or missing file")


def context(freeze_path, freeze_sha, cpu_path, cpu_sha):
    freeze, cpu = pinned(freeze_path, freeze_sha), pinned(cpu_path, cpu_sha)
    require(
        freeze["sources"]["train.py"] == TRAINER_SHA
        and freeze["sources"]["cpu-preflight.py"] == CPU_SHA
        and freeze["sources"]["select-recipe.py"] == SELECTOR_SHA
        and freeze["sources"]["results/cpu-preflight.json"] == cpu_sha,
        "source freeze",
    )
    require(digest(HERE / "train.py") == TRAINER_SHA, "local trainer changed")
    spec = importlib.util.spec_from_file_location("verified_v5_training", HERE / "train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    helper = trainer.load_module(
        HERE.parent / "scene-adapter-v2-2026-09-08/train.py",
        trainer.V2_SHA256,
        "verified_v5_parent",
    )
    expected = {
        "schema_version": 1,
        "preflight_sha256": CPU_SHA,
        "trainer_sha256": TRAINER_SHA,
        "v2_helper_sha256": trainer.V2_SHA256,
        "data_manifest_sha256": freeze["sources"]["manifest.json"],
        "development_manifest_sha256": freeze["sources"]["development-manifest.json"],
        "model_manifest_sha256": freeze["model_manifest_sha256"],
        "fewshot_prompt_sha256": trainer.PROMPT_SHA256,
        "all_completion_masks_and_text_roundtrips_verified": True,
        "truncation_used": False,
        "gpu_used": False,
        "model_inference_performed": False,
        "test_inputs_or_gold_read": False,
    }
    require(all(same(cpu[k], value) for k, value in expected.items()), "CPU proof binding")
    require(
        len(cpu["rows"]) == 5056 and len({r["id"] for r in cpu["rows"]}) == 5056,
        "CPU unique examples",
    )
    for split, count in (("training", 4800), ("development", 256)):
        selected = [r for r in cpu["rows"] if r["split"] == split]
        require(len(selected) == count, "CPU split")
        for row in selected:
            require(
                type(row["completion_tokens"]) is int
                and 0 < row["completion_tokens"] <= 256
                and type(row["total_tokens"]) is int
                and row["total_tokens"] <= 2048,
                "CPU token bounds",
            )
    return cpu, trainer, helper


def expected_targets(recipe):
    targets = set()
    for i in range(35):
        prefix = f"model.language_model.layers.{i}."
        targets.add(prefix + "self_attn.q_proj")
        if i < 15:
            targets.add(prefix + "self_attn.v_proj")
        if recipe == "text-linear":
            targets.add(prefix + "self_attn.o_proj")
            if i < 15:
                targets.add(prefix + "self_attn.k_proj")
            targets.update(
                prefix + name
                for name in (
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                    "per_layer_input_gate",
                    "per_layer_projection",
                )
            )
    require(recipe in ("qv", "text-linear"), "recipe")
    return targets


def tensor_shapes(recipe):
    """Shapes from the pinned Gemma4 E2B config, including shared-KV double-width MLPs."""
    result = {}
    for name in expected_targets(recipe):
        layer = int(name.split(".")[3])
        head = 512 if layer % 5 == 4 else 256
        intermediate = 12288 if layer >= 15 else 6144
        leaf = name.rsplit(".", 1)[1]
        inputs, outputs = {
            "q_proj": (1536, 8 * head),
            "k_proj": (1536, head),
            "v_proj": (1536, head),
            "o_proj": (8 * head, 1536),
            "gate_proj": (1536, intermediate),
            "up_proj": (1536, intermediate),
            "down_proj": (intermediate, 1536),
            "per_layer_input_gate": (1536, 256),
            "per_layer_projection": (256, 1536),
        }[leaf]
        result[f"base_model.model.{name}.lora_A.weight"] = (16, inputs)
        result[f"base_model.model.{name}.lora_B.weight"] = (outputs, 16)
    return result


def target_configuration(configured, expected):
    require(isinstance(configured, list) and configured
            and all(isinstance(name, str) and name for name in configured)
            and len(configured) == len(set(configured)), "target configuration type")
    if set(configured) == expected:
        return True
    # PEFT minimizes large lists to suffixes. Expand against every module in
    # the pinned architecture, including non-text modules, before comparing.
    proof = pinned(HERE / "results/target-config-proof.json", TARGET_PROOF_SHA)
    names = proof["module_names"]
    matches = {suffix: {name for name in names
                        if name == suffix or name.endswith("." + suffix)}
               for suffix in configured}
    return all(matches.values()) and set.union(*matches.values()) == expected


def weights(directory, proof, recipe, helper):
    import torch
    from safetensors.torch import load_file

    inventory(directory, proof["files"])
    config = read(directory / "adapter_config.json")
    targets = expected_targets(recipe)
    require(
        target_configuration(config["target_modules"], targets)
        and config["r"] == 16
        and config["lora_alpha"] == 32
        and config["lora_dropout"] == 0
        and config["bias"] == "none"
        and not config["modules_to_save"]
        and not config.get("use_dora", False)
        and not config.get("use_rslora", False)
        and not config.get("rank_pattern")
        and not config.get("alpha_pattern"),
        "adapter configuration",
    )
    saved = load_file(str(directory / "adapter_model.safetensors"), device="cpu")
    expected = tensor_shapes(recipe)
    require(
        set(saved) == set(expected) and same(proof["tensor_count"], len(expected)),
        "tensor inventory",
    )
    for name, tensor in saved.items():
        require(
            tensor.dtype == torch.float32
            and tuple(tensor.shape) == expected[name]
            and bool(torch.isfinite(tensor).all()),
            "adapter tensor",
        )
    require(
        sum(t.numel() for t in saved.values()) == (2678784 if recipe == "qv" else 26165248),
        "adapter parameter count",
    )
    normalized = {
        name.replace(".lora_A.", ".lora_A.default.").replace(".lora_B.", ".lora_B.default."): tensor
        for name, tensor in saved.items()
    }
    require(
        helper.parameter_hash(normalized) == proof["final_adapter_sha256"], "saved parameter hash"
    )


def verify_run(directory, complete_sha, cpu, trainer, helper):
    complete = pinned(directory / "completed.json", complete_sha)
    require(not (directory / "failure.json").exists(), "failed run")
    inventory(directory, complete["files"], ("completed.json",))
    protocol = read(directory / "protocol.json")
    steps, recipe = protocol["max_steps"], protocol["recipe"]
    require(type(steps) is int and type(complete["completed_steps"]) is int, "step type")
    schedule = trainer.checkpoint_steps(steps)
    settings = {
        "runner_sha256": TRAINER_SHA,
        "helper_sha256": trainer.V2_SHA256,
        "model_id": helper.MODEL_ID,
        "revision": helper.REVISION,
        **{
            k: cpu[k]
            for k in (
                "model_manifest_sha256",
                "data_manifest_sha256",
                "development_manifest_sha256",
                "fewshot_prompt_sha256",
            )
        },
        "training_rows": 4800,
        "development_rows": 256,
        "checkpoint_steps": list(schedule),
        "seed": helper.SEED,
        "max_length": 2048,
        "max_new_tokens": 256,
        "batch_size": 1,
        "gradient_accumulation": 1,
        "rank": 16,
        "alpha": 32,
        "dropout": 0,
        "learning_rate": 0.0001,
        "optimizer": "AdamW",
        "quantization": "nf4-double-quant-bf16",
        "loss": "completion-only",
        "fresh_base": True,
        "screen_inputs_read": False,
        "evaluation_gold_read": False,
        "production_export": False,
        "selection": "lowest token-weighted development completion loss, earliest step on tie",
    }
    require(all(same(protocol[k], v) for k, v in settings.items()), "training configuration")
    require(
        set(protocol["versions"]) == set(helper.VERSIONS)
        and all(protocol["versions"][k].split("+")[0] == v for k, v in helper.VERSIONS.items())
        and protocol["cuda"] == "12.8"
        and "L4" in protocol["device"],
        "GPU runtime",
    )
    require(
        complete["completed_steps"] == steps
        and complete["recipe"] == recipe
        and complete["screen_inputs_read"] is False
        and complete["quality_accepted"] is False
        and complete["production_export"] is False
        and complete["training_ms_includes_development_and_checkpointing"] is True,
        "completion scope",
    )
    require(
        type(protocol["max_runtime_seconds"]) is int
        and finite(complete["wall_seconds"], True)
        and complete["wall_seconds"] <= protocol["max_runtime_seconds"] <= 7200,
        "finite training lifetime",
    )
    train = [r for r in cpu["rows"] if r["split"] == "training"]
    dev = [r for r in cpu["rows"] if r["split"] == "development"]
    require(
        protocol["max_train_tokens"] == max(r["total_tokens"] for r in train)
        and protocol["max_development_tokens"] == max(r["total_tokens"] for r in dev),
        "training context maxima",
    )
    order = list(range(len(train)))
    random.Random(helper.SEED).shuffle(order)
    raw = rows(directory / "training.jsonl")
    require(len(raw) == steps, "training row count")
    for step, (row, index) in enumerate(zip(raw, order[:steps], strict=True), 1):
        require(
            type(row["step"]) is int
            and row["step"] == step
            and row["id"] == train[index]["id"]
            and type(row["supervised_tokens"]) is int
            and row["supervised_tokens"] == train[index]["completion_tokens"]
            and finite(row["loss"])
            and finite(row["gradient_norm_before_clip"], True)
            and finite(row["latency_ms"], True),
            "training order, mask or loss",
        )
    require(
        complete["training_compute_ms"] == accumulated(r["latency_ms"] for r in raw)
        and finite(complete["training_ms"], True)
        and complete["training_compute_ms"]
        <= complete["training_ms"]
        <= complete["wall_seconds"] * 1000,
        "training timing",
    )
    loaded = read(directory / "loaded.json")
    targets = expected_targets(recipe)
    initial = loaded["initial_adapter_sha256"]
    require(
        isinstance(initial, str)
        and SHA.fullmatch(initial)
        and loaded["targets"] == sorted(targets)
        and loaded["trainable_parameters"] == (2678784 if recipe == "qv" else 26165248),
        "loaded adapter",
    )
    require(recipe != "qv" or initial == trainer.INITIAL_SHA256, "Q/V initialization")
    require(loaded["expected_initial_adapter_sha256"] in (None, initial), "initial pin")
    require(
        steps != 4800 or loaded["expected_initial_adapter_sha256"] == initial,
        "missing full initialization pin",
    )
    losses = []
    for step in schedule:
        raw_dev = rows(directory / f"development-{step}.jsonl")
        require(len(raw_dev) == len(dev), "development row count")
        for row, expected in zip(raw_dev, dev, strict=True):
            require(
                type(row["step"]) is int
                and row["step"] == step
                and row["id"] == expected["id"]
                and type(row["supervised_tokens"]) is int
                and row["supervised_tokens"] == expected["completion_tokens"]
                and finite(row["completion_loss"]),
                "development identity or loss",
            )
        count = sum(r["supervised_tokens"] for r in raw_dev)
        loss = accumulated(r["supervised_tokens"] * r["completion_loss"] for r in raw_dev) / count
        aggregate = read(directory / f"development-{step}.json")
        require(
            same(aggregate["step"], step)
            and same(aggregate["rows"], 256)
            and same(aggregate["supervised_tokens"], count)
            and aggregate["token_weighted_loss"] == loss
            and finite(aggregate["latency_ms"], True),
            "development aggregate",
        )
        losses.append(aggregate)
        proof = read(directory / f"checkpoint-{step}.json")
        require(
            same(proof["step"], step)
            and proof["initial_adapter_sha256"] == initial
            and finite(proof["delta_l2"], True)
            and proof["final_adapter_sha256"] != initial,
            "checkpoint identity or delta",
        )
        weights(directory / f"adapter-{step}", proof, recipe, helper)
    selection = read(directory / "selection.json")
    selected = trainer.select_checkpoint(losses, schedule, helper)
    require(
        same(selection["development"], losses)
        and same(selection["selected_step"], selected)
        and same(complete["selected_step"], selected)
        and selection["checkpoint_receipt_sha256"]
        == digest(directory / f"checkpoint-{selected}.json")
        and selection["selection_inputs"] == "development completion loss only"
        and selection["selected_adapter_reloaded_and_verified"] is True
        and selection["screen_generation_started"] is False,
        "selected checkpoint",
    )
    require(
        complete["training_compute_ms"] + sum(r["latency_ms"] for r in losses)
        <= complete["training_ms"],
        "development time outside training",
    )
    return {
        "recipe": recipe,
        "steps": steps,
        "selected_step": selected,
        "development_loss": next(r["token_weighted_loss"] for r in losses if r["step"] == selected),
        "initial_adapter_sha256": initial,
        "completed_sha256": complete_sha,
        "selected_adapter_sha256": digest(
            directory / f"adapter-{selected}/adapter_model.safetensors"
        ),
    }


def verify(args):
    cpu, trainer, helper = context(
        args.freeze, args.freeze_sha256, args.cpu_preflight, args.cpu_preflight_sha256
    )
    result = verify_run(args.directory, args.completed_sha256, cpu, trainer, helper)
    pilots = []
    if result["steps"] == 4800:
        require(
            all(
                getattr(args, name, None)
                for name in (
                    "qv_pilot",
                    "qv_completed_sha256",
                    "text_linear_pilot",
                    "text_linear_completed_sha256",
                    "recipe_selection",
                    "recipe_selection_sha256",
                )
            ),
            "full run requires both pilots and selector proof",
        )
        for recipe in ("qv", "text-linear"):
            prefix = recipe.replace("-", "_")
            pilot = verify_run(
                getattr(args, prefix + "_pilot"),
                getattr(args, prefix + "_completed_sha256"),
                cpu,
                trainer,
                helper,
            )
            require(pilot["recipe"] == recipe and pilot["steps"] == 1200, "pilot recipe")
            pilots.append(
                {
                    "recipe": recipe,
                    "selected_pilot_step": pilot["selected_step"],
                    "development_loss": pilot["development_loss"],
                    "pilot_completed_sha256": pilot["completed_sha256"],
                    "initial_adapter_sha256": pilot["initial_adapter_sha256"],
                }
            )
        chosen = min(pilots, key=lambda r: (r["development_loss"], r["recipe"] != "qv"))
        selection = pinned(args.recipe_selection, args.recipe_selection_sha256)
        require(
            selection["selector_sha256"] == SELECTOR_SHA
            and selection["trainer_sha256"] == TRAINER_SHA
            and same(selection["candidates"], pilots)
            and same(selection["selected"], chosen)
            and selection["test_inputs_or_gold_read"] is False
            and selection["quality_accepted"] is False,
            "recipe selection",
        )
        expected_common = {
            k: cpu[k]
            for k in (
                "model_manifest_sha256",
                "data_manifest_sha256",
                "development_manifest_sha256",
                "fewshot_prompt_sha256",
            )
        }
        expected_common.update(seed=helper.SEED, learning_rate=0.0001, rank=16, alpha=32)
        require(
            same(selection["common_protocol"], expected_common)
            and selection["selection_rule"]
            == "lowest selected pilot development loss; Q/V on exact tie"
            and selection["next_run"] == "fresh initialization, full 4800-example pass",
            "selector protocol",
        )
        require(
            result["recipe"] == chosen["recipe"]
            and result["initial_adapter_sha256"] == chosen["initial_adapter_sha256"],
            "full run differs from selected pilot",
        )
    return {
        "schema_version": 1,
        "verified": True,
        "verifier_sha256": digest(Path(__file__)),
        "freeze_sha256": args.freeze_sha256,
        "cpu_preflight_sha256": args.cpu_preflight_sha256,
        "recipe_selection_sha256": getattr(args, "recipe_selection_sha256", None),
        "training": result,
        "pilots": pilots,
        "quality_accepted": False,
        "scope": "Retained bytes and receipts verified; no re-execution of GPU optimization, "
        "no recomputation of initial tensors or delta L2, no quality inference.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "freeze", "cpu-preflight", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("completed", "freeze", "cpu-preflight"):
        parser.add_argument(f"--{name}-sha256", required=True)
    for name in ("qv-pilot", "text-linear-pilot", "recipe-selection"):
        parser.add_argument(f"--{name}", type=Path)
    for name in ("qv-completed", "text-linear-completed", "recipe-selection"):
        parser.add_argument(f"--{name}-sha256")
    args = parser.parse_args()
    result = verify(args)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
