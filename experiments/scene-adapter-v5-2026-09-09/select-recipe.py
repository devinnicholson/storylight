"""Select a V5 pilot recipe using independently authored development loss only."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(args):
    if digest(args.trainer) != args.trainer_sha256:
        raise ValueError("trainer changed")
    spec = importlib.util.spec_from_file_location("v5_pilot_trainer", args.trainer)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    helper = trainer.load_module(args.v2_trainer, trainer.V2_SHA256, "v5_pilot_helper")
    candidates = []
    common = None
    for recipe, directory, sha in (("qv", args.qv, args.qv_sha256),
                                   ("text-linear", args.text_linear, args.text_linear_sha256)):
        complete = helper.pinned_json(directory / "completed.json", sha)
        helper.require(complete["completed_steps"] == 1200
                       and complete["recipe"] == recipe, "pilot_incomplete")
        helper.require({"protocol.json", "selection.json", "loaded.json"}
                       <= set(complete["files"]) and not (directory / "failure.json").exists(),
                       "pilot_proof_missing_or_failed")
        for name, expected in complete["files"].items():
            path = Path(name)
            helper.require(not path.is_absolute() and ".." not in path.parts, "unsafe_path")
            helper.require(digest(directory / path) == expected, "pilot_file_changed")
        protocol = json.loads((directory / "protocol.json").read_text())
        helper.require(protocol["runner_sha256"] == args.trainer_sha256
                       and protocol["recipe"] == recipe and protocol["max_steps"] == 1200,
                       "pilot_protocol")
        identity = {k: protocol[k] for k in (
            "model_manifest_sha256", "data_manifest_sha256", "development_manifest_sha256",
            "fewshot_prompt_sha256", "seed", "learning_rate", "rank", "alpha")}
        helper.require(common is None or identity == common, "unmatched_pilots")
        common = identity
        selection = json.loads((directory / "selection.json").read_text())
        step = trainer.select_checkpoint(selection["development"],
                                         trainer.checkpoint_steps(1200), helper)
        helper.require(step == selection["selected_step"] == complete["selected_step"]
                       and selection["selected_adapter_reloaded_and_verified"] is True
                       and selection["screen_generation_started"] is False,
                       "pilot_selection")
        loss = next(r["token_weighted_loss"] for r in selection["development"] if r["step"] == step)
        loaded = json.loads((directory / "loaded.json").read_text())
        candidates.append({"recipe": recipe, "selected_pilot_step": step,
                           "development_loss": loss, "pilot_completed_sha256": sha,
                           "initial_adapter_sha256": loaded["initial_adapter_sha256"]})
    chosen = min(candidates, key=lambda row: (row["development_loss"], row["recipe"] != "qv"))
    helper.write_json(args.output, {
        "selector_sha256": digest(Path(__file__)), "trainer_sha256": args.trainer_sha256,
        "selection_rule": "lowest selected pilot development loss; Q/V on exact tie",
        "candidates": candidates, "selected": chosen, "common_protocol": common,
        "next_run": "fresh initialization, full 4800-example pass",
        "test_inputs_or_gold_read": False, "quality_accepted": False})
    print(json.dumps(chosen, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trainer", "v2-trainer", "qv", "text-linear", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("trainer", "qv", "text-linear"):
        parser.add_argument(f"--{name}-sha256", required=True)
    select(parser.parse_args())


if __name__ == "__main__":
    main()
