"""Paired old/new adapters on one BF16 base, both constrained by the frozen V3 grammar."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import statistics
import time
from pathlib import Path

V3_SHA256 = "8a62d71cf77ce7647c5e14b9bed84c868d1da44530bdcbb2f45354322dd25d50"
GRAMMAR_SHA256 = "622bc2bf36a2182c21a7e8a889f292f9b10a91dd149ee0aa913dd70ad20404df"
OLD_COMPLETED_SHA256 = "ec562987e5c1349ec092b058dc3d649f0d76ed6568649a7a1d2b4956582da35d"
OLD_ADAPTER_SHA256 = "d8127b1421e5aa9bfae01efe6f2b1d9646a88d59fc25666be71ecca2508162ba"
OLD_TRAINER_SHA256 = "67cc69d6851eca9c9bc4052dd9e883923544dbf3e68daa22895ae0268c62e4ea"
ARMS = ("old", "new")


def schedule(rows, preflight=False):
    plan = []
    for index, row in enumerate(rows):
        for repetition in range(1 if preflight else 2):
            arms = ARMS if (index + repetition) % 2 == 0 else ARMS[::-1]
            for arm in arms:
                plan.append({"dispatch_ordinal": len(plan), "case_index": index,
                             "id": row["id"], "repetition": repetition, "arm": arm})
    return plan


def parameter_identity(model):
    return {n: (id(p), p._version) for n, p in model.named_parameters()}


def select_adapter(model, arm, identity, helper):
    model.set_adapter(arm)
    model.requires_grad_(False)
    model.eval()
    helper.require(model.active_adapters == [arm]
                   and not any(p.requires_grad for p in model.parameters())
                   and identity == parameter_identity(model), "adapter_selection_integrity")


def checkpoint(directory, completed_sha, runner_sha, model_sha, helper, wrapper, old=False):
    complete = helper.pinned_json(directory / "completed.json", completed_sha)
    helper.require(complete["completed_steps"] == (1200 if old else 4800), "training_incomplete")
    files = complete["files"]
    for name, sha in files.items():
        relative = Path(name)
        helper.require(not relative.is_absolute() and ".." not in relative.parts,
                       "unsafe_training_path")
        helper.require(helper.digest(directory / name) == sha, "training_file_changed")
    helper.require({"protocol.json", "selection.json", "loaded.json"} <= set(files),
                   "training_proof_missing")
    protocol = json.loads((directory / "protocol.json").read_text())
    helper.require(protocol["runner_sha256"] == runner_sha
                   and protocol["model_manifest_sha256"] == model_sha, "training_protocol")
    selection = json.loads((directory / "selection.json").read_text())
    step = (helper.select_checkpoint(selection["development"]) if old else
            wrapper.select_checkpoint(selection["development"],
                                      wrapper.checkpoint_steps(4800), helper))
    helper.require(step == selection["selected_step"] == complete["selected_step"]
                   and selection["selected_adapter_reloaded_and_verified"] is True
                   and selection["screen_generation_started"] is False, "selection_proof")
    name = f"checkpoint-{step}.json"
    helper.require(files[name] == selection["checkpoint_receipt_sha256"], "checkpoint_binding")
    proof = json.loads((directory / name).read_text())
    tensor_count = 100 if old or protocol["recipe"] == "qv" else 550
    helper.require(proof["step"] == step and proof["tensor_count"] == tensor_count
                   and math.isfinite(proof["delta_l2"]) and proof["delta_l2"] > 0,
                   "checkpoint_delta")
    adapter = directory / f"adapter-{step}"
    helper.require({str(p.relative_to(adapter)) for p in adapter.rglob("*") if p.is_file()}
                   == set(proof["files"]), "adapter_inventory")
    for name, sha in proof["files"].items():
        helper.require(files[f"adapter-{step}/{name}"] == sha
                       and helper.digest(adapter / name) == sha, "adapter_file")
    return adapter, proof, protocol


def inputs(args, wrapper, helper):
    old = checkpoint(args.old_run, OLD_COMPLETED_SHA256, OLD_TRAINER_SHA256,
                     args.model_manifest_sha256, helper, wrapper, old=True)
    helper.require(old[1]["step"] == 800
                   and old[1]["files"]["adapter_model.safetensors"] == OLD_ADAPTER_SHA256,
                   "old_adapter_changed")
    new = checkpoint(args.new_run, args.new_completed_sha256, args.v5_trainer_sha256,
                     args.model_manifest_sha256, helper, wrapper)
    helper.require(new[2]["helper_sha256"] == wrapper.V2_SHA256
                   and new[2]["screen_inputs_read"] is False
                   and new[2]["fewshot_prompt_sha256"] == wrapper.PROMPT_SHA256,
                   "new_training_scope")
    loaded = json.loads((args.new_run / "loaded.json").read_text())
    helper.require(loaded["initial_adapter_sha256"] == loaded["expected_initial_adapter_sha256"]
                   and loaded["initial_adapter_sha256"] == args.pilot_initial_sha256,
                   "new_initial_adapter_changed")
    if args.preflight_only:
        manifest = helper.pinned_json(args.data_dir / "manifest.json",
                                      new[2]["data_manifest_sha256"])
        train = wrapper.read_rows(args.data_dir, manifest, "train-messages.jsonl", 4800, helper,
                                  ["system", "user", "assistant"])
        selected = [next(r for r in train if r["messages"][-1]["content"].startswith("V2\n")),
                    next(r for r in train if r["messages"][-1]["content"] == "REFUSE")]
        rows = [{"id": r["id"], "messages": r["messages"][:2]} for r in selected]
        input_sha = manifest["files"]["train-messages.jsonl"]["sha256"]
    else:
        manifest = helper.pinned_json(args.screen_dir / "screen-manifest.json",
                                      args.screen_manifest_sha256)
        helper.require(manifest["training_manifest_sha256"] == new[2]["data_manifest_sha256"]
                       and manifest["development_manifest_sha256"]
                       == new[2]["development_manifest_sha256"], "screen_training_binding")
        rows = wrapper.read_rows(args.screen_dir, manifest, "screen-inputs.jsonl", 128, helper,
                                  ["system", "user"])
        input_sha = manifest["files"]["screen-inputs.jsonl"]["sha256"]
    for row in rows:
        helper.require(hashlib.sha256(row["messages"][0]["content"].encode()).hexdigest()
                       == new[2]["fewshot_prompt_sha256"], "prompt_changed")
    helper.require(helper.digest(args.grammar) == GRAMMAR_SHA256, "grammar_changed")
    return rows, {"old": old, "new": new}, {
        "input_sha256": input_sha, "data_manifest_sha256": new[2]["data_manifest_sha256"],
        "fewshot_prompt_sha256": new[2]["fewshot_prompt_sha256"],
        "development_manifest_sha256": new[2]["development_manifest_sha256"]}


def execute(args, helper, state):
    wrapper = args.wrapper
    generator = wrapper.load_module(args.v3_runner, V3_SHA256, "v5_generation_parent")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    state["stage"] = "input_proofs"
    rows, checkpoints, pins = inputs(args, wrapper, helper)
    helper.verify_model(args.model_dir, args.model_manifest, args.model_manifest_sha256)
    import bitsandbytes as bnb
    import peft
    import torch
    import transformers
    import xgrammar as xgr
    from safetensors.torch import load_file
    from xgrammar.contrib.hf import LogitsProcessor

    helper.require(importlib.metadata.version("xgrammar") == "0.2.6", "xgrammar_version")
    runtime = helper.runtime_evidence(torch, transformers, peft, bnb)
    runtime["versions"]["xgrammar"] = "0.2.6"
    grammar_sources = (("hf_processor", LogitsProcessor), ("matcher", xgr.GrammarMatcher),
                       ("compiler", xgr.GrammarCompiler), ("tokenizer", xgr.TokenizerInfo))
    runtime["xgrammar_sources"] = {n: helper.digest(inspect.getsourcefile(v))
                                  for n, v in grammar_sources}
    torch.manual_seed(helper.SEED)
    torch.cuda.manual_seed_all(helper.SEED)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False)
    examples = [{**r, "input_ids": helper.chat_tokens(tokenizer, r["messages"], True)}
                for r in rows]
    helper.require(all(len(r["input_ids"]) + 256 <= 2048 for r in examples), "context_overflow")
    plan = schedule(rows, args.preflight_only)
    helper.write_json(args.output / "protocol.json", {
        "runner_sha256": helper.digest(__file__), "v5_trainer_sha256": args.v5_trainer_sha256,
        "v2_helper_sha256": wrapper.V2_SHA256, "v3_helper_sha256": V3_SHA256,
        "model_manifest_sha256": args.model_manifest_sha256, "grammar_sha256": GRAMMAR_SHA256,
        "old_completed_sha256": OLD_COMPLETED_SHA256,
        "new_completed_sha256": args.new_completed_sha256,
        "screen_manifest_sha256": args.screen_manifest_sha256,
        "selected_steps": {a: p[1]["step"] for a, p in checkpoints.items()},
        "adapter_files": {a: p[1]["files"] for a, p in checkpoints.items()},
        "schedule": plan, "calls": len(plan), "repetitions": 1 if args.preflight_only else 2,
        "preflight_only": args.preflight_only, "arms": list(ARMS),
        "max_runtime_seconds": args.max_runtime_seconds, "max_new_tokens": 256,
        "greedy": True, "precision": "one-bf16-base-two-unmerged-adapters",
        "grammar_both_arms": True, "gold_read": False, "production_export": False,
        "latency_scope": "adapter selection and freezing plus tokenization through decoded text",
        **pins, **runtime})
    state["stage"] = "model_load"
    started = time.monotonic()
    base = transformers.Gemma4ForConditionalGeneration.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa")
    arm_targets = {a: wrapper.targets(p[2].get("recipe", "qv"), base)
                   for a, p in checkpoints.items()}
    modules = {n: m for n, m in base.named_modules() if n in set.union(*arm_targets.values())}
    helper.require(set(modules) == set.union(*arm_targets.values())
                   and not any(isinstance(m, bnb.nn.Linear4bit) for m in base.modules())
                   and all(isinstance(m, torch.nn.Linear) and m.weight.dtype == torch.bfloat16
                           for m in modules.values()), "bf16_layout")
    model = peft.PeftModel.from_pretrained(base, checkpoints["old"][0], adapter_name="old",
                                         is_trainable=False, local_files_only=True)
    model.load_adapter(checkpoints["new"][0], adapter_name="new", is_trainable=False,
                       local_files_only=True)
    for arm in ARMS:
        config = model.peft_config[arm]
        helper.require(set(config.target_modules) == arm_targets[arm] and config.r == 16
                       and config.lora_alpha == 32 and config.lora_dropout == 0
                       and config.bias == "none" and not config.modules_to_save, "adapter_config")
        saved = load_file(str(checkpoints[arm][0] / "adapter_model.safetensors"))
        loaded = peft.get_peft_model_state_dict(model, adapter_name=arm)
        helper.require(set(saved) == set(loaded) and len(saved) == 2 * len(arm_targets[arm])
                       and all(torch.equal(saved[n], loaded[n].cpu()) for n in saved),
                       "adapter_tensors")
        normalized = {n.replace(f".{arm}.", ".default."): p for n, p in model.named_parameters()
                      if f".lora_A.{arm}." in n or f".lora_B.{arm}." in n}
        expected = checkpoints[arm][1]["final_adapter_sha256"]
        helper.require(helper.parameter_hash(normalized) == expected,
                       "adapter_parameter_hash")
    model.requires_grad_(False)
    model.eval()
    identity = parameter_identity(model)
    torch.cuda.synchronize()
    load_ms = helper.elapsed_ms(started)
    eos = model.generation_config.eos_token_id
    helper.require(model.get_output_embeddings().weight.shape[0] == 262144
                   and isinstance(eos, list) and set(eos) == {1, 106, 50}, "vocabulary_eos")
    state["stage"] = "grammar_compile"
    started = time.monotonic()
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=262144, stop_token_ids=eos)
    compiled = xgr.GrammarCompiler(info).compile_grammar(
        xgr.Grammar.from_ebnf(args.grammar.read_text()))
    dtypes = {str(p.dtype) for p in model.parameters()}
    helper.write_json(args.output / "loaded.json", {
        "model_load_ms": load_ms, "grammar_compile_ms": helper.elapsed_ms(started),
        "vocab_size": 262144, "stop_token_ids": eos,
        "verified_adapter_tensors": sum(2 * len(t) for t in arm_targets.values()),
        "unmerged": True, "all_frozen": True,
        "dtype_elements": {d: sum(p.numel() for p in model.parameters() if str(p.dtype) == d)
                           for d in sorted(dtypes)},
        **helper.memory(torch)})
    records = []
    for dispatch in plan:
        state.update(stage="generation", dispatch=dispatch)
        helper.append_json(args.output / "dispatch.jsonl", dispatch)
        started = time.monotonic()
        select_adapter(model, dispatch["arm"], identity, helper)
        torch.cuda.synchronize()
        selection_ms = helper.elapsed_ms(started)
        row = generator.generate_one(
            model, tokenizer, examples[dispatch["case_index"]], {**dispatch, "arm": "constrained"},
            compiled, xgr, LogitsProcessor, torch, helper)
        row.update(arm=dispatch["arm"], adapter_selection_ms=selection_ms,
                   latency_ms=row["latency_ms"] + selection_ms)
        helper.append_json(args.output / "raw.jsonl", row)
        helper.require(not row["terminated_with_eos"] or row["grammar_accepts_complete_tokens"],
                       "constrained_grammar_failure")
        if args.preflight_only:
            helper.require(row["terminated_with_eos"], "preflight_incomplete")
        records.append(row)
    helper.require(identity == parameter_identity(model),
                   "post_generation_parameter_change")
    helper.write_json(args.output / "predictions.json", {
        "schema_version": 1, "input_sha256": pins["input_sha256"],
        "provenance": {"implementation": f"V5 paired frozen adapters @{helper.digest(__file__)}",
                       "runner_sha256": helper.digest(__file__), "grammar_sha256": GRAMMAR_SHA256,
                       "screen_manifest_sha256": args.screen_manifest_sha256,
                       "old_completed_sha256": OLD_COMPLETED_SHA256,
                       "new_completed_sha256": args.new_completed_sha256,
                       "raw_sha256": helper.digest(args.output / "raw.jsonl"),
                       "preflight_only": args.preflight_only,
                       "evaluation_gold_read": False, **pins},
        "records": [{k: r[k] for k in ("id", "arm", "repetition", "output", "latency_ms")}
                    for r in records]})
    return {"calls": len(records), "preflight_only": args.preflight_only,
            "arms": {a: {"calls": sum(r["arm"] == a for r in records),
                          "median_latency_ms": statistics.median(r["latency_ms"] for r in records
                                                                 if r["arm"] == a)} for a in ARMS},
            "quality_accepted": False, "production_export": False,
            "callback_timing_scope": "host elapsed includes synchronization and GPU dispatch"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v5-trainer", "v2-trainer", "v3-runner", "model-dir", "model-manifest",
                 "old-run", "new-run", "grammar", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("v5-trainer", "model-manifest", "new-completed"):
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--screen-dir", type=Path)
    parser.add_argument("--screen-manifest-sha256")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=int, default=3000)
    parser.add_argument("--pilot-initial-sha256", required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.v5_trainer.read_bytes()).hexdigest() != args.v5_trainer_sha256:
        raise ValueError("unreviewed V5 trainer")
    spec = importlib.util.spec_from_file_location("v5_trainer", args.v5_trainer)
    args.wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(args.wrapper)
    helper = args.wrapper.load_module(args.v2_trainer, args.wrapper.V2_SHA256, "v5_parent")
    helper.require((args.preflight_only and args.data_dir and not args.screen_dir
                    and not args.screen_manifest_sha256)
                   or (not args.preflight_only and args.screen_dir and args.screen_manifest_sha256
                       and not args.data_dir), "input_mode")
    args.wrapper.bounded(args, helper, execute, 3600)


if __name__ == "__main__":
    main()
