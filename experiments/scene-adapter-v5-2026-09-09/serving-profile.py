"""Post-selection resident BF16 serving diagnostic on four fixed training inputs.

Each phase loads the same base anew; phase order is explicit. No held-out input,
training, retry, quality selection, merged export, or production change occurs.
"""

import argparse
import gc
import importlib.util
import os
import statistics
from pathlib import Path

TRAINER_SHA = "c77f744bd70196babc4a5edb84c11ffe4f5b785382bc460c6e8f72d809479355"
RUNNER_SHA = "bf567c749842ceede7e97ba21bd740ac5624dd12323809bf94efd34eed698c7b"
INDICES = (0, 2, 2400, 3000)
REPETITIONS = 3


def plan(examples, phase):
    return [
        {
            "id": r["id"],
            "case_index": i,
            "repetition": repetition,
            "phase": phase,
            "warmup": repetition == -1,
            "dispatch_ordinal": ordinal,
        }
        for ordinal, (repetition, i, r) in enumerate(
            (rep, i, r) for rep in range(-1, REPETITIONS) for i, r in enumerate(examples)
        )
    ]


def comparison(records):
    arms = {}
    for phase in ("unmerged", "merged"):
        rows = [r for r in records if r["phase"] == phase and not r["warmup"]]
        if len(rows) != 12 or len({(r["id"], r["repetition"]) for r in rows}) != 12:
            raise ValueError("incomplete_profile")
        arms[phase] = {(r["id"], r["repetition"]): r for r in rows}
    if arms["unmerged"].keys() != arms["merged"].keys():
        raise ValueError("unmatched_inputs")
    paired = [(r, arms["merged"][key]) for key, r in arms["unmerged"].items()]
    matching = [(a, b) for a, b in paired if a["token_ids"] == b["token_ids"]]
    return {
        "measured_pairs": len(paired),
        "identical_token_pairs": len(matching),
        "median_latency_ms": {
            phase: statistics.median(r["latency_ms"] for r in rows.values())
            for phase, rows in arms.items()
        },
        "matched_token_median_fraction_reduction": statistics.median(
            1 - b["latency_ms"] / a["latency_ms"] for a, b in matching
        )
        if matching
        else None,
        "quality_accepted": False,
        "merge_equivalence_established": False,
        "scope": "four training inputs; finite prefix logits and greedy-token parity only",
    }


def checked_merge(model, torch, helper):
    layers = [m for m in model.modules() if hasattr(m, "lora_A")]
    helper.require(bool(layers), "missing_lora_layers")
    expected = []
    with torch.inference_mode():
        for layer in layers:
            helper.require(
                set(layer.lora_A) == {"default"}
                and not layer.lora_variant
                and not layer.lora_bias["default"],
                "unsupported_merge",
            )
            base = layer.get_base_layer()
            weight = base.weight.detach().clone()
            weight += layer.get_delta_weight("default").to(weight.dtype)
            helper.require(bool(torch.isfinite(weight).all()), "nonfinite_merge")
            expected.append((base, helper.parameter_hash({"weight": weight})))
        merged = model.merge_and_unload(safe_merge=True)
    helper.require(not any(hasattr(m, "lora_A") for m in merged.modules()), "remaining_lora")
    helper.require(
        all(helper.parameter_hash({"weight": m.weight}) == sha for m, sha in expected),
        "merged_weight_mismatch",
    )
    return merged, [sha for _, sha in expected]


def execute(args, helper, state):
    wrapper, runner = args.wrapper, args.runner
    generator = wrapper.load_module(args.v3_runner, runner.V3_SHA256, "serving_generation")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    state["stage"] = "proofs"
    adapter, proof, protocol = runner.checkpoint(
        args.training_run,
        args.completed_sha256,
        TRAINER_SHA,
        args.model_manifest_sha256,
        helper,
        wrapper,
    )
    helper.require(
        protocol["screen_inputs_read"] is False
        and protocol["fewshot_prompt_sha256"] == wrapper.PROMPT_SHA256,
        "training_scope",
    )
    manifest = helper.pinned_json(args.data_dir / "manifest.json", protocol["data_manifest_sha256"])
    training = wrapper.read_rows(
        args.data_dir,
        manifest,
        "train-messages.jsonl",
        4800,
        helper,
        ["system", "user", "assistant"],
    )
    rows = [{"id": training[i]["id"], "messages": training[i]["messages"][:2]} for i in INDICES]
    helper.require(helper.digest(args.grammar) == runner.GRAMMAR_SHA256, "grammar_changed")
    helper.verify_model(args.model_dir, args.model_manifest, args.model_manifest_sha256)
    import importlib.metadata

    import bitsandbytes as bnb
    import peft
    import torch
    import transformers
    import xgrammar as xgr
    from safetensors.torch import load_file, save_file
    from xgrammar.contrib.hf import LogitsProcessor

    helper.require(importlib.metadata.version("xgrammar") == "0.2.6", "xgrammar_version")
    runtime = helper.runtime_evidence(torch, transformers, peft, bnb)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    examples = [
        {**r, "input_ids": helper.chat_tokens(tokenizer, r["messages"], True)} for r in rows
    ]
    helper.require(all(len(r["input_ids"]) + 256 <= 2048 for r in examples), "context_overflow")
    phases = (
        ("unmerged", "merged") if args.phase_order == "unmerged-first" else ("merged", "unmerged")
    )
    helper.write_json(
        args.output / "protocol.json",
        {
            "source_sha256": helper.digest(__file__),
            "trainer_sha256": TRAINER_SHA,
            "paired_runner_sha256": RUNNER_SHA,
            "generation_helper_sha256": runner.V3_SHA256,
            "completed_sha256": args.completed_sha256,
            "selected_step": proof["step"],
            "model_manifest_sha256": args.model_manifest_sha256,
            "adapter_files": proof["files"],
            "data_manifest_sha256": protocol["data_manifest_sha256"],
            "input_indices": INDICES,
            "examples": rows,
            "phase_order": phases,
            "schedule": {phase: plan(examples, phase) for phase in phases},
            "max_runtime_seconds": args.max_runtime_seconds,
            "gold_read": False,
            "latency_scope": ("resident tokenization to decoded text; excludes load, merge, "
                              "logits probes, warmups and grammar verification"),
            "order_limitation": ("phases are sequential; a reverse-order process "
                                 "is a separate confirmation"),
            **runtime,
        },
    )
    info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=262144, stop_token_ids=[1, 106, 50]
    )
    compiled = xgr.GrammarCompiler(info).compile_grammar(
        xgr.Grammar.from_ebnf(args.grammar.read_text())
    )
    records, logits_by_phase = [], {}
    for phase in phases:
        state.update(stage="load", phase=phase)
        torch.manual_seed(helper.SEED)
        torch.cuda.manual_seed_all(helper.SEED)
        base = transformers.Gemma4ForConditionalGeneration.from_pretrained(
            args.model_dir,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        targets = wrapper.targets(protocol["recipe"], base)
        helper.require(
            all(
                isinstance(base.get_submodule(n), torch.nn.Linear)
                and base.get_submodule(n).weight.dtype == torch.bfloat16
                for n in targets
            )
            and not any(isinstance(m, bnb.nn.Linear4bit) for m in base.modules()),
            "bf16_layout",
        )
        model = peft.PeftModel.from_pretrained(
            base, adapter, is_trainable=False, local_files_only=True
        )
        config = model.peft_config["default"]
        helper.require(
            set(config.target_modules) == targets
            and config.r == 16
            and config.lora_alpha == 32
            and config.lora_dropout == 0
            and config.bias == "none"
            and not config.modules_to_save,
            "adapter_config",
        )
        saved, loaded = (
            load_file(str(adapter / "adapter_model.safetensors")),
            peft.get_peft_model_state_dict(model),
        )
        helper.require(
            set(saved) == set(loaded)
            and len(saved) == 2 * len(targets)
            and all(torch.equal(saved[n], loaded[n].cpu()) for n in saved),
            "adapter_tensors",
        )
        del saved, loaded
        merge_hashes = []
        if phase == "merged":
            state["stage"] = "merge"
            model, merge_hashes = checked_merge(model, torch, helper)
        model.requires_grad_(False)
        model.eval()
        helper.require(
            model.get_output_embeddings().weight.shape[0] == 262144
            and set(model.generation_config.eos_token_id) == {1, 106, 50},
            "vocabulary",
        )
        identity = runner.parameter_identity(model)
        state["stage"] = "prefix_logits"
        with torch.inference_mode():
            logits = []
            for example in examples:
                ids = torch.tensor([example["input_ids"]], dtype=torch.long, device="cuda")
                values = (
                    model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        use_cache=False,
                        logits_to_keep=1,
                    )
                    .logits[0, -1]
                    .float()
                    .cpu()
                )
                helper.require(bool(torch.isfinite(values).all()), "nonfinite_logits")
                logits.append(values)
            logits_by_phase[phase] = torch.stack(logits)
        save_file(
            {"prefix_logits": logits_by_phase[phase]},
            str(args.output / f"logits-{phase}.safetensors"),
        )
        helper.write_json(
            args.output / f"loaded-{phase}.json",
            {
                "merged_weight_hashes": merge_hashes,
                "adapter_tensors_verified": 2 * len(targets),
                "all_frozen": not any(p.requires_grad for p in model.parameters()),
                **helper.memory(torch),
            },
        )
        for dispatch in plan(examples, phase):
            state.update(stage="generate", dispatch=dispatch)
            helper.append_json(args.output / "dispatch.jsonl", dispatch)
            row = generator.generate_one(
                model,
                tokenizer,
                examples[dispatch["case_index"]],
                {**dispatch, "arm": "constrained"},
                compiled,
                xgr,
                LogitsProcessor,
                torch,
                helper,
            )
            helper.append_json(args.output / "raw.jsonl", row)
            helper.require(
                row["terminated_with_eos"] and row["grammar_accepts_complete_tokens"],
                "incomplete_constrained_output",
            )
            records.append(row)
        helper.require(identity == runner.parameter_identity(model), "parameter_changed")
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
    diff = logits_by_phase["merged"] - logits_by_phase["unmerged"]
    result = comparison(records)
    result["prefix_logits"] = {
        "max_absolute_difference": diff.abs().max().item(),
        "root_mean_square_difference": diff.square().mean().sqrt().item(),
        "argmax_equal": (
            logits_by_phase["merged"].argmax(-1) == logits_by_phase["unmerged"].argmax(-1)
        ).tolist(),
    }
    helper.write_json(args.output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "v5-trainer",
        "v5-runner",
        "v2-trainer",
        "v3-runner",
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
    parser.add_argument(
        "--phase-order", choices=("unmerged-first", "merged-first"), default="unmerged-first"
    )
    parser.add_argument("--max-runtime-seconds", type=int, default=600)
    args = parser.parse_args()
    import hashlib

    if hashlib.sha256(args.v5_trainer.read_bytes()).hexdigest() != TRAINER_SHA:
        raise ValueError("unreviewed_trainer")
    spec = importlib.util.spec_from_file_location("serving_wrapper", args.v5_trainer)
    args.wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(args.wrapper)
    args.runner = args.wrapper.load_module(args.v5_runner, RUNNER_SHA, "serving_runner")
    helper = args.wrapper.load_module(args.v2_trainer, args.wrapper.V2_SHA256, "serving_helper")
    args.wrapper.bounded(args, helper, execute, 600)


if __name__ == "__main__":
    main()
