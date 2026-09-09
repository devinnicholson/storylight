"""Bounded 54-call dynamic/static/compiled cache screen on fixed training probes."""

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

PROFILE_SHA = "300cd393fc0ba40b1a34270579adb6d4d173ad19bbf6451563f5b94a8adb1ade"
INDICES = (0, 1, 2, 800, 1600, 2400, 3000, 3600)
MODES = ("dynamic", "static-eager", "static-compile")
CAPACITY = 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plan(examples, mode):
    require(len(examples) == len({r["id"] for r in examples}) == 8, "eight_probes_required")
    order = sorted(range(8), key=lambda i: (len(examples[i]["input_ids"]), i))
    longest, shortest = order[-1], order[0]
    sequence = [(-1, longest), (-1, shortest)]
    sequence += [(0, i) for i in [longest, *order[:-1]]]
    sequence += [(1, i) for i in reversed([longest, *order[:-1]])]
    return [
        dict(
            dispatch_ordinal=n,
            mode=mode,
            id=examples[i]["id"],
            case_index=i,
            repetition=rep,
            warmup=rep == -1,
        )
        for n, (rep, i) in enumerate(sequence)
    ]


def generation_options(mode, cache):
    require(mode in MODES and (cache is None) == (mode == "dynamic"), "cache_mode")
    if mode == "dynamic":
        return dict(cache_implementation="dynamic", disable_compile=True)
    return dict(past_key_values=cache, disable_compile=mode == "static-eager")


def reset_cache(cache, torch):
    if cache is None:
        return []
    # Generation initializes these tensors inside inference_mode.
    with torch.inference_mode():
        cache.reset()
    shapes = []
    for layer in cache.layers:
        require(int(layer.get_seq_length()) == 0, "cache_length_not_reset")
        if layer.is_initialized:
            require(
                not bool(torch.count_nonzero(layer.keys))
                and not bool(torch.count_nonzero(layer.values)),
                "cache_values_not_reset",
            )
            shapes.append(
                dict(
                    type=type(layer).__name__,
                    keys=list(layer.keys.shape),
                    values=list(layer.values.shape),
                )
            )
    return shapes


def comparison(records):
    require(len(records) == 54, "incomplete_profile")
    grouped = {}
    for mode in MODES:
        all_rows = [r for r in records if r["mode"] == mode]
        rows = [r for r in all_rows if not r["warmup"]]
        require(len(all_rows) == 18 and len(rows) == 16, "incomplete_mode")
        keyed = {(r["id"], r["repetition"]): r for r in rows}
        require(len(keyed) == 16, "duplicate_observation")
        require(
            all(
                type(r["latency_ms"]) in (int, float)
                and math.isfinite(r["latency_ms"])
                and r["latency_ms"] > 0
                for r in all_rows
            ),
            "latency_value",
        )
        grouped[mode] = keyed
    baseline = grouped["dynamic"]
    result = {}
    for mode, rows in grouped.items():
        require(rows.keys() == baseline.keys(), "unmatched_probes")
        pairs = [(baseline[k], r) for k, r in rows.items()]
        matching = [(a, b) for a, b in pairs if a["token_ids"] == b["token_ids"]]
        by_id = {}
        for row in rows.values():
            by_id.setdefault(row["id"], []).append(row)
        require(
            len(by_id) == 8
            and all({r["repetition"] for r in rs} == {0, 1} for rs in by_id.values()),
            "repetition_scope",
        )
        result[mode] = dict(
            measured_calls=16,
            identical_baseline_token_pairs=len(matching),
            repeat_token_parity=sum(
                rs[0]["token_ids"] == rs[1]["token_ids"] for rs in by_id.values()
            ),
            median_ms=statistics.median(r["latency_ms"] for r in rows.values()),
            max_ms=max(r["latency_ms"] for r in rows.values()),
            matched_median_fraction_reduction=statistics.median(
                1 - b["latency_ms"] / a["latency_ms"] for a, b in matching
            )
            if matching
            else None,
        )
    # Include the cold long/short probes in parity checks, not in warm timing.
    tokens = {}
    for row in records:
        tokens.setdefault(row["id"], []).append(row["token_ids"])
    return dict(
        modes=result,
        all_calls_token_identical=all(
            all(value == values[0] for value in values) for values in tokens.values()
        ),
        quality_accepted=False,
        production_promotion=False,
        scope="eight training probes; cache/reset and greedy token parity, not general accuracy",
    )


class SteppingProcessor:
    def __init__(self, timed, profiler):
        self.timed, self.profiler = timed, profiler

    def __call__(self, ids, scores):
        result = self.timed(ids, scores)
        if self.profiler is not None:
            self.profiler.step()
        return result


def generate(
    model,
    tokenizer,
    example,
    dispatch,
    cache,
    compiled,
    xgr,
    processor_class,
    generator,
    torch,
    helper,
    profiler=None,
):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    ids = helper.chat_tokens(tokenizer, example["messages"], True)
    require(ids == example["input_ids"] and len(ids) + 256 <= CAPACITY, "prompt_changed_or_long")
    reset_started = time.monotonic()
    shapes = reset_cache(cache, torch)
    torch.cuda.synchronize()
    reset_ms = helper.elapsed_ms(reset_started)
    inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
    timed = generator.TimedProcessor(processor_class(compiled))
    processor = SteppingProcessor(timed, profiler)
    torch.cuda.synchronize()
    generation_started = time.monotonic()
    with torch.inference_mode():
        result = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=[1, 106, 50],
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=[processor],
            **generation_options(dispatch["mode"], cache),
        )
    torch.cuda.synchronize()
    generation_ms = helper.elapsed_ms(generation_started)
    tokens = result[0, len(ids) :].tolist()
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    latency_ms = helper.elapsed_ms(started)
    ended = bool(tokens and tokens[-1] in (1, 106, 50))
    return dict(
        **dispatch,
        prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        input_tokens=len(ids),
        token_ids=tokens,
        generated_tokens=len(tokens),
        prediction=text,
        output=text if ended and len(text.encode()) <= 4096 else "INVALID_GENERATION_LIMIT_OR_SIZE",
        terminated_with_eos=ended,
        grammar_accepts_complete_tokens=generator.grammar_accepts(compiled, tokens, xgr),
        latency_ms=latency_ms,
        generation_ms=generation_ms,
        cache_reset_ms=reset_ms,
        reset_verified=cache is not None,
        initialized_cache_shapes=shapes,
        grammar_host_callback_ms=timed.wall_ms,
        grammar_callback_calls=timed.calls,
        profiler_active=profiler is not None,
        **helper.memory(torch),
    )


def execute(args, helper, state):
    wrapper, runner, profile = args.wrapper, args.runner, args.profile
    generator = wrapper.load_module(args.v3_runner, runner.V3_SHA256, "cache_generation")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    state["stage"] = "proofs"
    adapter, proof, training_protocol = runner.checkpoint(
        args.training_run,
        args.completed_sha256,
        profile.TRAINER_SHA,
        args.model_manifest_sha256,
        helper,
        wrapper,
    )
    require(
        training_protocol["screen_inputs_read"] is False
        and training_protocol["fewshot_prompt_sha256"] == wrapper.PROMPT_SHA256,
        "training_scope",
    )
    manifest = helper.pinned_json(
        args.data_dir / "manifest.json", training_protocol["data_manifest_sha256"]
    )
    training = wrapper.read_rows(
        args.data_dir,
        manifest,
        "train-messages.jsonl",
        4800,
        helper,
        ["system", "user", "assistant"],
    )
    rows = [dict(id=training[i]["id"], messages=training[i]["messages"][:2]) for i in INDICES]
    require(helper.digest(args.grammar) == runner.GRAMMAR_SHA256, "grammar_changed")
    helper.verify_model(args.model_dir, args.model_manifest, args.model_manifest_sha256)
    import importlib.metadata

    import bitsandbytes as bnb
    import peft
    import torch
    import transformers
    import xgrammar as xgr
    from safetensors.torch import load_file
    from xgrammar.contrib.hf import LogitsProcessor

    require(importlib.metadata.version("xgrammar") == "0.2.6", "xgrammar_version")
    runtime = helper.runtime_evidence(torch, transformers, peft, bnb)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    examples = [
        {**r, "input_ids": helper.chat_tokens(tokenizer, r["messages"], True)} for r in rows
    ]
    require(all(len(r["input_ids"]) + 256 <= CAPACITY for r in examples), "cache_capacity")
    helper.write_json(
        args.output / "protocol.json",
        dict(
            source_sha256=helper.digest(__file__),
            merge_helper_sha256=PROFILE_SHA,
            trainer_sha256=profile.TRAINER_SHA,
            runner_sha256=profile.RUNNER_SHA,
            generation_helper_sha256=runner.V3_SHA256,
            grammar_sha256=runner.GRAMMAR_SHA256,
            completed_sha256=args.completed_sha256,
            selected_step=proof["step"],
            model_manifest_sha256=args.model_manifest_sha256,
            adapter_files=proof["files"],
            data_manifest_sha256=training_protocol["data_manifest_sha256"],
            input_indices=INDICES,
            examples=rows,
            schedule={m: plan(examples, m) for m in MODES},
            calls=54,
            mode_order=MODES,
            max_runtime_seconds=args.max_runtime_seconds,
            static_capacity=CAPACITY,
            compile_request=dict(mode="reduce-overhead", fullgraph=False, dynamic=None),
            latency_scope="tokenize/reset/generate/decode; excludes load/merge/grammar compile",
            order_limitation="sequential modes/shared host cache; reverse-order screen separate",
            test_data_read=False,
            production_export=False,
            **runtime,
        ),
    )
    started = time.monotonic()
    info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=262144, stop_token_ids=[1, 106, 50]
    )
    compiled = xgr.GrammarCompiler(info).compile_grammar(
        xgr.Grammar.from_ebnf(args.grammar.read_text())
    )
    helper.write_json(args.output / "grammar.json", dict(compile_ms=helper.elapsed_ms(started)))
    records, merged_hashes = [], None
    for mode in MODES:
        state.update(stage="model_load", mode=mode)
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        torch.manual_seed(helper.SEED)
        torch.cuda.manual_seed_all(helper.SEED)
        started = time.monotonic()
        base = transformers.Gemma4ForConditionalGeneration.from_pretrained(
            args.model_dir,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        targets = wrapper.targets(training_protocol["recipe"], base)
        require(
            not any(isinstance(m, bnb.nn.Linear4bit) for m in base.modules())
            and all(
                isinstance(base.get_submodule(n), torch.nn.Linear)
                and base.get_submodule(n).weight.dtype == torch.bfloat16
                for n in targets
            ),
            "bf16_layout",
        )
        model = peft.PeftModel.from_pretrained(
            base, adapter, is_trainable=False, local_files_only=True
        )
        config = model.peft_config["default"]
        require(
            set(config.target_modules) == targets
            and config.r == 16
            and config.lora_alpha == 32
            and config.lora_dropout == 0
            and config.bias == "none"
            and not config.modules_to_save,
            "adapter_config",
        )
        saved = load_file(str(adapter / "adapter_model.safetensors"))
        loaded = peft.get_peft_model_state_dict(model)
        require(
            set(saved) == set(loaded)
            and len(saved) == 2 * len(targets)
            and all(torch.equal(saved[n], loaded[n].cpu()) for n in saved),
            "adapter_tensors",
        )
        del saved, loaded
        torch.cuda.synchronize()
        load_ms = helper.elapsed_ms(started)
        state["stage"] = "verified_merge"
        started = time.monotonic()
        model, hashes = profile.checked_merge(model, torch, helper)
        require(merged_hashes is None or hashes == merged_hashes, "merged_modes_differ")
        merged_hashes = hashes
        torch.cuda.synchronize()
        merge_ms = helper.elapsed_ms(started)
        model.requires_grad_(False)
        model.eval()
        require(
            set(model.generation_config.eos_token_id) == {1, 106, 50}
            and model.get_output_embeddings().weight.shape[0] == 262144,
            "vocabulary",
        )
        identity = runner.parameter_identity(model)
        model.generation_config.compile_config = transformers.CompileConfig(
            mode="reduce-overhead", fullgraph=False, dynamic=None
        )
        cache = (
            None
            if mode == "dynamic"
            else transformers.StaticCache(config=model.config, max_cache_len=CAPACITY)
        )
        trace_evidence = {}

        def trace_ready(prof, mode=mode, trace_evidence=trace_evidence):
            names = [event.name for event in prof.events()]
            trace = args.output / f"trace-{mode}.json"
            prof.export_chrome_trace(str(trace))
            trace_evidence.update(
                trace_sha256=helper.digest(trace),
                cuda_graph_launch_events=sum("cudaGraphLaunch" in n for n in names),
                cuda_launch_kernel_events=sum("cudaLaunchKernel" in n for n in names),
                events=len(names),
            )

        for dispatch in plan(examples, mode):
            state.update(stage="generate", dispatch=dispatch)
            helper.append_json(args.output / "dispatch.jsonl", dispatch)
            tracing = dispatch["dispatch_ordinal"] == 1
            context = (
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1),
                    on_trace_ready=trace_ready,
                )
                if tracing
                else nullcontext(None)
            )
            with context as prof:
                row = generate(
                    model,
                    tokenizer,
                    examples[dispatch["case_index"]],
                    dispatch,
                    cache,
                    compiled,
                    xgr,
                    LogitsProcessor,
                    generator,
                    torch,
                    helper,
                    prof,
                )
            helper.append_json(args.output / "raw.jsonl", row)
            require(
                row["terminated_with_eos"]
                and row["grammar_accepts_complete_tokens"]
                and row["output"] != "INVALID_GENERATION_LIMIT_OR_SIZE",
                "incomplete_output",
            )
            records.append(row)
        require(identity == runner.parameter_identity(model), "parameters_changed")
        counters = {group: dict(values) for group, values in torch._dynamo.utils.counters.items()}
        unique_graphs = counters.get("stats", {}).get("unique_graphs", 0)
        helper.write_json(
            args.output / f"mode-{mode}.json",
            dict(
                model_load_ms=load_ms,
                verified_merge_ms=merge_ms,
                merged_weight_hashes=hashes,
                compile_counters=counters,
                compilation_observed=unique_graphs > 0,
                cuda_graph_launch_observed=trace_evidence.get("cuda_graph_launch_events", 0) > 0,
                trace=trace_evidence,
                preparation_latencies_ms=[
                    r["latency_ms"] for r in records if r["mode"] == mode and r["warmup"]
                ],
                **helper.memory(torch),
            ),
        )
        require(mode != "static-compile" or unique_graphs > 0, "compilation_not_observed")
        del model, base, cache
        gc.collect()
        torch.cuda.empty_cache()
    summary = comparison(records)
    helper.write_json(args.output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
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
    args = parser.parse_args()
    require(
        hashlib.sha256(args.merge_helper.read_bytes()).hexdigest() == PROFILE_SHA, "merge_helper"
    )
    spec = importlib.util.spec_from_file_location("cache_merge_helper", args.merge_helper)
    args.profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(args.profile)
    require(
        hashlib.sha256(args.v5_trainer.read_bytes()).hexdigest() == args.profile.TRAINER_SHA,
        "trainer_pin",
    )
    spec = importlib.util.spec_from_file_location("cache_training_helper", args.v5_trainer)
    args.wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(args.wrapper)
    args.runner = args.wrapper.load_module(args.v5_runner, args.profile.RUNNER_SHA, "cache_runner")
    helper = args.wrapper.load_module(args.v2_trainer, args.wrapper.V2_SHA256, "cache_helper")
    args.wrapper.bounded(args, helper, execute, 900)


if __name__ == "__main__":
    main()
