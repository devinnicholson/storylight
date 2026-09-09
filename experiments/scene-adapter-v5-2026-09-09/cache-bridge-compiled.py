"""Bounded dynamic versus bridged eager/compiled decode on eight fixed training probes."""

import argparse
import hashlib
import importlib.util
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

PROFILE_SHA = "300cd393fc0ba40b1a34270579adb6d4d173ad19bbf6451563f5b94a8adb1ade"
INDICES = (0, 1, 2, 800, 1600, 2400, 3000, 3600)
CAPACITY = 1024
BRIDGE_SHA = "e42f6ca4a24d60a756c29122ea323b151ae9d89b4631a6fbba63e64adfd31a31"


def require(condition, message):
    if not condition:
        raise ValueError(message)


MODES = ("dynamic", "bridge-eager", "bridge-compile")


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


class SteppingProcessor:
    def __init__(self, timed, profiler):
        self.timed, self.profiler = timed, profiler

    def __call__(self, ids, scores):
        result = self.timed(ids, scores)
        if self.profiler is not None:
            self.profiler.step()
        return result


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
            bridge_sha256=BRIDGE_SHA,
            merge_helper_sha256=PROFILE_SHA,
            trainer_sha256=profile.TRAINER_SHA,
            runner_sha256=profile.RUNNER_SHA,
            completed_sha256=args.completed_sha256,
            selected_step=proof["step"],
            model_manifest_sha256=args.model_manifest_sha256,
            adapter_files=proof["files"],
            data_manifest_sha256=training_protocol["data_manifest_sha256"],
            grammar_sha256=runner.GRAMMAR_SHA256,
            examples=rows,
            input_indices=INDICES,
            calls=54,
            schedule={mode: plan(examples, mode) for mode in MODES},
            compile_request=dict(mode="reduce-overhead", fullgraph=False, dynamic=None),
            compiler_cache_preparation=args.compiler_cache_preparation,
            static_capacity=CAPACITY,
            max_runtime_seconds=args.max_runtime_seconds,
            test_data_read=False,
            scope="Fifty-four full generations; fixed training probes, no quality acceptance",
            **runtime,
        ),
    )
    info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=262144, stop_token_ids=[1, 106, 50]
    )
    compiled = xgr.GrammarCompiler(info).compile_grammar(
        xgr.Grammar.from_ebnf(args.grammar.read_text())
    )
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
    model = peft.PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
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
    helper.write_json(
        args.output / "model.json",
        dict(model_load_ms=load_ms, verified_merge_ms=merge_ms, merged_weight_hashes=hashes),
    )
    import inspect

    helper.write_json(
        args.output / "runtime-sources.json",
        {
            name: dict(path=inspect.getfile(obj), sha256=helper.digest(inspect.getfile(obj)))
            for name, obj in {
                "model": type(model),
                "generation": transformers.GenerationMixin,
                "cache": transformers.StaticCache,
                "masking": transformers.masking_utils.create_causal_mask,
            }.items()
        },
    )
    bridge = wrapper.load_module(args.bridge_helper, BRIDGE_SHA, "dynamic_static_bridge")
    records = []
    model.generation_config.compile_config = transformers.CompileConfig(
        mode="reduce-overhead", fullgraph=False, dynamic=None
    )
    for mode in MODES:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        static = (
            None
            if mode == "dynamic"
            else transformers.StaticCache(config=model.config, max_cache_len=CAPACITY)
        )
        trace_evidence = {}

        def trace_ready(prof, mode=mode, evidence=trace_evidence):
            names = [event.name for event in prof.events()]
            path = args.output / f"trace-{mode}.json"
            prof.export_chrome_trace(str(path))
            evidence.update(
                trace_sha256=helper.digest(path),
                cuda_graph_launch_events=sum("cudaGraphLaunch" in n for n in names),
                cuda_launch_kernel_events=sum("cudaLaunchKernel" in n for n in names),
            )

        for dispatch in plan(examples, mode):
            ordinal, index = dispatch["dispatch_ordinal"], dispatch["case_index"]
            state.update(stage="generate", dispatch=dispatch)
            helper.append_json(args.output / "dispatch.jsonl", dispatch)
            example = examples[index]
            ids = torch.tensor([example["input_ids"]], device="cuda", dtype=torch.long)
            timed = generator.TimedProcessor(LogitsProcessor(compiled))
            context = (
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(
                        wait=1 if mode == "dynamic" else 2, warmup=0, active=1, repeat=1
                    ),
                    on_trace_ready=trace_ready,
                )
                if ordinal == 1
                else nullcontext(None)
            )
            torch.cuda.synchronize()
            started = time.monotonic()
            prefix_ms, transfer_ms, transfer_receipt = None, None, None
            with context as profiler, torch.inference_mode():
                processor = SteppingProcessor(timed, profiler)
                if mode == "dynamic":
                    result = model.generate(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        max_new_tokens=256,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                        eos_token_id=[1, 106, 50],
                        pad_token_id=tokenizer.pad_token_id,
                        logits_processor=[processor],
                        cache_implementation="dynamic",
                        disable_compile=True,
                    )
                else:
                    dynamic = transformers.DynamicCache(
                        config=model.config.get_text_config(decoder=True)
                    )
                    mask = torch.ones_like(ids)
                    prepared = model.prepare_inputs_for_generation(
                        ids,
                        past_key_values=dynamic,
                        attention_mask=mask,
                        position_ids=mask.long().cumsum(-1) - 1,
                        use_cache=True,
                        logits_to_keep=1,
                        is_first_iteration=True,
                    )
                    prefetched = model(**prepared, return_dict=True)
                    logits = prefetched.logits[:, -1, :].float().clone()
                    first = processor(ids, logits).argmax(dim=-1, keepdim=True)
                    require(int(first.item()) not in (1, 106, 50), "unexpected_first_eos")
                    torch.cuda.synchronize()
                    prefix_ms = helper.elapsed_ms(started)
                    transfer_started = time.monotonic()
                    bridge.transfer(dynamic, static, torch)
                    torch.cuda.synchronize()
                    transfer_ms = helper.elapsed_ms(transfer_started)
                    transfer_receipt = dict(
                        source_lengths=[int(layer.get_seq_length()) for layer in dynamic.layers],
                        target_lengths=[int(layer.get_seq_length()) for layer in static.layers],
                        target_shapes=[list(layer.keys.shape) for layer in static.layers],
                        target_addresses=[
                            (layer.keys.data_ptr(), layer.values.data_ptr())
                            for layer in static.layers
                        ],
                        first_token=int(first.item()),
                    )
                    del dynamic, prepared, prefetched, logits
                    combined = torch.cat([ids, first], dim=-1)
                    result = model.generate(
                        input_ids=combined,
                        attention_mask=torch.ones_like(combined),
                        max_new_tokens=255,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                        eos_token_id=[1, 106, 50],
                        pad_token_id=tokenizer.pad_token_id,
                        logits_processor=[processor],
                        past_key_values=static,
                        disable_compile=mode != "bridge-compile",
                    )
                    del combined, first
            torch.cuda.synchronize()
            latency_ms = helper.elapsed_ms(started)
            tokens = result[0, len(example["input_ids"]) :].tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            ended = bool(tokens and tokens[-1] in (1, 106, 50))
            accepts = generator.grammar_accepts(compiled, tokens, xgr)
            row = dict(
                **dispatch,
                input_tokens=len(example["input_ids"]),
                prompt_sha256=hashlib.sha256(json.dumps(example["input_ids"]).encode()).hexdigest(),
                token_ids=tokens,
                generated_tokens=len(tokens),
                prediction=text,
                terminated_with_eos=ended,
                grammar_accepts_complete_tokens=accepts,
                latency_ms=latency_ms,
                prefix_ms=prefix_ms,
                transfer_ms=transfer_ms,
                transfer_receipt=transfer_receipt,
                grammar_callback_calls=timed.calls,
                grammar_host_callback_ms=timed.wall_ms,
            )
            helper.append_json(args.output / "raw.jsonl", row)
            records.append(row)
            require(
                ended and accepts and len(text.encode()) <= 4096 and len(tokens) <= 256,
                "incomplete_generation",
            )
        counters = {group: dict(values) for group, values in torch._dynamo.utils.counters.items()}
        helper.write_json(
            args.output / f"mode-{mode}.json",
            dict(
                compile_counters=counters,
                trace=trace_evidence,
                compilation_observed=counters.get("stats", {}).get("unique_graphs", 0) > 0,
                cuda_graph_launch_observed=trace_evidence.get("cuda_graph_launch_events", 0) > 0,
            ),
        )
        require(
            mode != "bridge-compile" or counters.get("stats", {}).get("unique_graphs", 0) > 0,
            "compilation_not_observed",
        )
        del static
    require(identity == runner.parameter_identity(model), "parameters_changed")
    comparisons = []
    for i in range(8):
        for rep in (0, 1):
            rows = {
                mode: next(
                    r
                    for r in records
                    if r["case_index"] == i and r["mode"] == mode and r["repetition"] == rep
                )
                for mode in MODES
            }
            comparisons.append(
                dict(
                    case_index=i,
                    repetition=rep,
                    id=examples[i]["id"],
                    identical_tokens=all(
                        row["token_ids"] == rows["dynamic"]["token_ids"] for row in rows.values()
                    ),
                    latencies_ms={mode: row["latency_ms"] for mode, row in rows.items()},
                )
            )
    all_tokens = {}
    for row in records:
        all_tokens.setdefault(row["id"], []).append(row["token_ids"])
    summary = dict(
        calls=len(records),
        comparisons=comparisons,
        all_tokens_identical=all(r["identical_tokens"] for r in comparisons),
        all_calls_token_identical=all(
            all(v == values[0] for v in values) for values in all_tokens.values()
        ),
        quality_accepted=False,
        production_promotion=False,
    )
    helper.write_json(args.output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "bridge-helper",
        "compiler-cache-root",
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
        args.compiler_cache_root.is_absolute()
        and not args.compiler_cache_root.exists()
        and not args.compiler_cache_root.is_symlink(),
        "fresh compiler cache root",
    )
    args.compiler_cache_root.mkdir(exist_ok=False)
    locations = {
        "TORCHINDUCTOR_CACHE_DIR": args.compiler_cache_root / "inductor",
        "TRITON_CACHE_DIR": args.compiler_cache_root / "triton",
    }
    for name, path in locations.items():
        path.mkdir(exist_ok=False)
        os.environ[name] = str(path)
    args.compiler_cache_preparation = {name: str(path) for name, path in locations.items()}

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
