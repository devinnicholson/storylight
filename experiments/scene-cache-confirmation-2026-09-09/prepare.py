"""Freeze answer-free inputs and cache capacity proof; no model or gold loading."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V5 = ROOT / "experiments/scene-adapter-v5-2026-09-09"
TOKENIZER = Path("/private/tmp/bookforge-v3-tokenizer")
INPUT_SHA = "79150ddb559cde4056e32f435fd0b2271f2c466597cb3beed3793554ea7e8ea8"
TRAIN_SHA = "96a566c1d53f1e6c02607b188a09caece3f40307a2e3f24e33f78d80b2baa1e9"
MODEL_SHA = "703bbb89d61aaed083846d7cb3d4ee1a68220e25de93a035b1f4b49d24062f2d"
COMPLETE_SHA = "66326bd05dd934feae4ec1375a7b004f6efb0d267c0e0be72bfd3881f781768c"
ADAPTER_SHA = "8acf87bdb245a4e1f92e91fc3b2495b9e5c5492a5038fd7d2432f93c6e980123"
MODES = ("dynamic", "bridge-compile")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def pinned(path, sha):
    require(digest(path) == sha, "changed input")
    return path.read_text()


def write(name, value):
    with (HERE / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    import transformers

    require(transformers.__version__ == "5.13.0", "tokenizer version")
    screen = list(map(json.loads, pinned(V5 / "screen-inputs.jsonl", INPUT_SHA).splitlines()))
    training = list(map(json.loads, pinned(V5 / "train-messages.jsonl", TRAIN_SHA).splitlines()))
    require(len(screen) == 128 and len(training) == 4800, "source scope")
    warmups = [{key: r[key] for key in ("id", "messages")} for r in training[:2]]
    warmups = [{**r, "messages": r["messages"][:2]} for r in warmups]
    rows = warmups + screen
    require(len({r["id"] for r in rows}) == 130, "duplicate IDs")
    require(
        all(
            set(r) == {"id", "messages"}
            and [m["role"] for m in r["messages"]] == ["system", "user"]
            and all(set(m) == {"role", "content"} for m in r["messages"])
            for r in rows
        ),
        "answer-free message scope",
    )
    model = json.loads(pinned(V5 / "results/gpu/model-manifest.json", MODEL_SHA))
    token_files = {name: sha for name, sha in model["files"].items() if name != "model.safetensors"}
    for name, sha in token_files.items():
        pinned(TOKENIZER / name, sha)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        TOKENIZER, local_files_only=True, trust_remote_code=False
    )
    tokens = [
        tokenizer.apply_chat_template(
            r["messages"],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for r in rows
    ]
    require(all(len(ids) + 256 <= 1024 for ids in tokens), "capacity exceeded")
    plan = {}
    for mode_index, mode in enumerate(MODES):
        order = list(range(2, 130))
        if mode == "bridge-compile":
            order.reverse()
        plan[mode] = [
            dict(
                dispatch_ordinal=n,
                global_ordinal=mode_index * 130 + n,
                mode=mode,
                id=rows[index]["id"],
                case_index=index,
                repetition=-1 if n < 2 else 0,
                warmup=n < 2,
            )
            for n, index in enumerate([0, 1, *order])
        ]
    with (HERE / "inputs.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    protocol = dict(
        schema_version=1,
        source_sha256=digest(Path(__file__)),
        screen_input_sha256=INPUT_SHA,
        training_input_sha256=TRAIN_SHA,
        input_sha256=digest(HERE / "inputs.jsonl"),
        model_manifest_sha256=MODEL_SHA,
        training_completed_sha256=COMPLETE_SHA,
        adapter_sha256=ADAPTER_SHA,
        selected_step=2400,
        warmup_training_indices=[0, 1],
        input_rows=130,
        measured_rows=128,
        measured_calls=256,
        warmup_calls=4,
        calls=260,
        modes=list(MODES),
        schedule=plan,
        static_capacity=1024,
        max_new_tokens=256,
        eos_token_ids=[1, 106, 50],
        do_sample=False,
        attention="sdpa",
        dtype="bfloat16",
        merged=True,
        same_resident_base=True,
        dynamic_cache_fresh_per_request=True,
        static_cache_reused_with_verified_reset=True,
        grammar_both_arms=True,
        grammar_sha256="622bc2bf36a2182c21a7e8a889f292f9b10a91dd149ee0aa913dd70ad20404df",
        bridge_sha256="e42f6ca4a24d60a756c29122ea323b151ae9d89b4631a6fbba63e64adfd31a31",
        generation_reference_sha256="b308c7e236ef81b30915fe416639a6b8032e7a5faa2b94a422754082d62a4939",
        compile_request=dict(mode="reduce-overhead", fullgraph=False, dynamic=None),
        fresh_local_compiler_directories=True,
        max_runtime_seconds=1600,
        supervisor_timeout_seconds=1660,
        provider_deadline_utc="2026-09-09T23:31:21Z",
        conditional_on_prior_compiled_bridge_parity=True,
        runner_requires_separate_source_review=True,
        held_out=False,
        gold_used_for_selection=False,
        gold_uploaded=False,
        training=False,
        production_promotion=False,
        primary_gate_unchanged=True,
        scope="Previously exposed runtime parity confirmation; no fresh quality gate",
        timing_scope="Retain cold warmups separately; compare identical measured token sequences",
        order_limitation="Dynamic input order then compiled reverse input order; one repetition",
    )
    write("protocol.json", protocol)
    write(
        "cpu-preflight.json",
        dict(
            source_sha256=digest(Path(__file__)),
            protocol_sha256=digest(HERE / "protocol.json"),
            input_sha256=digest(HERE / "inputs.jsonl"),
            tokenizer_files=token_files,
            tokenizer_version=transformers.__version__,
            no_gold_read=True,
            no_model_load=True,
            no_inference=True,
            all_completion_free_inputs=True,
            unique_inputs=130,
            min_prompt_tokens=min(map(len, tokens)),
            max_prompt_tokens=max(map(len, tokens)),
            max_with_generation_budget=max(map(len, tokens)) + 256,
            capacity=1024,
            rows=[
                dict(
                    id=r["id"],
                    prompt_tokens=len(ids),
                    prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                )
                for r, ids in zip(rows, tokens, strict=True)
            ],
        ),
    )


if __name__ == "__main__":
    main()
