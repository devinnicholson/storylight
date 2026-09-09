"""Actual cached-tokenizer completion-mask check; training/development inputs only."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

GRAMMAR_SHA256 = "622bc2bf36a2182c21a7e8a889f292f9b10a91dd149ee0aa913dd70ad20404df"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trainer", "v2-trainer", "data-dir", "development-dir", "tokenizer",
                 "model-manifest", "grammar", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("trainer", "data-manifest", "development-manifest", "model-manifest"):
        parser.add_argument(f"--{name}-sha256", required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.trainer.read_bytes()).hexdigest() != args.trainer_sha256:
        raise ValueError("trainer pin")
    spec = importlib.util.spec_from_file_location("v5_cpu_trainer", args.trainer)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    helper = trainer.load_module(args.v2_trainer, trainer.V2_SHA256, "v5_cpu_parent")
    helper.require(not args.output.exists(), "output_exists")
    train, dev, prompt_sha = trainer.data(args, helper)
    model = helper.pinned_json(args.model_manifest, args.model_manifest_sha256)
    helper.require(model["model_id"] == helper.MODEL_ID and model["revision"] == helper.REVISION,
                   "model_identity")
    names = {"chat_template.jinja", "config.json", "generation_config.json",
             "tokenizer.json", "tokenizer_config.json"}
    helper.require({p.name for p in args.tokenizer.iterdir()} == names, "tokenizer_inventory")
    files = {n: helper.digest(args.tokenizer / n) for n in sorted(names)}
    helper.require(all(sha == model["files"][n] for n, sha in files.items()), "tokenizer_bytes")
    version = importlib.metadata.version("transformers")
    helper.require(version == "5.13.0", "tokenizer_version")
    helper.require(importlib.metadata.version("xgrammar") == "0.2.6", "grammar_version")
    helper.require(helper.digest(args.grammar) == GRAMMAR_SHA256, "grammar_source")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import transformers
    import xgrammar as xgr

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False)
    eos = json.loads((args.tokenizer / "generation_config.json").read_text())["eos_token_id"]
    helper.require(set(eos) == {1, 106, 50}, "stop_tokens")
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=262144, stop_token_ids=eos)
    compiled = xgr.GrammarCompiler(info).compile_grammar(
        xgr.Grammar.from_ebnf(args.grammar.read_text()))
    rows = []
    for split, examples in (("training", train), ("development", dev)):
        for row in examples:
            example = helper.completion_example(tokenizer, row)
            helper.require(example["prompt_tokens"] + 256 <= 2048, "full_generation_context")
            target = tokenizer.encode(row["messages"][-1]["content"], add_special_tokens=False)
            helper.require(len(target) + 1 <= 256, "target_stop_budget")
            for stop in eos:
                matcher = xgr.GrammarMatcher(compiled)
                helper.require(all(matcher.accept_token(t) for t in [*target, stop])
                               and matcher.is_terminated(), "target_grammar_rejection")
            rows.append({"id": row["id"], "split": split,
                         "total_tokens": len(example["input_ids"]),
                         "prompt_tokens": example["prompt_tokens"],
                         "completion_tokens": example["completion_tokens"],
                         "target_tokens_including_stop": len(target) + 1,
                         "token_and_mask_sha256": hashlib.sha256(json.dumps(
                             {k: example[k] for k in ("input_ids", "labels")},
                             sort_keys=True).encode()).hexdigest()})
    helper.write_json(args.output, {
        "schema_version": 1, "preflight_sha256": helper.digest(__file__),
        "trainer_sha256": args.trainer_sha256, "v2_helper_sha256": trainer.V2_SHA256,
        "data_manifest_sha256": args.data_manifest_sha256,
        "development_manifest_sha256": args.development_manifest_sha256,
        "model_manifest_sha256": args.model_manifest_sha256,
        "tokenizer_files": files, "transformers_version": version,
        "grammar_sha256": GRAMMAR_SHA256, "xgrammar_version": "0.2.6",
        "target_token_sequences_accepted_with_each_stop": eos,
        "fewshot_prompt_sha256": prompt_sha, "rows": rows,
        "maximum_total_tokens": max(r["total_tokens"] for r in rows),
        "maximum_completion_tokens": max(r["completion_tokens"] for r in rows),
        "all_completion_masks_and_text_roundtrips_verified": True,
        "truncation_used": False, "gpu_used": False, "model_inference_performed": False,
        "test_inputs_or_gold_read": False})


if __name__ == "__main__":
    main()
