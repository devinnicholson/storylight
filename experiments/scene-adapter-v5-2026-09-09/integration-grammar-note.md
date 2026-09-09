# Local grammar decoding feasibility

**The existing Jetson service cannot execute V5's constrained decoder unchanged.**
The INT4-AWQ/NVMe-PLE export precedent establishes a plausible memory path, but
Edge-LLM v0.10.0 lacks the required grammar interface in its deployed server and
sampling path. Passing a quality gate would not remove this integration blocker.
This September 9 source audit performed no export, inference, device access or
deployment. See [the separate memory/export assessment](integration-feasibility.md).

The [installer](../../deploy/jetson/install-tensorrt-edge-llm.sh) pins NVIDIA
`71dd1bae032e70771265917ec74d3ff4cad07a10` (v0.10.0); the
[launcher](../../deploy/jetson/run-tensorrt-edge-server.sh) runs its
`experimental.server` on loopback. Thirteen public source files inspected here
matched that revision's Git tree blob hashes. Conclusions concern this pinned
path, not every future Edge-LLM release.

| Requirement | Exact supported primitive or missing feature |
| --- | --- |
| V2 graph or whole-response `REFUSE` EBNF | Missing from Edge-LLM's request interface. [`SamplingParams` and `_build_request`](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/71dd1bae032e70771265917ec74d3ff4cad07a10/experimental/server/engine.py#L84) expose sampling, stop strings and fixed `logit_bias`; there is no grammar or per-token logits-processor field. The [HTTP request construction](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/71dd1bae032e70771265917ec74d3ff4cad07a10/experimental/server/api_server.py#L1038) likewise does not forward an EBNF `response_format`. |
| Prefix-dependent token mask | Missing in both required locations: [prefill sampling](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/71dd1bae032e70771265917ec74d3ff4cad07a10/cpp/runtime/llmInferenceRuntime.cpp#L1773) and [vanilla decode sampling](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/71dd1bae032e70771265917ec74d3ff4cad07a10/cpp/runtime/decoding/vanillaDecoder.cpp#L112) apply static bias then sample/argmax. The Python bias map is fixed per request, at most 1,024 entries with finite values from −100 to 100. It cannot implement a changing hard mask over 262,144 tokens. |
| EOS `[1,106,50]` | Supported in principle: [engine config parsing](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/71dd1bae032e70771265917ec74d3ff4cad07a10/cpp/runtime/config/llmEngineConfig.cpp#L729) reads an EOS array, and the runtime adds it to [tokenizer EOS membership](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/71dd1bae032e70771265917ec74d3ff4cad07a10/cpp/tokenizer/tokenizer.h#L255). A new export must prove the resulting set is exactly the intended set; this audit does not establish that the old engine contains token 50. Disable thinking and `EDGELLM_IGNORE_EOS` for this contract. |
| 256 generated tokens | Edge-LLM accepts `max_tokens=256`, but [runtime capacity handling](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/71dd1bae032e70771265917ec74d3ff4cad07a10/cpp/runtime/llmInferenceRuntime.cpp#L991) can shorten it for KV capacity. The [old build](../../deploy/jetson/build-gemma4-tensorrt-edge-engine.sh) uses input 1,280/KV 1,536; the [slot client](../../src/bookforge/tensorrt_slot_client.py) separately caps output at 128 and uses another prompt. Exact V5 prompt tokenization and capacity need verification. |

NVIDIA documents an executable EBNF API in **TensorRT-LLM**, using
`guided_decoding_backend: xgrammar` and
`response_format={"type":"ebnf","ebnf":...}`. That is a different package and
runtime; its [guided-decoding documentation](https://nvidia.github.io/TensorRT-LLM/1.2.0/features/guided-decoding.html#ebnf-grammar)
does not establish compatibility with this Edge-LLM engine, Jetson memory strategy
or Gemma export. Edge-LLM's own [server documentation](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/examples/experimental-server.html)
documents the narrower sampling interface. JSON-only output would also change
the frozen V2/REFUSE wire contract.

The concrete local path would require a **new C++ runtime integration**, not a
request flag: compile the [frozen EBNF](../scene-adapter-v3-2026-09-09/grammar.ebnf)
against the exact tokenizer, keep one fresh matcher per request, apply its hard
mask before the first and every subsequent argmax, and advance it with each
selected token. EOS must remain masked until grammar completion; all three EOS
IDs must terminate correctly. Start with vanilla decoding and full vocabulary;
speculative acceptance and vocabulary reduction require additional proofs.
This is an implementation proposal, not an available tested extension.

Before exporting a candidate, prove that integration with CPU tokenizer/mask
replay and a substitute-engine test, including Unicode token boundaries, each
EOS, length exhaustion, cancellation and request-state reset. Then separately
verify actual quantized on-device output, memory and latency. Keep the original
source/privacy/schema validators and treat non-EOS or incomplete output as a
failed parse, never `REFUSE`. Post-validating unconstrained output is possible in
principle but would be a different decoding experiment; it does not reproduce
V5's constrained results or qualify the learned fallback for activation.
