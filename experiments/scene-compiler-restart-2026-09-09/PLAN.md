# Compiler cache reuse across processes

The preceding restart retained its compiler files but still missed both AOT and
FX graph caches. Its first compiled preparation request fell only from 111.35 to
103.45 seconds. Inspection found that both generated decoder partitions swap the
two rotary-frequency arguments; after renaming those arguments, their ASTs match.
The pinned Gemma implementation creates and iterates sets of attention-layer types.

The separate CPU proof instantiates the actual model on the meta device in four
fresh processes. Seeds 0, 1, 0 and 1 reproduce stable order within a seed and
reversed order across the two seeds, including rotary-buffer registration order.
It does not load weights or execute downloaded compiler artifacts. This supports
a specific cache-key hypothesis, but the previous GPU restart also changed helper
module names. The new experiment holds wrapper source and module names constant.

## Fixed three-process experiment

Each arm runs the unchanged `cache-bridge-compiled.py` execution: eight training
probes, two preparation calls and sixteen measured calls per mode, 54 calls total.
The model, checked BF16 adapter merge, grammar, cache-transfer helper and generation
settings remain pinned. Each process has a 900-second internal bound.

1. **cold-0:** launch Python with `PYTHONHASHSEED=0`, creating new local compiler
   directories. This establishes the baseline and cache artifacts.
2. **reuse-0:** launch the same wrapper in a fresh Python process with seed 0 and
   exactly the previous compiler inventory. Compare every output with cold-0.
3. **reuse-1:** launch the same wrapper in a fresh Python process with seed 1 and
   exactly reuse-0's final inventory. Compare every output with reuse-0.

The seed is set before Python starts. The wrapper records Python version, hash
metadata, helper module names and the actual loaded model's layer/buffer order.
It checks the order on the GPU runtime rather than assuming that a CPU proof using
a different Python version transfers unchanged. Cache inventories reject symlinks
and require unchanged files between arms. Prior completion receipts are pinned.

## Interpretation

Report all first compiled preparation times, AOT/FX graph hit/miss counters,
compiler inventories and token comparisons. A faster same-seed restart with graph
cache hits, followed by changed-order misses under seed 1, supports a seed-sensitive
reuse mechanism under the stable launcher. It does not isolate rotary ordering
from every other hash-sensitive operation. Failure to obtain that pattern must
remain in the results; it is not a reason to discard or relabel an arm.

Every completed token stream must retain grammar/EOS validity and match its paired
baseline before a serving benefit is claimed. The first compiled request includes
generation and follows dynamic/eager modes that warm the model and GPU. It is not
whole-service startup, isolated compilation time or a fresh-device cold start.
This experiment does not modify the live demo or cure the adapter's failed refusal
gate. A device integration or general-purpose seed setting is a separate decision.

## Environment reproduction

The new allocation uses the unchanged V5 `setup.sh`. Its unpinned transitive dependency resolved to `regex==2026.9.10`; before any arm, it was restored with `.venv/bin/python -m pip install regex==2026.9.3`. The complete package list then matched the previous allocation byte for byte (SHA-256 `681a5407958e8afe08b5c69498c4692fe7533ad366c3b7a95e2e9a630974bde9`). All three arms use that same environment. `setup/review.json` retains the model, driver and package checks.
