# Repeated-process compiler-cache experiment review

A fixed Python hash seed and identical wrapper imports can test whether compiler artifacts survive a process restart. This is an experiment on startup reproducibility. The previous miss does not establish that hash randomization alone caused it: the previous warm wrapper also changed imported module names.

The installed Gemma4 source constructs sets for rotary layer types and unique text-model layer types. Iteration over the latter constructs position embeddings before decoding. An independent local Python 3.14.6 check produced `full_attention, sliding_attention` in two processes with seed 0, and the reverse order in two processes with seed 1. The GPU interpreter must record its own result; the local observation is not a GPU cache-hit result.

Before dispatch:

- Set `PYTHONHASHSEED=0` in the shell before launching Python. Setting the environment inside the running interpreter is too late.
- Use the same wrapper source, imported module names, model/runtime versions, checked merged weights, probes, cache shapes, and generation settings in both processes.
- First process uses new compiler directories; second uses exactly those directories. Hash the first output and the existing compiler inventory before starting the second.
- Retain the unchanged 54-call schedule, all token streams, load and first compiled preparation costs, full process wall time, compiler counters, and CUDA traces. Keep finite process deadlines and exclusive output directories.

Acceptance requires all 54 outputs to match within and across runs, stable cache addresses within each process, complete token/grammar replay, and a measured reduction in first compiled preparation cost. Record actual graph-cache/AOT-cache hits and misses; a populated directory or successful generation does not prove a hit. Inspect retained generated argument order to see whether the proposed source of instability was removed.

A successful two-process result supports the effectiveness of a stable launch configuration. To attribute the change specifically to the hash seed, run a separate process with the same wrapper/import names and a seed that demonstrably reverses the layer order. Keep its result even if it does not support the hypothesis.

The compiled phase follows dynamic and eager phases in the current benchmark. Its first preparation call is not the first request of a newly started service. Any report must retain that distinction and avoid claiming cold-start latency is solved from compiler-cache reuse alone.
