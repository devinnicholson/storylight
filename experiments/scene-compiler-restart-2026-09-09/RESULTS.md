# Compiler reuse across fresh processes

Keeping the wrapper and Python hash seed stable allowed the second process to reuse both AOT and FX graph caches. Its first compiled preparation request fell from **109.25 seconds to 22.49 seconds**, a **79.41% reduction**, with all 54 output token sequences preserved. Changing the hash seed in the third process reversed the model's recorded layer/buffer order, restored both graph-cache misses and raised preparation to **102.61 seconds**, again preserving every output. This supports a hash-seed-sensitive reuse mechanism under the stable launcher.

| Measurement | Fresh directories, seed 0 | Existing directories, seed 0 | Existing directories, seed 1 |
| --- | ---: | ---: | ---: |
| First compiled preparation request | 109.251 s | 22.492 s | 102.613 s |
| AOT graph cache hits / misses | 0 / 2 | 2 / 0 | 0 / 2 |
| FX graph cache hits / misses | 0 / 2 | 2 / 0 | 0 / 2 |
| Compiler files before / after | 0 / 992 | 992 / 992 | 992 / 998 |
| Whole benchmark process | 300.074 s | 210.451 s | 291.201 s |
| Outputs matching previous process | Baseline | 54/54 | 54/54 |

All three arms independently passed actual tokenizer, EOS and grammar replay for all 162 calls. Outputs also matched across dynamic, bridged eager and bridged compiled modes within each process. The loaded GPU model recorded the same attention-layer and rotary-buffer ordering in the two seed-0 processes. The second process's compiler inventory remained byte-for-byte identical to the first process's final inventory, according to the retained runtime hashes.

The first compiled preparation timer includes generation and follows dynamic and eager modes, which already warm the model and GPU. It excludes model loading and adapter merging, each approximately 3.5–3.6 seconds per process. It is neither an isolated compilation timer nor the first request of a cold service. The whole-process measurement covers all 54 benchmark calls, not a production startup path. Twenty-two seconds of remaining preparation is still substantial.

These are eight fixed training probes, one three-process sequence, one model revision and one L4 environment. The unchanged wrapper holds helper module names constant; only the third process changes the hash seed. The verified control changes the GPU model's layer order from full/sliding attention to sliding/full attention and reverses rotary-buffer registration order. Cache hits disappear and six new compiler files appear. That controlled pattern implicates hash-sensitive execution state, but it does not isolate rotary ordering from every other hash-sensitive operation or establish the cause of every preceding unseeded miss.

No new quality score was computed. Runtime parity does not repair the adapter's failed refusal gate, and the live demo remains unchanged. Evidence is retained in `results/verified-cold-0.json`, `results/verified-reuse-0.json`, `results/verified-reuse-1.json` and the corresponding raw run directories.

After all required artifacts were downloaded, the experiment VM, boot disk and dedicated firewall rules were verified absent at **23:18:40 UTC on September 9**. Creation through confirmed absence spans 1,898.273441 seconds. The public VM/disk/IPv4 list-rate estimate is **$0.46**, excluding transfer, earlier allocations and other workloads; it is not an invoice or account spending total. See `cloud-01/cleanup.json` and `cloud-01/lifecycle-estimate.json`.

Subsequent static analysis of the retained generated decoder partitions found that swapping only the two rotary-argument names makes both changed-seed partitions' syntax trees match their originals. The comparison was independently reproduced without executing downloaded compiler code. [GENERATED-ORDER.md](GENERATED-ORDER.md) records this additional support for the ordering explanation and its limits.
