# Compiler artifact reuse: preparation fell from 109.25 to 22.49 seconds

Keeping the wrapper and Python hash seed stable allowed a fresh process to reuse PyTorch compiler artifacts for the Gemma scene decoder. The first compiled preparation request fell from 109.251 to 22.492 seconds, a 79.41 percent reduction. All 54 outputs matched the preceding process token for token.

| Measurement | Fresh cache, seed 0 | Reused cache, seed 0 | Reused cache, seed 1 |
| --- | ---: | ---: | ---: |
| First compiled preparation | 109.251 s | 22.492 s | 102.613 s |
| AOTAutograd hits / misses | 0 / 2 | 2 / 0 | 0 / 2 |
| FX graph hits / misses | 0 / 2 | 2 / 0 | 0 / 2 |
| Compiler files before / after | 0 / 992 | 992 / 992 | 992 / 998 |
| Outputs matching prior process | baseline | 54/54 | 54/54 |

The controlled sequence ran three fresh Python processes against the same merged model and eight fixed training probes. Each process made 54 calls across three recorded decode modes. Independent replay checked all 162 token streams against the tokenizer and the combined grammar/EOS rules.

## The cache hierarchy

`torch.compile` does not produce one indivisible cache entry. Dynamo captures Python-level execution, AOTAutograd prepares graph partitions, Inductor lowers them, and Triton or CUDA code supplies lower-level GPU kernels. The experiment preserved the compiler directories between processes and recorded cache counters at the AOT and FX graph layers.

The first seed-0 process began with empty directories and wrote 992 files. The second process used the same wrapper and module names under `PYTHONHASHSEED=0`. It recorded two AOT hits plus two FX hits, added no files, and cut the first compiled preparation by 86.759 seconds. The compiler inventory remained byte-identical.

The third process changed only the Python hash seed to 1 while keeping the on-disk artifacts. Preparation returned to 102.613 seconds, with two misses at both graph layers. Six files appeared. The model recorded its full-attention and sliding-attention layers in the opposite order, along with reversed rotary-buffer registration. Static analysis of the generated decoder partitions found that swapping the two rotary argument names made the changed-seed syntax trees match the originals.

Stable tensor shapes were insufficient; Python-level ordering changed the captured graph's identity even though all generated tokens remained the same. For compiled inference, wrapper source, symbol names, container versions, hash seed, model revision, and input-shape policy form an execution contract.

## Measurement boundary

The preparation timer includes generation and follows dynamic and eager calls that have already warmed the model and GPU. Model loading and adapter merging, each about 3.5 to 3.6 seconds in this sequence, are excluded. The 22.492-second value is therefore a compiled preparation phase inside a warm process, not service cold start.

The experiment used one L4 environment and one three-process cycle. It supports a hash-sensitive graph-cache explanation, while rotary ordering has not been isolated from every other operation affected by the seed. The eight probes are previously exposed training inputs.

## What we learned and what is next

The same-seed restart retained exact output parity and removed most of the compilation wait without changing model weights or the decode algorithm. That is the operational win: compiler reuse can turn a two-minute preparation event into roughly 22 seconds when the graph identity stays stable.

The next implementation should record a cache manifest beside every built artifact, including software revisions, GPU capability, shape bounds, hash seed, and hashes for the captured wrapper. CI can reject an artifact when any field changes. A repeated cross-node study should then separate filesystem persistence from driver-level and GPU-level warm state, with multiple restarts for a latency distribution.

Evidence: [`research/results.json`](../../research/results.json), [`restart results`](../../experiments/scene-compiler-restart-2026-09-09/RESULTS.md), and [`generated-order analysis`](../../experiments/scene-compiler-restart-2026-09-09/GENERATED-ORDER.md).
