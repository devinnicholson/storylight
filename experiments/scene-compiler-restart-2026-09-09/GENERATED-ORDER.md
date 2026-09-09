# Controlled graph-order evidence

The controlled three-process run supports a seed-sensitive compiler-cache reuse
mechanism under the same wrapper and helper module names:

| Process | Hash seed | First compiled preparation | AOT / FX graph cache |
| --- | ---: | ---: | --- |
| New local compiler directories | 0 | 109.25 s | 2 misses / 2 misses |
| Reused directories, same seed | 0 | 22.49 s | 2 hits / 2 hits |
| Reused directories, changed seed | 1 | 102.61 s | 2 misses / 2 misses |

The same-seed preparation request is 79.41% shorter. Every arm completed all 54
calls; repeated-arm outputs match their predecessor token for token. These first
compiled request times include generation and follow modes that warm the model
and GPU. They are not whole-service or fresh-device startup measurements.

## What the retained code shows

`inspect-generated-order.py` reads the downloaded generated Python as syntax trees.
It verifies every inspected file against the completed run's cache inventory. It
does not execute generated code or load pickle objects.

The same-seed process leaves the complete 992-file inventory unchanged. The
changed-seed process adds six files: two AOT artifacts, two FX graph artifacts and
two generated Python modules. Existing artifacts remain unchanged.

Both generated decoder partitions differ in the identities of two rotary-frequency
arguments. Seed 0 gives `arg8_1` length 256 and `arg9_1` length 128; seed 1 reverses
those shapes. Renaming only these two argument identifiers makes the corresponding
`partition_0` syntax trees exactly equal. This comparison covers those partition
bodies, not the entire generated modules or serialized cache keys.

The actual model receipts also show the attention-layer and rotary-buffer order
switching from full-then-sliding to sliding-then-full. The pinned model constructs
these collections as sets. A separate four-process CPU proof reproduces that order
change using the actual model constructor on the meta device, without weights.

Together, the controlled cache hit/miss pattern, model order and generated-code
comparison support rotary ordering as the explanation for this cache divergence.
Changing the Python hash seed can affect other hash-sensitive behavior, so this
experiment is not a surgical intervention on only that one set. It also does not
prove a general fix across models, versions or deployments. The local proof is
`results/generated-order-proof.json` (SHA `0473aa43…d62db`).
