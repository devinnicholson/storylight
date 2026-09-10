# JAX and MaxText training: cached jobs fell from 738 to 331 seconds

Persistent JAX compilation caching reduced the measured Gemma 4 E2B LoRA training job from 738 to 331 seconds on two NVIDIA L4 GPUs. Provider cost fell from about $0.64 to $0.32. Verification found 205 paired LoRA modules in the native MaxText state and confirmed that nonzero gradients changed the adapter weights.

| Measurement | First job | Cached job |
| --- | ---: | ---: |
| Job duration | 738 s | 331 s |
| Approximate provider cost | $0.64 | $0.32 |
| Hardware | 2 x NVIDIA L4 | 2 x NVIDIA L4 |
| LoRA coverage | 205 paired modules | 205 paired modules |

These Modal runs used Google's JAX and MaxText software on NVIDIA hardware. They were separate from the later NF4 QLoRA work on Google Cloud, and the 331-second figure measures the training job rather than scene-generation latency.

## The conversion and training path

The pipeline begins with a pinned Hugging Face Gemma checkpoint and converts it into MaxText's native parameter tree. It inserts LoRA state before training on Storylight's structured scene examples. A second conversion merges the trained adapter back into an inference checkpoint, with output checks around both boundaries to catch parameter-tree mistakes that shape validation alone can miss.

MaxText exposes the training state as JAX arrays distributed across the two L4 devices. LoRA adds a low-rank update to selected dense projections, represented by paired matrices whose product supplies the weight delta. The verification counted 205 such pairs in the native state, checked that each selected path had both halves, observed nonzero gradients, and confirmed that optimizer steps changed the adapter values. Those checks distinguish a completed training loop from a job that only loaded data and compiled.

JAX compilation is a large part of a short specialization run. XLA traces the training step, lowers the graph for the device mesh, compiles GPU executables, and specializes them to array shapes and sharding. Repeating that work can consume more wall time than the small LoRA dataset needs for useful updates. Persisting the compilation cache allowed the second job to reuse compiled work instead of paying the full trace and build cost again.

The two-L4 configuration also tests a boundary that matters for open-model training: state layout must survive conversion and sharding before any quality score is meaningful. The checkpoint moves from Hugging Face to MaxText, then returns as a merged inference model; parameter names change at both boundaries. The pipeline verifies lineage and output behavior across those representations instead of assuming a successful file write preserved the model.

## Reusable training path

The cached run cut iteration time by 407 seconds, or 55.1 percent, and roughly halved the provider charge. That changes the practical research loop. A developer can adjust data or adapter targeting and receive another measured result in about five and a half minutes on rented L4s, while keeping the base model and training stack open for inspection.

The adapter from this campaign did not replace the live planner. Its contribution was the verified training path: checkpoint conversion, native LoRA updates, merged export, and reusable compilation on modest cloud GPUs.

## What we learned and what is next

Compilation state belongs in the experiment protocol. Cache directories, JAX and MaxText revisions, device topology, shapes, sharding, and source hashes need to be pinned together; a warm cache from a different graph can add storage without saving time. Future runs should publish the cache key material and break job duration into environment startup, conversion, compilation, training, checkpointing, and verification.

The public repository retains lineage validation, though the original large adapter artifacts and operational receipts remain archived. A full reproduction package would add the exact MaxText revision, container digest, cache manifest, and merged checkpoint hashes, then repeat the cold and cached jobs under the same billing window.

Related implementation: [`validate_fidelity_release.py`](../../scripts/validate_fidelity_release.py). Project context: [`README.md`](../../README.md#jax-and-maxtext-training-the-scene-planner).
