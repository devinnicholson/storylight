# NVIDIA NIM engine reuse: readiness fell from 631 to 229 seconds

Persisting NVIDIA NIM's TensorRT vision and language engines reduced container-start-to-ready time from 631 to 229 seconds after a same-node restart on Google Kubernetes Engine. All 12 baseline Nemotron verdicts were unchanged. The 402-second reduction, 63.7 percent, came from reusing built engines while keeping the model image and review contract fixed.

| Measurement | Initial start | Same-node restart |
| --- | ---: | ---: |
| Container start to ready | 631 s | 229 s |
| Baseline verdicts preserved | 12/12 | 12/12 |
| GPU | NVIDIA L4 | NVIDIA L4 |
| Inference server | NVIDIA NIM 1.3.1 | NVIDIA NIM 1.3.1 |
| Model | Llama 3.1 Nemotron Nano VL 8B | Same |

## Why startup was expensive

NIM packages the model server and its CUDA/TensorRT execution path, but the first start still has to prepare model-specific engines for the available GPU. Nemotron Nano VL has language and vision components, so engine construction touches both paths before `/v1/health/ready` can succeed. On this GKE L4 pod, the initial preparation took more than ten minutes.

Storylight mounts an 80 GiB `ReadWriteOnce` persistent volume at `/opt/nim/.cache` and sets `NIM_CACHE_PATH` to that mount. A recreated pod on the same node can reopen the prepared TensorRT artifacts instead of rebuilding them from model weights. The Kubernetes deployment uses `Recreate`, requests one L4, allocates 32 GiB of host memory with a 41 GiB limit, and provides an 8 GiB memory-backed `/dev/shm`. NIM runs with a 2,048-token maximum, batch size one, and low-memory mode because the critic handles one bounded illustration at a time.

The initial BF16 engine build had previously peaked at 42.50 GB of host memory, which ruled out the smaller `g2-standard-8` allocation once Kubernetes overhead was included. The manifest requests enough memory to place the workload on `g2-standard-12` while retaining exactly one L4. GPU serving capacity here depends on host memory and accelerator memory, while persistent engine storage and health-probe timing determine whether the pod becomes usable.

## The review contract

The NIM endpoint uses an OpenAI-compatible chat-completions API. Storylight sends a generated image and a privacy-minimized visual brief, with no reader recording or original passage. The image is capped at 8 MiB and serialized contract text at 1,400 UTF-8 bytes. Generation stops at 128 output tokens.

Nemotron returns a compact JSON object with fidelity, composition, projector legibility, identity consistency, unintended-text detection, a decision, and a short reason. NIM 1.3.1 falls back from xgrammar to a slower backend for several rich JSON Schema constructs, so the wire schema uses short keys and basic types. Pydantic enforces score ranges, decision values, nullable corrections, and cross-field rules after generation. An accepted verdict cannot claim identity failure or unintended text, and a refine or reject verdict must include a correction.

The 12-call baseline repeated six visual contracts twice against one inspected synthetic fox image. The cases varied count, physical action, forbidden subjects, and left/right placement. Preserving all 12 decisions across restart checked that engine reuse changed readiness without changing the observed critic behavior on that screen.

## What we learned and what is next

Engine persistence made an on-demand NIM service much more practical. Readiness remained 229 seconds, so Storylight keeps visual review off the first-image path and uses it for prepared Story Packs, where work can start before the reader reaches a page. A watchdog limits GPU runtime and can scale the deployment back to zero.

The next experiment should compare a same-node restart with a clean-node start and record the reused engine files. Timing should separate image pull, model download, engine deserialization, and health probes. The six-contract screen also needs broader images and independently authored failure cases before critic accuracy can support automated rejection. Original operational receipts remain private, so the public result documents the measured summary and the implementation rather than a complete replay package.

Implementation: [`nemotron_critic.py`](../../src/storylight/nemotron_critic.py), [`paired critic benchmark`](../../src/storylight/nemotron_critic_benchmark.py), and [`GKE deployment`](../../infra/gcp/k8s/anticipatory.yaml).
