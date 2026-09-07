# GCP transformer loading experiment

The candidate replaces only Klein's transformer loading path with an explicitly
loaded FlashPack artifact. The restored watercolor prompts, BF16 weights,
1024×576 output, four steps, Qwen, VAE and depth paths remain unchanged. This is
experimental code; production has not been promoted.

The preceding tiny CPU experiment demonstrated that FlashPack can preserve the
ordinary tensor cases but does not safely preserve every model structure. Scalar
shape changes, tied parameters and overwritten destination declarations rule out
a global loader switch. The pinned Klein transformer contains 169 nonscalar BF16
tensors; the producer checks their complete bytes and actual model structure.
See [CPU evidence](gcp-klein-cpu-startup-2026-09-07.md).

## Producer

The source lives in `experiments/renderer-transformer-flashpack/producer/`.
It runs in the immutable original G image, checks the original checkpoint's full
SHA-256, packs on CPU, compares every packed tensor with the source, then assigns
the pack to a config-created meta model on CPU and checks every tensor again.
It requires exact names, shapes, BF16 dtype, contiguous strides, no aliases or
buffers, no remaining meta tensors and evaluation mode.

The eight-CPU builder has 8 GB RAM. Pinned safetensors and FlashPack source supports
file-backed views, but does not guarantee that the process fits. The producer
releases both the source dictionary and final loop tensor before assignment,
records peak process RSS and available cgroup counters, and fails after 900 seconds.
The outer build has a 1,800-second deadline and no automatic retry.

The reviewed context has five files, 79,605 bytes. Both the local archive and
actual uploaded archive were checked, including the writable evidence directory
needed by the image's non-root user. Evidence is under
`benchmarks/gcp-klein-transformer-pack-2026-09-07/`.

The single build, `e69c41fc-bf6f-4aed-891a-3f4b84326942`, succeeded. All 169
tensors passed source-to-pack and CPU-model equality checks in 151.732 seconds.
Peak process RSS was 7,049,424,896 bytes (6.57 GiB); the available cgroup reading
did not establish total VM usage. The complete build, including image pull and
private artifact upload, took 422.145 seconds.

The packed file is 7,751,107,978 bytes, SHA-256
`2d4325e06bcae9ba04c048b7f318e2fcd50cb739f91d7066a5fc462fd8905983`.
Generation-pinned proof and memory files passed local checks against the build
manifest's MD5 values. Object metadata matches all four exported artifacts;
the pack's complete content was verified by the producer inside GCP, not downloaded
or independently rehashed on the workstation. The retained original header
matches all proof tensor names, shapes and BF16 declarations.

## Cost and scope

The separate $0.75 allowance reserves $0.468 for at most 30 minutes of CPU8 build
time, $0.10 for storage and operations, and $0.182 margin. Prior unreconciled holds
remain $26.7339592. These reservations are neither invoices nor platform hard caps.
The billing guard reported $22.39 gross project cost at
2026-09-07T01:43:14.472450Z; reporting is delayed and project-wide. The completed
build's duration-based compute estimate is $0.109758, not an invoice. Its $0.568
work/storage reservation remains held, bringing total unreconciled holds to
$27.3019592.

A subsequent location check found that the source bucket is US multi-region.
The initial line-item estimate omitted about $0.144376 for replication of the
uploaded pack. Seven days of live pack storage adds about $0.043193. Together
with observed build compute, these remain within the retained $0.568 hold.
`cost-correction.json` records this correction; soft-deletion retention can
extend storage charges, and no billing closure is claimed. Rates come from
[Cloud Storage pricing](https://cloud.google.com/storage/pricing).

The private pack is limited to 7,752,000,000 bytes and seven days' retention,
after which it must be deleted unless a separately funded reuse plan assumes
its storage. The full model artifact stays in GCP. Only small proofs, logs and
metadata return to the workstation. This allowance includes no GPU request or
new OCI image build.

## Next measurement

CPU equality is a prerequisite, not a speed or image-quality result. A subsequent
reviewed build must verify the packed OCI layer inside GCP and pin the actual
proof into the candidate worker. A common packed base can control large image
layers between baseline and candidate. Such a comparison still cannot establish
startup improvement over the original smaller image by itself.

Whole-factory and client elapsed time must include the candidate's imports,
metadata checks, meta construction and assignment. Image fidelity and timing
must be reported separately, including baseline nondeterminism. No measured
startup improvement is claimed at this point.

Local verification: 960 Python tests, four JavaScript suites, four focused
transformer checks and scoped Ruff pass. Independent review cleared the producer,
submitted context and runtime proof interface before the paid build.
