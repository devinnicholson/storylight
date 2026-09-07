# Packed transformer image qualification

The [CPU producer](gcp-klein-transformer-pack-2026-09-07.md) passed all 169
transformer tensor comparisons. This next phase builds one private common image
from the original immutable G image plus the pack, its proof, audited wheel and
license. It makes no GPU requests and does not promote the candidate.

The seven-file, 31,250-byte source context pins the three private GCS object
generations and hashes. The download happens inside GCP and checks all content
hashes. Pinned BuildKit produces one new layer. A separate streaming verifier
checks its compressed SHA-256, full uncompressed diffID and four file hashes,
plus all thirteen inherited layer descriptors, diffIDs and original runtime
configuration. It never downloads inherited model layers. The large pack does
not return to the workstation.

The single CPU8 build is bounded to 1,800 seconds, with bounded preparation and
verification children and no automatic retry. Failure after image publication
still rejects the image; registry presence alone is not acceptance. Small
verification receipts are the only build outputs downloaded for local review.

The final separate allowance is $1.10: $0.468 build, $0.25 registry storage,
$0.16 for reading the US multi-region source pack, $0.05 for small evidence and
operations, and $0.172 margin. The proposed $0.85 amount was corrected before
execution when the source bucket location was checked. New layer storage is
bounded to nine GiB and seven days, after which deletion or a separately funded
reuse plan is required. Prior unreconciled reservations remain $27.3019592.
These are allowances and estimates, not billing hard caps or invoices.

Same-region Artifact Registry transfer is free, while US multi-region Cloud
Storage reads into the regional builder have a separate charge. Image-scanning
APIs were checked and are disabled. Sources:
[Artifact Registry pricing](https://cloud.google.com/artifact-registry/pricing),
[Cloud Storage pricing](https://cloud.google.com/storage/pricing), and
[Cloud Build pricing](https://cloud.google.com/build/pricing).

Build `2d6086d7-66da-4686-9caa-e311d495217f` published image
`sha256:42acfcdbb13dde3af9864afe7c5b1a0b8c1740b6eb39adea7abae4dbc4321eda`,
then failed verification during registry metadata reads. Its 655.735-second
duration gives a $0.170491 compute estimate, not an invoice. The $0.928
work/storage hold remains unreconciled, bringing all holds to $28.2299592.

Local small-object probes reproduced HTTP 302 relative redirects for both
configuration blobs. A separately reviewed verifier permits one GET redirect
within the exact HTTPS registry origin; metadata-server redirects remain
forbidden. With that repair, all thirteen inherited layer descriptors, diffIDs
and runtime configuration match. The new compressed layer is 6,087,237,044 bytes.
At that point its full contents remained unverified, so the image was not
GPU-admissible.

The verification-only repair has a separate $0.35 allowance: $0.26 for a
1,000-second CPU8 build, $0.02 for small evidence/operations, and $0.07 margin.
It uses the pinned Python base image and a 900-second verifier deadline, streams
the existing layer inside GCP, and does not rebuild the image. The original
failed build and verifier remain retained unchanged.

The repair succeeded in build `92e8f9c7-074f-4b6c-92b6-48439963f413`.
The full stream took 136.092 seconds; the complete build took 162.853 seconds
($0.042342 estimated CPU compute). All four file contents, compressed layer SHA,
uncompressed diffID and inherited image identity passed. Five generation-pinned
small receipts totaling 25,840 bytes passed independent local provenance review.
The large payload stayed in GCP. The common image now clears for reviewed
overlays, with no GPU correctness or speed claim. The separate $0.28 work/evidence
hold remains unreconciled, bringing total holds to $28.5099592.

A later GPU comparison must
measure complete client and factory times, then verify actual CUDA tensor bytes
after the timed requests. Common large layers control image-size differences
between variants; they do not prove an advantage over the original smaller image.
