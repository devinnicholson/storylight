# GCP qualification GPU release check

The cache experiment exposed a gap between service deletion and resource release. M's service was deleted at 22:32:25 UTC, but unaligned Monitoring samples reported one active instance through 22:38. N's first request received an explicit RTX quota rejection. M's active and idle counts were both zero at 22:39, 22:40 and 22:41. These observations do not establish exact billable duration or prove the cause of the quota rejection.

The new read-only guard checks the previous experiment service before another GPU service is started. It requires both active and idle counts for the expected trial revision and every other observed revision, two complete zero samples at least 60 seconds apart after deletion, a latest sample no older than 180 seconds, and at least ten minutes since the deletion audit timestamp. Missing, stale, partial, paginated or nonzero evidence remains unproven. It polls once per minute for at most 900 seconds and retains each exact query, successful raw response, and verdict. Credentials stay in memory.

Monitoring samples this metric every 60 seconds and can publish it up to 120 seconds later. The ten-minute floor is a conservative experiment policy based on our observed delay, not a documented release guarantee. The guard neither reserves quota nor establishes availability for other services. [Metric definition](https://docs.cloud.google.com/monitoring/api/metrics_gcp_p_z), [Cloud Run runtime contract](https://docs.cloud.google.com/run/docs/container-contract).

Run it after verified service deletion, using the actual deletion audit timestamp:

```sh
PATH=/opt/homebrew/bin:$PATH .venv/bin/python scripts/wait_gcp_klein_gpu_release.py \
  --service bookforge-klein-qualification-20260906-m \
  --after 2026-09-06T22:32:26.192549Z \
  --output /private/tmp/klein-m-release-check
```

Run the guard promptly after deletion: old series can stop publishing, so missing recent data cannot be treated as zero. A retained response may be audited at its recorded acquisition time; that historical observation is distinct from a new capacity check. The retained M response qualifies at its recorded 22:42:33 UTC capture, when its latest sample was 93 seconds old and deletion was over ten minutes earlier.

The trial supervisor also now rejects an incomplete workload even when the evidence-collecting benchmark client exits with code zero. It requires ten successful cases from one worker before treating the run as completed or exporting a cache. Failure still reaches service deletion and the closure receipt. Earlier N evidence remains unchanged: its client exited normally after recording a rejected workload.

Future experiment allowances must account for possible resource use after control-plane deletion. No further GPU work is authorized by this document or by unused allowance from a failed phase.
