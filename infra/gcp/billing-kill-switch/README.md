# Storylight emergency billing disconnect

This second-generation Cloud Run function listens only to the project-scoped $175 gross-cost
emergency budget. At or above $175, it unlinks `your-gcp-project` from its billing account. It deliberately
checks the billing account, budget ID, display name, currency, and amount before acting.

A separate project-scoped $150 gross-spend budget provides thresholds at approximately $10, $50,
$100, $135, and $150. Both budgets exclude credits when calculating gross spend. These values and
the project-to-billing-account link were verified read-only on 2026-09-03; recheck them before a
billable experiment because console configuration can change independently of this repository.

The runtime identity has only `roles/billing.projectManager` and `roles/logging.logWriter` on this
project, plus the Eventarc receiver role and service-specific Cloud Run invoker needed for private
delivery. It has no organization-wide or billing-account administrator role.

Function build images are kept to two recent versions and deleted after 30 days. Google-managed
source and upload buckets also have bounded lifecycle rules, preventing deployment storage from
growing indefinitely.

The notification path is asynchronous: Google Cloud budget messages can arrive hours after costs
are recorded. This is a strong emergency brake, not a guaranteed real-time hard cap. Service-level
spend caps and keeping expensive services disabled remain the first line of defense.

Never publish an at-threshold test message to the production topic. Unit tests exercise that path
with a fake event and no Cloud Billing client call:

```bash
npm test --prefix infra/gcp/billing-kill-switch
```
