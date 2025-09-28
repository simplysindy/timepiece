# Monitoring Setup

These files define Cloud Monitoring alert policies for the production watch forecasting deployment. Update the
placeholder `PROJECT_ID` and `CHANNEL_ID_PLACEHOLDER` values before applying them.

## Prerequisites
- `gcloud` CLI authenticated against the target project (`timepiece-473511` in production)
- A notification channel (email, Slack, PagerDuty, etc.) already registered with Cloud Monitoring

## Deployment Commands
```bash
# Create (or update) the latency alert for the Cloud Function
sed "s/PROJECT_ID/timepiece-473511/g; s/CHANNEL_ID_PLACEHOLDER/CHANNEL_ID/g" \
  infra/monitoring/predict_latency_policy.yaml \
  | gcloud alpha monitoring policies create --policy-from-file=-

# Create the Cloud Run 5xx alert
sed "s/PROJECT_ID/timepiece-473511/g; s/CHANNEL_ID_PLACEHOLDER/CHANNEL_ID/g" \
  infra/monitoring/cloud_run_errors_policy.yaml \
  | gcloud alpha monitoring policies create --policy-from-file=-
```

Use `gcloud alpha monitoring policies list` to confirm both alerts were created. When updating an existing policy,
include `--replace` instead of `create`.
