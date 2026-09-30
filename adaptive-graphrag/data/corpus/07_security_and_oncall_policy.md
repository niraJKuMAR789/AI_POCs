# Security, Data and On-Call Policy

## Access control

All production access at Helix Dynamics uses single sign-on with hardware security keys.
Engineers receive production access through just-in-time grants that expire after 4 hours.
Access to customer telemetry requires approval from the owning group lead.

## Data retention

Raw device telemetry is retained for 30 days in TimescaleDB and then downsampled to hourly
aggregates retained for 2 years. Camera images are not uploaded by default; customers must opt in
per site before images can be collected for model training.

## LLM usage

Generative-AI features must run on approved model endpoints. The approved endpoints are NVIDIA NIM
microservices, either hosted on the NVIDIA API catalog or self-hosted in the Helix Dynamics
Kubernetes clusters. Customer data must never be sent to endpoints that retain prompts for training.
Every LLM feature must log prompts and responses for 30 days for audit purposes.

## On-call

Engineers on a Tier 1 rotation must acknowledge pages within 10 minutes. On-call shifts last one
week and hand off every Monday at 10:00 local time. Each rotation must have at least 6 engineers.
After a SEV-1 incident, a blameless postmortem must be published within 5 business days.
