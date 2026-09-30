# Postmortem INC-2041: Dispatch Delays in us-central

**Severity:** SEV-1  **Date:** 2026-02-11  **Duration:** 97 minutes
**Incident commander:** Amara Okafor

## Summary

For 97 minutes, Kestrel robots in the us-central region received missions with delays of up to
14 minutes. Northgate Logistics reported that 6 of its warehouses stopped picking during the
peak morning shift.

## Impact

Dispatch Scheduler p95 assignment latency rose from 1.2 seconds to over 800 seconds. About 38,000
missions were delayed. The Heron API itself stayed available, so dashboards looked healthy at first.

## Root cause

A schema change to the battery-telemetry Protobuf message was deployed to **Telemetry Ingest**
without a compatible reader. Telemetry Ingest began rejecting 70 percent of messages, so Kafka
topics consumed by Dispatch Scheduler stopped receiving fresh robot positions. Dispatch Scheduler
waited for fresh positions before assigning missions, which caused the delays.

## Detection gap

The alert on Telemetry Ingest rejection rate was configured as Tier 2, so it did not page. The
incident was detected 31 minutes late through a customer escalation.

## Action items

1. Enforce backward-compatible schema checks in CI for all Protobuf changes (owner: Rahul Iyer).
2. Reclassify the telemetry rejection-rate alert as Tier 1 (owner: Amara Okafor).
3. Allow Dispatch Scheduler to fall back to last-known positions for up to 5 minutes
   (owner: Sofia Petrova).
