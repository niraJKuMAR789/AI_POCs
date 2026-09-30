# Postmortem INC-2107: VisionCore Detection Regression

**Severity:** SEV-2  **Date:** 2026-04-03  **Duration:** 6 hours
**Incident commander:** Luis Fernandez

## Summary

After VisionCore model version 3.2 was rolled out, Kestrel robots began misclassifying
shrink-wrapped pallets as obstacles. Robots stopped and requested remote assistance roughly
4 times more often than usual.

## Root cause

Version 3.2 was trained on a refreshed dataset in which shrink-wrapped pallets under sodium-vapour
warehouse lighting were under-represented. Offline evaluation reported a higher overall mAP than
version 3.1, but the evaluation suite had no slice for low-colour-temperature lighting.

## Mitigation

The OTA Update Service halted the rollout at the 10 percent canary wave, and the fleet was rolled
back to VisionCore 3.1. Because the rollback used the canary mechanism, only about 10 percent of
robots were affected.

## Action items

1. Add lighting-condition slices to the VisionCore evaluation suite (owner: Luis Fernandez).
2. Gate OTA promotion on a per-slice regression check rather than overall mAP (owner: Grace Kim).
3. Collect 20,000 additional labelled images from Northgate Logistics sites (owner: Priya Raman).
