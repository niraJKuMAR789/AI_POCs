# Service Ownership Registry

This registry is the source of truth for which team owns each production service, its on-call
rotation, and its tier. Tier 1 services page the on-call engineer immediately; Tier 2 services
page during business hours only.

## Fleet Platform group (lead: Marcus Chen)

- **Heron API** (Tier 1): the public REST and gRPC gateway used by customers and robots.
  Owned by the Fleet Core team. Primary on-call lead: Sofia Petrova.
- **Dispatch Scheduler** (Tier 1): assigns picking and transport missions to Kestrel robots.
  Owned by the Fleet Core team. Primary on-call lead: Sofia Petrova.
- **Telemetry Ingest** (Tier 1): receives robot and drone telemetry streams and writes them to
  Apache Kafka and the TimescaleDB time-series store. Owned by the Data Platform team.
  Primary on-call lead: Rahul Iyer.
- **OTA Update Service** (Tier 2): ships firmware and model updates to devices. Owned by the
  Device Management team. Primary on-call lead: Grace Kim.

## Perception group (lead: Priya Raman)

- **VisionCore** (Tier 1): the on-robot object-detection and segmentation model runtime used by
  Kestrel and Osprey. Owned by the Perception Models team. Primary on-call lead: Luis Fernandez.
- **Fleet Assistant** (Tier 2): the natural-language assistant in the Heron console, built on a
  retrieval-augmented LLM. Owned by the Applied AI team. Primary on-call lead: Daniel Brooks.

## Autonomy group (lead: Kenji Watanabe)

- **PathPlanner** (Tier 1): global and local motion planning for Kestrel. Owned by the Navigation
  team. Primary on-call lead: Hannah Muller.

## Escalation

If the primary on-call lead does not acknowledge a Tier 1 page within 10 minutes, the page
escalates to the owning group lead, and after 20 minutes to the SRE incident commander rotation
run by Amara Okafor.
