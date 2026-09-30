# Heron Platform Architecture

Heron is an event-driven platform deployed on Kubernetes in two regions (us-central and
eu-west). It is owned by the Fleet Platform group.

## Request path

Robots and customer applications call the **Heron API**, which authenticates requests with
short-lived mTLS certificates and routes them to internal services over gRPC.

## Services and dependencies

- **Telemetry Ingest** consumes device telemetry over MQTT, validates it against Protobuf schemas,
  and publishes it to Apache Kafka. A downstream consumer writes it to TimescaleDB.
- **Dispatch Scheduler** reads robot positions and battery levels from Kafka topics produced by
  Telemetry Ingest. It uses a constraint solver built on Google OR-Tools to assign missions.
  Because it depends on fresh telemetry, a Telemetry Ingest outage directly delays dispatching.
- **OTA Update Service** stores signed firmware and model artifacts in object storage and rolls
  them out in canary waves of 1 percent, 10 percent and 100 percent of the fleet.
- **Fleet Assistant** answers operator questions using retrieval-augmented generation over Heron
  runbooks and live fleet metrics. It uses NVIDIA NIM microservices for the LLM and for NeMo
  Retriever embeddings, with Qdrant as the vector database.

## Data stores

PostgreSQL stores customers, sites and mission history. TimescaleDB stores time-series telemetry.
Apache Kafka is the event backbone, with a retention of 72 hours.

## Service-level objectives

The Heron API targets 99.95 percent monthly availability. Dispatch Scheduler targets a p95 mission
assignment latency under 2 seconds.
