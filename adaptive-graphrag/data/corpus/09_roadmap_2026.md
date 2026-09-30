# 2026 Engineering Roadmap

## Theme 1: Reliability

After INC-2041, the top reliability goal is to make Heron resilient to telemetry outages. The
Fleet Platform group will ship last-known-position fallback in Dispatch Scheduler and move
Telemetry Ingest to a multi-region active-active deployment by Q3 2026.

## Theme 2: Safer model releases

After INC-2107, the Perception group will introduce slice-based evaluation gates for every
VisionCore release, and the Device Management team will integrate those gates into the OTA Update
Service. The target is zero SEV-2 or higher incidents caused by model releases in the second half
of 2026.

## Theme 3: Generative AI in Heron

The Applied AI team, led by Daniel Brooks, will expand the Fleet Assistant from answering
questions about runbooks to taking actions through tools, such as pausing a robot or creating an
incident ticket. Tool access will be exposed through Model Context Protocol (MCP) servers with
per-tool permissions, and every action will require operator confirmation.

## Theme 4: Osprey scale-up

Osprey will expand from power-line inspection to wind-turbine blade inspection, with a dedicated
VisionCore variant for blade-surface defects, targeting general availability in Q4 2026.
