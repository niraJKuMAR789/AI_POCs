# Kestrel Product Sheet

Kestrel is an autonomous mobile robot for warehouse picking and pallet transport, built by Helix
Dynamics.

## Specifications

- Payload: up to 1,500 kg
- Top speed: 2.0 m/s
- Battery: 8 hours of continuous operation, 45-minute fast charge
- Sensors: two 3D lidars, six stereo cameras, and an IMU
- Compute: NVIDIA Jetson Orin running VisionCore for perception and PathPlanner for navigation

## Software

Kestrel receives missions from the Dispatch Scheduler in the Heron platform and streams telemetry
to Telemetry Ingest every 200 milliseconds. Firmware and perception models are updated by the OTA
Update Service.

## Safety

Kestrel stops within 0.5 m when VisionCore detects a person in its path. It is certified to
ISO 3691-4 for driverless industrial trucks.

## Customers

The largest Kestrel deployment is at Northgate Logistics, with 1,200 robots across 14 warehouses.
