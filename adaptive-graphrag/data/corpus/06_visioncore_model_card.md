# Model Card: VisionCore

## Overview

VisionCore is the perception model runtime shipped on every Kestrel robot and Osprey drone. It is
owned by the Perception Models team in the Perception group led by Priya Raman.

## Architecture

The current production release, VisionCore 3.1, is a YOLO-family single-stage detector with a
segmentation head, exported to TensorRT and running on an NVIDIA Jetson Orin module on each device.
Inference runs at 45 frames per second on Kestrel. Osprey uses a variant fine-tuned for power-line
components such as insulators, dampers and conductors.

## Training data

VisionCore 3.1 was trained on 1.8 million labelled images collected from customer sites, augmented
with synthetic images generated in NVIDIA Isaac Sim.

## Evaluation

| Version | Overall mAP@0.5 | Low-light pallet recall |
|---------|-----------------|-------------------------|
| 3.1     | 0.87            | 0.91                    |
| 3.2     | 0.89            | 0.62                    |

Version 3.2 was withdrawn after incident INC-2107 because of its low-light pallet recall.

## Intended use and limitations

VisionCore is intended for obstacle, pallet and person detection in structured industrial
environments. It is not validated for outdoor use on Kestrel. Person detection is used only for
safety stops, never for identifying individuals.
