# Gateway models

The repository keeps every released model version auditable. The machine-readable
source of truth is [`models/registry.json`](../../models/registry.json).

## Model generations

| | V1 development demonstration | V2 real-PPG candidate |
|---|---|---|
| Status | Installed complete/default route | Installed separate candidate route |
| Training source | Generated normal windows seeded from the supplied CSV | Real Pulse Transit Time PPG recordings |
| People | 60 simulated identities | 22 real participants |
| Input | 24-hour sequence | 5-second vector about every 30 seconds |
| Continuous SpO2 | Generated | Not present in the public dataset |
| Purpose | Prove the complete gateway and Pi path | Ground feature learning and validation in real physiology |

Read the [V1 model card](development-demo-v1.md) and the
[V2 model card](real-ppg-v2.md) before quoting results. Their metrics answer different
questions and must not be presented as directly comparable clinical accuracy.

V2 has passed participant-held-out evaluation, int8 runtime loading, and
eight-minute aggregation on a development computer. The Pi loads both models,
but V1 remains the complete/default contract until the exact wearable feature
formulas/BLE mapping are confirmed and V2 latency and memory are measured on the
target Pi. The registry default changes only after those two gates pass.
