# Gateway models

The repository keeps every released model version auditable. The machine-readable
source of truth is [`models/registry.json`](../../models/registry.json).

## Model generations

| | V1 development demonstration | V2 real-PPG candidate |
|---|---|---|
| Status | Current development default | Implementation in progress |
| Training source | Generated normal windows seeded from the supplied CSV | Real Pulse Transit Time PPG recordings |
| People | 60 simulated identities | 22 real participants |
| Input | 24-hour sequence | Short wearable feature vector |
| Continuous SpO2 | Generated | Not present in the public dataset |
| Purpose | Prove the complete gateway and Pi path | Ground feature learning and validation in real physiology |

Read the [V1 model card](development-demo-v1.md) and the
[V2 scope](real-ppg-v2.md) before quoting results. Their metrics answer different
questions and must not be presented as directly comparable clinical accuracy.

V1 remains available for reproduction until V2 has passed participant-held-out,
quantized-runtime, eight-minute aggregation, and Raspberry Pi validation. The
registry default changes only after those gates pass.
