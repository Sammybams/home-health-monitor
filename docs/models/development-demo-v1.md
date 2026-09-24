# Model V1: development demonstration

## Status

`development_default` — deployable for integration testing, not a field model.

## Training source

V1 is the existing 18 KiB integer autoencoder. The executed
[training notebook](../../notebooks/train-and-evaluate-autoencoder.ipynb) used
11 distinct normal rows from the supplied monitoring CSV to seed 240 generated
24-hour windows for 60 simulated identities.

The model receives 288 five-minute steps. Each step contains normalized heart
rate, SpO2, temperature, motion, four quality values, and eight presence masks.

## Reported evaluation

The held-out evaluation compared generated normal windows with controlled
feature shifts. It reported 94.44% balanced accuracy, 100% controlled-anomaly
recall, and 88.89% specificity. These values demonstrate that the training,
threshold, integer export, and gateway inference path work together. They do
not measure performance on real people.

Exact artifacts and reports remain in
[`models/development-demo`](../../models/development-demo), with figures in
[`docs/assets/development-demo`](../assets/development-demo).

## Replacement rule

Do not delete or overwrite V1. V2 receives a separate model ID and directory.
V1 stops being the default only after V2 passes the release gates documented in
the [V2 scope](real-ppg-v2.md).
