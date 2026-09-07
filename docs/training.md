# Gateway model training

Training happens on a development computer, never on the 512 MB Pi. The Pi only
loads the final integer model and performs inference.

## Data required

Use continuous normal recordings from the exact target wearable: the same
sensor models, placement, firmware, feature formulas, sampling cycle, and
quality calculations used in deployment. The first 48 hours for each person are
used to build their personal normalization profile. Later normal readings are
converted into UTC-aligned 24-hour windows.

The production training input is JSON Lines. Each line is one normalized
window:

```json
{
  "schema_version": 1,
  "subject_id": "private-person-id",
  "feature_manifest_id": "features-v1",
  "decision": "normal",
  "feature_names": [
    "heart_rate_bpm",
    "spo2_percent",
    "temperature_c",
    "motion_intensity",
    "quality_ppg",
    "quality_spo2",
    "quality_temperature",
    "quality_motion"
  ],
  "values": [[0.0, 0.0, 0.0, 0.0, 0.96, 0.94, 1.0, 0.99]],
  "masks": [[1, 1, 1, 1, 1, 1, 1, 1]]
}
```

The example shows one time step for readability. Both `values` and `masks` must
contain exactly 288 rows and eight columns. Masks contain only `0` or `1`.
Values must be finite. Every row must say `normal`; the loader rejects anomaly
rows because this is normal-only unsupervised learning.

Generated windows and participant data belong under an ignored external data
directory, not in Git.

## Prepare a training environment

Use 64-bit Python 3.9-3.12. Current TensorFlow release wheels do not support the
Python 3.14 interpreter used by some development machines.

```sh
python3.12 -m venv .venv-train
. .venv-train/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[gateway-train]'
```

The implementation follows TensorFlow's full-integer post-training quantization
flow: it supplies representative target data and restricts conversion to int8
built-in operations. See the official [integer quantization guide](https://www.tensorflow.org/lite/performance/post_training_integer_quant).

## Validate the pipeline first

When TensorFlow is installed, run the one-epoch synthetic smoke test:

```sh
home-health-gateway-train --smoke-test artifacts/gateway-smoke
```

This exercises model creation, training, int8 conversion, metadata generation,
checksum generation, and artifact reload. The smoke data proves only that the
pipeline runs.

## Train production candidates

```sh
home-health-gateway-train \
  /secure-data/home-monitor/normal-windows.jsonl \
  artifacts/gateway \
  --epochs 30 \
  --batch-size 16 \
  --seed 42
```

The split is by person:

- approximately 70% of people for model fitting;
- approximately 15% for threshold selection;
- approximately 15% untouched for final normal-data evaluation.

No subject appears in more than one split. Early stopping restores the weights
with the best validation loss.

## Exported files

```text
artifacts/gateway/model.tflite
artifacts/gateway/model-metadata.json
```

Metadata contains:

- model and feature-manifest identities;
- training time and model SHA-256;
- exact input/output shape and int8 quantization parameters;
- persistent and severe reconstruction thresholds;
- train/validation/test participant IDs and window counts;
- held-out reconstruction and false-anomaly summaries.

The command fails if the model is 1 MiB or larger or if the exported model
cannot be loaded back through the runtime contract.

## Evaluation and release checklist

Before copying a candidate to a gateway, record:

1. normal false-anomaly fraction on untouched people;
2. false alerts per person per day;
3. errors by rest, walking, running, and other expected activities;
4. behavior with real missing packets and poor sensor contact;
5. error distributions for relevant age, sex, skin-tone, and health subgroups;
6. sensitivity to simulated changes, labelled only as engineering simulation;
7. model size;
8. peak RSS and inference latency on the 512 MB Pi Zero 2 W.

The model can learn normal without disease outcomes. Simulated anomalies can
test whether the pipeline responds, but they are not training labels and do not
establish field performance.

## Install a selected artifact

Copy both files together. The gateway rejects mismatched or modified files:

```sh
scp artifacts/gateway/model.tflite pi@PI_ADDRESS:/tmp/
scp artifacts/gateway/model-metadata.json pi@PI_ADDRESS:/tmp/
```

The exact installation and service restart commands are in the
[Pi deployment guide](pi-deployment.md).
