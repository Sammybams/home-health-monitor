# Gateway model training

Training happens on a development computer, never on the 512 MB Pi. The Pi only
loads the final integer model and performs inference.

There are two reproducible pipelines. V1 trains the installed 24-hour
development demonstration described below. V2 trains the real-PPG short-vector
candidate with the separate notebook and command in the next section.

## Train the V2 real-PPG candidate

Download the audited Pulse Transit Time PPG 1.1.0 ZIP outside Git, prepare the
training environment, and run the authoritative notebook:

```sh
python3.12 -m venv .venv-train
. .venv-train/bin/activate
python -m pip install -e '.[gateway-train,analysis,notebook]'
export PULSE_TRANSIT_PPG_ZIP=/secure-data/pulse-transit-time-ppg.zip
MPLBACKEND=Agg python -m jupyter nbconvert \
  --execute --to notebook --inplace \
  --ExecutePreprocessor.timeout=900 \
  notebooks/train-real-ppg-vector-autoencoder.ipynb
```

That notebook performs the raw archive audit, extracts five-second captures at
the proposed 30-second wearable cadence, runs five participant-grouped folds,
quantizes every fold before threshold scoring, evaluates the locked people,
exports the final int8 model, and renders six plots. Equivalent individual
commands are `home-health-ppg-features`, `home-health-ppg-train`, and
`home-health-ppg-report`.

V2's first 48 hours are calibration history, not a model tensor. The runtime
starts with its grouped public-data threshold so it can always return a binary
result, then changes to the person's mean reconstruction error plus three
standard deviations after both 48 elapsed hours and at least 288 valid
eight-minute calibration intervals (80% coverage).

The V2 artifact remains a development candidate until the final 20-field
wearable formulas and BLE payload match its manifest, normal target-wearable
data including continuous SpO2 is collected, and the target Pi is benchmarked.

## V1 24-hour pipeline

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

## Runnable development model available now

The repository includes a small model explicitly tagged `development_demo`,
plus its full training report and plots. It was trained on generated normal
patterns seeded by the distinct normal rows in the supplied CSV. GalaxyPPG and
BIDMC were compared separately because neither has the complete aligned feature
set. See [the development model report](development-model.md) for exact data,
commands, metrics, and interpretation.

The authoritative development run is
[`notebooks/train-and-evaluate-autoencoder.ipynb`](../notebooks/train-and-evaluate-autoencoder.ipynb).
It performs the training itself and invokes this module's reusable Python
implementation; it does not merely display previously calculated results.

## Install a selected artifact

Copy both files together. The gateway rejects mismatched or modified files:

```sh
scp artifacts/gateway/model.tflite pi@PI_ADDRESS:/tmp/
scp artifacts/gateway/model-metadata.json pi@PI_ADDRESS:/tmp/
```

The exact installation and service restart commands are in the
[Pi deployment guide](pi-deployment.md).
