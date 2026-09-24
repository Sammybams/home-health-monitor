# Pulse Transit Time PPG data audit

This dataset is the real physiological source for the V2 development model. It
does not replace later validation with the actual wearable and target
population.

## Verified archive

```text
file: pulse-transit-time-ppg.zip
SHA-256: fbd8defee1be6af4496b9191cbc915a5bdf81b4766148b830aae6f4cfa7677f9
dataset version: 1.1.0
```

The complete machine-readable result is
[`data-audit.json`](../models/real-ppg-v2/data-audit.json). Reproduce it without
extracting the archive:

```sh
PYTHONPATH=src python3 -m home_health_monitor.datasets.audit \
  pulse-transit /secure-data/pulse-transit-time-ppg.zip
```

## Verified contents

| Item | Result |
|---|---:|
| Participants | 22 |
| Recordings | 66 |
| Activities | 22 sitting, 22 walking, 22 running |
| WFDB waveform samples | 16,217,132 |
| Structurally valid CSV rows | 16,156,786 |
| Total duration from WFDB headers | 9.009518 hours |
| Sex recorded in metadata | 7 female, 15 male |
| Age range | 20–55 years |
| Mean age | 28.41 years |
| Endpoint SpO2 observations | 132, ranging from 94% to 99% |

Continuous channels are ECG, six PPG waveforms, two attachment-pressure
channels, three temperature channels, acceleration, and gyroscope. Heart rate
can be derived from PPG and checked against ECG peaks. Temperature and motion
are continuous. SpO2 is only recorded at the beginning and end of an activity;
it is not a continuous waveform.

## Data defect policy

The archive contains one malformed final row in `s18_walk.csv`:

```text
line: 182283
expected columns: 20
actual columns: 9
```

That CSV also ends 60,345 samples before its WFDB header count. All other CSV
record lengths match their WFDB headers. The feature pipeline must report the
defect, skip the incomplete row, and use only complete five-second segments
from the valid prefix. A segment must never be padded, copied, or joined to a
different activity to hide the missing tail.

## Allowed use

Use this dataset to train and evaluate real-data development features for PPG
heart rate, PPG quality, temperature, and motion. Split by participant so that
no recording from a test participant appears in training.

Do not:

- interpolate endpoint SpO2 into a continuous signal;
- describe activity labels as health outcomes;
- claim the short laboratory sessions validate 48-hour calibration;
- claim the sensors or participants match the target deployment;
- commit the source archive or extracted participant data to Git.

## Reproducible feature result

The committed builder samples a complete five-second capture every 30 seconds,
matching the proposed wearable cycle. It produced 1,112 vectors with no feature
extraction failures. PPG heart rate was accepted for 69.6% of vectors; accepted
estimates had 1.81 BPM mean absolute error against ECG peaks.

Rebuild the ignored JSONL corpus and its portable report:

```sh
home-health-ppg-features \
  /secure-data/pulse-transit-time-ppg.zip \
  notebooks/.work/real-ppg-features.jsonl \
  --report models/real-ppg-v2/feature-extraction-report.json
```

The committed result is
[`feature-extraction-report.json`](../models/real-ppg-v2/feature-extraction-report.json).
