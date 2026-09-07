# Dataset roles

No reviewed public dataset contains the exact target wearable's continuous
motion, SpO2, temperature, derived heart rate, full-day coverage, and deployment
population. The datasets therefore have separate, explicit roles.

## GalaxyPPG: engineering validation only

[GalaxyPPG](https://zenodo.org/records/14635823) contains 24 participants with
Galaxy Watch 5, Empatica E4, and Polar H10 recordings. The published
[dataset paper](https://www.nature.com/articles/s41597-025-05152-z) and
[supplementary code](https://github.com/Kaist-ICLab/GalaxyPPG-Supplementary-Code)
describe Galaxy Watch PPG and accelerometer sampling at 25 Hz, watch heart rate
at 1 Hz, and skin/ambient temperature at one-minute intervals across rest,
daily activity, exercise, and stress sessions.

Use it to test:

- PPG-derived heart rate against Polar ECG/heart-rate reference;
- motion feature behavior and motion artefacts;
- signal status and quality handling;
- skin/ambient-temperature parsing;
- timestamps, missing streams, and participant-held-out evaluation code.

Do not use it to train the final gateway autoencoder. It has no SpO2 stream,
does not provide continuous 24-hour target-device windows, and does not contain
the target population. Activity or stress labels must never be converted into
illness/anomaly labels.

Audit an extracted copy outside Git:

```sh
PYTHONPATH=src python3 -m home_health_monitor.datasets.audit \
  galaxyppg /secure-data/GalaxyPPG
```

`convert_galaxy()` in `home_health_monitor.datasets.galaxyppg` emits records
marked `dataset_role: engineering_only` and
`production_training_eligible: false`. It parses Galaxy Watch heart rate, PPG,
skin/ambient temperature, and accelerometer files and derives a simple motion
intensity. It never emits a final decision label.

Samsung's [sensor specification](https://developer.samsung.com/health/sensor/guide/data-specifications.html)
also confirms that watch skin temperature is not body-core temperature. The
target wearable's temperature type and site must remain fixed and versioned.

## Supplied monitoring/fall CSV: excluded

The attached `healthmonitoringandfalldetection.csv` was audited directly:

```text
total rows:          612
unique rows:          36
duplicate rows:      576
repeated cycle size:  36
```

It is the same 36-row sequence repeated 17 times. The repository audit command
reproduces the result:

```sh
PYTHONPATH=src python3 -m home_health_monitor.datasets.audit \
  synthetic /path/to/healthmonitoringandfalldetection.csv
```

That file is not eligible for production training. A corrected Version 2 file,
if supplied, is still synthetic and can be used only for demonstrations and
pipeline/schema tests. The converter requires `corrected` and `v2` or
`version2` in its filename and marks each output row `synthetic_demo`.

## BIDMC: not the gateway training set

The [BIDMC PPG and Respiration Dataset](https://physionet.org/content/bidmc/1.0.0/)
contains short ICU recordings. It can support isolated optical-signal research,
but it lacks the intended temperature/motion combination, does not provide
target-device full-day normal windows, and represents a hospital rather than
home population. It is not fed into this autoencoder.

## Production source: the actual wearable

The production model uses normal periods collected from the exact hardware and
feature manifest deployed in the field. Each participant needs:

- private subject identity and relevant demographic metadata;
- sensor model/configuration and firmware identity;
- temperature type and placement;
- continuous timestamped heart rate, SpO2, temperature, motion, and quality;
- enough coverage to construct multiple 24-hour windows after calibration;
- reviewed normal periods and expected activity variety.

Recruitment must represent the intended deployment area. Personal robust
normalization reduces between-person differences but does not replace diverse
participants or subgroup evaluation.

## Data preparation sequence

1. Verify provenance, consent, sensor identity, and feature-manifest version.
2. Remove duplicate/device-replayed packets using device ID and sequence.
3. Reject invalid timestamps, incompatible temperature sites, and corrupt rows.
4. Build each person's profile from at least 48 hours, 80% valid coverage, and
   eight low-motion hours.
5. Create five-minute medians without copying values into gaps.
6. Create 288-step normal windows and matching masks.
7. Split people—not rows—into train, validation, and test groups.
8. Train and quantify the autoencoder using [the training guide](training.md).

Downloaded datasets, participant metadata, converted records, normal windows,
SQLite files, and model artifacts are excluded from Git. Only code, schemas,
documentation, and non-sensitive test fixtures belong in this repository.
