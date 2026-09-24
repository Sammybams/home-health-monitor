# Model V2: real-PPG development candidate

## Status

`implementation_in_progress` — no V2 artifact or performance claim exists yet.

## Intended training source

V2 will use version 1.1.0 of the Pulse Transit Time PPG dataset. It contains 66
approximately eight-minute recordings from 22 healthy participants performing
sitting, walking, and running. Continuous channels include ECG, six PPG
waveforms, three temperatures, acceleration, gyroscope, and attachment
pressure.

SpO2 is recorded only at the beginning and end of each activity. It will not be
interpolated or described as continuous training data. SpO2 remains part of the
wearable and personal-baseline decision paths, but not V2 reconstruction until
continuous target-wearable data exists.

## Intended model and runtime

V2 will reconstruct one versioned wearable feature vector at a time. The
gateway will process the vectors received during each eight-minute interval and
combine median, upper-percentile, maximum, above-threshold fraction, and
persistence evidence. The first 48 hours remain personal calibration, not one
model input.

The exact input feature manifest is intentionally not frozen until the wearable
formulas, units, sensors, and decoded BLE payload are confirmed. Training and
deployment must use the same formulas.

## Required validation

- feature extraction checked against ECG peak annotations;
- all splits grouped by participant;
- five-fold grouped validation inside the development participants;
- a locked participant holdout evaluated once;
- real-normal false anomalies reported by person and activity;
- controlled changes kept out of training and labelled as simulation;
- thresholds recalculated using the quantized model;
- model size, memory, and latency measured on the 512 MB Pi;
- authoritative notebook executable from raw archive to reports and artifact.

V2 can become the default only after these results and its exact limitations are
committed alongside the artifact.
