# Development autoencoder artifact

This directory contains a runnable **development demonstration**, not the final
field model. It exists so the complete home-gateway path can be exercised now,
including int8 inference, reconstruction scoring, thresholding, and plots.

## Contents

- `model.tflite`: 18 KiB fully int8 Conv1D autoencoder;
- `model-metadata.json`: tensor contract, checksum, thresholds, split, and metrics;
- `training-report.json`: loss history, score distributions, and one reconstruction;
- `performance-summary.json`: confusion counts and fixed-threshold metrics;
- `data-evidence.json`: reproducible profile of the three reviewed data sources.

The model was trained with seed `42` for 30 epochs on 240 generated normal
windows representing 60 virtual subjects. Each subject has four 24-hour
windows. The source generator used the 11 distinct `Normal` rows in the
supplied 612-row monitoring CSV to set development variability. It did not
train on the supplied condition labels or controlled anomalies.

GalaxyPPG and BIDMC were profiled to check real sensor ranges and feature
coverage. They were not joined into invented people and were not directly fed
to this four-feature autoencoder: GalaxyPPG lacks SpO2, while BIDMC lacks
temperature and motion and covers only short ICU recordings.

The authoritative run is the executed
[`train-and-evaluate-autoencoder.ipynb`](../../notebooks/train-and-evaluate-autoencoder.ipynb)
notebook. Rebuild instructions and interpretation are in the
[development-model report](../../docs/development-model.md), and the complete
runtime path is in the [end-to-end guide](../../docs/end-to-end.md).
