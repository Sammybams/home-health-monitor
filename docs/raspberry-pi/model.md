# The included model

## What trained it

The included model was trained from
`notebooks/data/supplied-monitoring.csv`, the supplied monitoring data. That
file has 612 rows, but only 36 are different and only 11 different rows are
marked normal.

Those normal values seeded a repeatable development dataset:

- 60 simulated people;
- four 24-hour normal windows per person;
- 240 windows in total;
- separate people for training, validation, and testing.

The model learned only normal patterns. High heart rate, low SpO2, high
temperature, and changed motion were added later to check whether it detected
clear changes.

GalaxyPPG was inspected as a useful reference for heart rate, temperature, and
motion. It did not contain SpO2, so it was not used to train this complete
four-signal model.

## What the model receives

It receives the latest 24 hours as 288 five-minute blocks. Each block contains
heart rate, SpO2, temperature, motion, sensor quality, and indicators showing
which measurements were present.

The model tries to rebuild that 24-hour pattern. A large rebuilding error means
the pattern does not look like the normal patterns it learned.

## What is installed

- Type: small integer autoencoder
- File size: 18 KiB
- Model ID: `gateway-ae-cc9b07502845`
- Output: `normal` or `anomaly`

The executable training source is
[`notebooks/train-and-evaluate-autoencoder.ipynb`](../../notebooks/train-and-evaluate-autoencoder.ipynb).
Training happens on a development computer, not on the Pi. Later, the same
notebook can train a replacement using normal recordings from the real wearable
and target population.
