# Data sources and the recommended training plan

This review was completed on 31 July 2026. Its purpose is to answer a simple
question: can an existing dataset train this product honestly?

## Short answer

No single public dataset has all of these at the same time:

- body or skin temperature;
- ambient temperature;
- heart rate;
- motion;
- full-day recordings;
- trustworthy labels showing current illness and later illness.

There are useful public datasets, but each solves only part of the problem. We
should use them for the part they genuinely support and collect a smaller local
dataset for the missing part. Joining unrelated rows from unrelated people would
create artificial examples and an invalid model.

## Best available sources

### 1. Stanford Phase 2 wearable-infection data

Source: [Nature Medicine study](https://www.nature.com/articles/s41591-021-01593-2)
and [public algorithm repository](https://github.com/StanfordBioinformatics/wearable-infection).

The study enrolled 3,318 people; 2,155 supplied wearable data. It used heart
rate, steps, sleep and health surveys. Eighty-four people with confirmed COVID-19
had enough wearable data around the infection period. The published system
detected unusual changes before symptoms in many of those cases.

The raw heart-rate and step archive is publicly downloadable, but it is about
5.54 GB. It does not contain the required body and ambient temperature pair.

Use it for:

- reproducing personalized resting-heart-rate baselines;
- testing infection-window and future-label creation;
- comparing simple alert algorithms;
- understanding false alerts caused by stress, alcohol, travel and exercise.

Do not use it as proof that our temperature-based model works.

### 2. GalaxyPPG

Source: [Scientific Data paper](https://www.nature.com/articles/s41597-025-05152-z)
and [Zenodo dataset](https://zenodo.org/records/14635823).

This is the closest match to the device inputs. Galaxy Watch recordings include
heart rate, accelerometer motion, body/skin temperature and ambient temperature.
The dataset covers 24 healthy participants performing rest, typing, standing,
walking, jogging, running and stress tasks. The archive is about 482 MB.

Use it for:

- building and testing the input converter;
- converting accelerometer readings to the configured motion value;
- learning how activity changes measured heart rate and skin temperature;
- testing missing-data and signal-quality handling;
- checking the body-versus-ambient temperature calculation.

Do not use its activity or stress labels as “sick”. It contains no illness
outcomes and only short recording sessions.

### 3. CovIdentify

Source: [PhysioNet CovIdentify](https://physionet.org/content/covidentify/1.0.0/).

This study has daily resting heart rate, steps, symptom surveys and self-reported
COVID-19 test results. Of 2,887 participants who connected a smartwatch, 1,289
reported at least one test result: 132 positive and 1,157 negative.

Use it for:

- external testing of daily heart-rate and activity features;
- testing symptom and diagnostic-date label creation;
- checking whether results transfer across Fitbit, Garmin and Apple devices.

Limitations:

- values are mainly daily summaries rather than our minute-level input;
- it has no matching temperature stream;
- test results were self-reported;
- access requires a PhysioNet account, CITI training and a signed data-use
  agreement.

The dataset must not be copied into this Git repository.

### 4. TemPredict

Source: [published TemPredict study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8891385/)
and [UCSF study page](https://osher.ucsf.edu/research/current-research-studies/tempredict).

This is the closest published illness study in terms of signals. The Oura Ring
collected skin temperature, heart rate, heart-rate variability, respiratory rate
and activity, with symptom and COVID-19 information.

It strongly supports the idea behind this product, but Oura's data-use policy
does not allow the wearable sensor data to be released freely to third parties.
We can use the published method as evidence, not use its private readings to
train our model.

### 5. Hospital infection datasets

MIMIC contains temperature, heart rate, respiratory rate and infection-related
hospital information. It is useful for research, but hospital patients, hospital
sensors and home wearable users are different populations.

A 2025 study found that a hospital model applied directly to home wearable data
was close to chance (AUROC 0.527). Performance improved only after restricting
the data to sleep and correcting the difference between datasets. See the
[Scientific Reports study](https://www.nature.com/articles/s41598-025-17593-y).

Therefore we should not train on hospital records and call the result a home
health model without a separately validated adaptation study.

## What these sources teach us

### Compare a person with themselves

A heart rate of 85 may be normal for one person and unusual for another. The
Stanford work first learned each person's healthy overnight pattern and detected
departures from that pattern.

The product should therefore keep:

- a short prediction window: the most recent 24 hours;
- a longer healthy baseline: ideally 7–14 valid days to start, then updated
  carefully over time.

The baseline can be stored as small summaries; the Pi does not need to retain
every old sensor reading.

### Prefer resting or sleeping readings

Running raises heart rate without indicating illness. Motion helps us separate
rest from exercise, but an explicit sleep/rest signal would be better. The model
should calculate both all-day and resting-only summaries.

### Temperature needs the actual device

Skin temperature, core temperature and contactless temperature are not
interchangeable. Ambient temperature helps explain changes in a skin sensor, but
the relationship depends on placement and hardware. GalaxyPPG can help build the
processing code; final calibration must use the product's actual sensors.

### An alert is not a disease name

Wearable changes can result from infection, exercise, stress, poor sleep,
alcohol, medication or sensor movement. The first honest output is “unusual
physiological change”, followed by advice defined by the product's medical and
safety owners. It is not “you have COVID” or “you will definitely be sick”.

## Final input recommendation

Keep the original four inputs required:

1. body or skin temperature;
2. ambient temperature;
3. heart rate;
4. motion.

Always include timestamp, sensor/device identity and a signal-quality indicator
when the hardware provides one.

Useful optional inputs, in priority order:

1. sleep or resting state;
2. respiratory rate;
3. heart-rate variability, preferably RMSSD;
4. blood oxygen saturation (SpO2);
5. a short symptom report.

Symptoms and diagnostic test results are especially important during data
collection because they create labels. They do not all need to be available for
every normal prediction.

## Recommended product in two stages

### Stage 1: personalized change monitor — implemented

This stage works before there are enough illness cases:

1. Collect 7–14 healthy baseline days from the person.
2. Calculate their normal resting temperature, heart rate and activity ranges.
3. Examine the most recent 24 hours.
4. Report normal, unusual change or insufficient data.

This is an anomaly detector. Its score is not an illness probability.

### Stage 2: validated illness-risk model

After target-device data includes enough independently confirmed illness events:

1. Define current illness and a fixed future horizon, such as 24 hours.
2. Create examples without allowing post-event readings into pre-event windows.
3. Split the data by person.
4. Train and compare logistic regression and small tree models off-device.
5. Calibrate the probabilities and choose thresholds on untouched people.
6. Export the small winning model to the Pi.

The response may then contain both:

- a personalized change score;
- a validated illness-risk probability.

Until the validated probability is available, the implemented API still returns
provisional current and future classifications. It marks their method,
confidence and score type so they cannot be confused with dataset-calibrated
probabilities.

## Data we should collect with the real device

For every participant:

- at least 7–14 ordinary baseline days;
- the four required signals with timestamps and quality flags;
- sleep/rest state if possible;
- daily symptoms and symptom-start time;
- diagnostic test type, result and collection time when available;
- vaccination, strenuous exercise, alcohol, travel and relevant medication
  events, because these can explain false alerts.

The study owner should obtain ethics and privacy review before collecting health
data. The required number of participants and illness events should come from a
statistical power calculation after defining the target and acceptable error,
not from an arbitrary number chosen by the software team.

## Repository rule

Downloaded health datasets remain outside Git. The repository should contain
only download instructions, checksums, schemas and conversion scripts. Every
dataset's licence or data-use agreement must be checked before downloading,
processing or redistributing it.
