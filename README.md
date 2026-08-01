# Home Health Monitor

This project is a small health-risk screening service designed to run on a
Raspberry Pi with only 512 MB or 1 GB of memory.

It is important to understand what that means: the service can look for patterns
that a trained model has learned, but it is **not a doctor**, it does not diagnose
illness, and it must not replace emergency or professional medical care.

## What it receives

The service receives a JSON list containing roughly one day of readings:

- body temperature;
- room or ambient temperature;
- heart rate;
- whether the person was moving;
- optionally, whether the reading was taken while resting or sleeping;
- the time of every reading.

A reading looks like this:

```json
{
  "timestamp": "2026-07-31T08:00:00Z",
  "body_temperature_c": 36.7,
  "ambient_temperature_c": 27.1,
  "heart_rate_bpm": 72,
  "motion": 1,
  "resting": false
}
```

`motion` is `1` when motion was detected and `0` when it was not. `resting` is
optional. When it is missing, the service treats `motion: 0` as the best
available resting signal.

## What it returns

The service can immediately return:

- whether the latest readings are within the person's baseline;
- whether one or more measurements changed unusually;
- which measurements changed and in which direction;
- warnings when the submitted history is incomplete.

This change score is not an illness probability. After a separately validated
illness model has been installed, the same response can also include:

- whether the readings show lower or higher current risk;
- a lower or higher future-risk result;
- the estimated future-risk probability;
- how many hours into the future the model covers;
- warnings when the submitted sensor history has gaps or is too short;
- the identity of the model that produced the result.

The response deliberately uses `within_personal_baseline`, `unusual_change`,
`lower_risk` or `higher_risk`. It does not say that somebody is definitely
healthy or sick.

## What happens inside

The Pi does five small jobs:

1. It checks that the JSON and sensor readings make sense.
2. It summarises the last 1, 6, and 24 hours.
3. It compares resting readings with a small personal baseline profile.
4. If installed, it also gives the summaries to a tiny illness-risk model.
5. It returns the results as JSON.

The summaries include average, minimum, maximum, variation and direction of
change. The service also measures activity, missing time intervals and the
difference between body and room temperature.

The Pi only performs predictions. Model training happens on a more powerful
development computer and creates a small `model.json` file for the Pi. This is
why the Pi does not need NumPy, pandas, scikit-learn, FastAPI or a deep-learning
framework.

## What is already implemented

- JSON validation and useful error responses
- Fixed limits to protect the Pi's memory
- One-, six-, and 24-hour feature calculation
- Creation of a robust profile from 7–30 healthy days
- Personalized resting temperature, heart-rate and activity comparison
- A change assessment that works without an illness model
- A very small model runner with no external runtime dependencies
- Separate current-risk and future-risk predictions
- A `/health` readiness endpoint
- A `/v1/predict` prediction endpoint
- Offline model-training and export code
- A Raspberry Pi `systemd` service definition
- Automated tests

The repository intentionally contains no made-up illness model. Without
`artifacts/model.json`, the service still returns the personal change assessment
and sets `prediction` to `null`. This prevents a demonstration formula from
being mistaken for a medically tested probability.

## Building a person's baseline

First collect 7–30 days that are believed to represent the person's ordinary,
healthy state. Store one request-shaped JSON object per line in a JSON Lines
file. Every line must use the same private `subject_id`, span at least six hours
and contain at least 12 resting samples.

Build the small profile:

```sh
PYTHONPATH=src python3 -m home_health_monitor.baseline_cli \
  data/healthy-days.jsonl \
  data/person-baseline.json
```

The baseline stores only four robust summaries, not all of the old readings.
The collector adds this small object under the `baseline` field of each future
prediction request. Its `subject_id` must match the request. An illustrative
profile is available at [`examples/baseline.json`](examples/baseline.json).

## Running the service

Python 3.10 or newer is required. Prediction has no third-party dependencies.

```sh
PYTHONPATH=src python3 -m home_health_monitor \
  --host 127.0.0.1 \
  --port 8080 \
  --model artifacts/model.json
```

Check whether the service and optional illness model are ready:

```sh
curl -sS http://127.0.0.1:8080/health
```

Send the included example request:

```sh
curl -sS -X POST http://127.0.0.1:8080/v1/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

Because this basic example has no baseline, its change result is
`insufficient_data`. Add a generated baseline object to obtain a personal
comparison. The complete request and response rules are in
[`docs/api.md`](docs/api.md).

## Training a model

Training must happen away from the Pi. Install the training tools on a
development computer:

```sh
python3 -m pip install -e '.[train]'
```

Each line in the training file represents one person's sensor window. It must
include a private person ID, the observations, and two answers recorded from the
real world:

```json
{
  "subject_id": "private-person-id",
  "observations": [],
  "labels": {
    "current_unhealthy": 0,
    "unhealthy_within_horizon": 1
  }
}
```

Then run:

```sh
home-health-train data/windows.jsonl artifacts/model.json \
  --future-horizon-hours 24
```

The trainer keeps different people in training and testing. That avoids testing
the model on the same people it has already learned from. It reports two useful
quality measurements and writes the small model file.

The default `0.5` decision point is only a software starting value. Medical and
product owners must choose the final decision point after reviewing missed
illnesses and false alarms.

## Testing

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Installing on the Pi

Copy this project and the trained model to `/opt/home-health-monitor`. Create a
non-administrator Linux user called `home-health`, then adapt and install
[`deploy/home-health-monitor.service`](deploy/home-health-monitor.service).

The included service starts with a 64 MB memory limit. That limit must be tested
on the real Pi and operating-system image. Keep the server bound to `127.0.0.1`
unless an authenticated and encrypted gateway is placed in front of it.

For the choices that still need to be made before real training, read
[`docs/design.md`](docs/design.md). The researched public datasets and the
recommended way to combine their lessons are documented in
[`docs/data-sources.md`](docs/data-sources.md).
