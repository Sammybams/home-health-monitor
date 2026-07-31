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
- the time of every reading.

A reading looks like this:

```json
{
  "timestamp": "2026-07-31T08:00:00Z",
  "body_temperature_c": 36.7,
  "ambient_temperature_c": 27.1,
  "heart_rate_bpm": 72,
  "motion": 1
}
```

`motion` is `1` when motion was detected and `0` when it was not.

## What it returns

After a real trained model has been installed, the service returns:

- whether the readings show lower or higher current risk;
- a lower or higher future-risk result;
- the estimated future-risk probability;
- how many hours into the future the model covers;
- warnings when the submitted sensor history has gaps or is too short;
- the identity of the model that produced the result.

The response deliberately says `lower_risk` or `higher_risk`. It does not say
that somebody is definitely healthy or sick.

## What happens inside

The Pi does four small jobs:

1. It checks that the JSON and sensor readings make sense.
2. It summarises the last 1, 6, and 24 hours.
3. It gives those summaries to a tiny mathematical model.
4. It returns the model's result as JSON.

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
- A very small model runner with no external runtime dependencies
- Separate current-risk and future-risk predictions
- A `/health` readiness endpoint
- A `/v1/predict` prediction endpoint
- Offline model-training and export code
- A Raspberry Pi `systemd` service definition
- Automated tests

The repository intentionally contains no made-up trained model. Until a model
created from real labelled data is copied to `artifacts/model.json`, `/health`
and `/v1/predict` return HTTP `503`. This prevents a demonstration formula from
being mistaken for a medically tested system.

## Running the service

Python 3.10 or newer is required. Prediction has no third-party dependencies.

```sh
PYTHONPATH=src python3 -m home_health_monitor \
  --host 127.0.0.1 \
  --port 8080 \
  --model artifacts/model.json
```

Check whether a model is ready:

```sh
curl -sS http://127.0.0.1:8080/health
```

Send the included example request:

```sh
curl -sS -X POST http://127.0.0.1:8080/v1/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

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
[`docs/design.md`](docs/design.md).
