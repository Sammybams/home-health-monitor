# Simple API guide

The service uses JSON over HTTP. It has no login or encryption of its own, so it
should normally listen only on the Raspberry Pi at `127.0.0.1`.

## Check the service

Send:

```text
GET /health
```

The endpoint returns HTTP 200 when the service is running:

```json
{
  "status": "ready",
  "change_assessment_available": true,
  "illness_model_loaded": false,
  "illness_model_reason": "could not load model: ..."
}
```

`illness_model_loaded: false` does not stop personal change detection.

## Request a prediction

Send:

```text
POST /v1/predict
Content-Type: application/json
```

The body contains:

```json
{
  "subject_id": "private-person-id",
  "baseline": {
    "schema_version": 1,
    "subject_id": "private-person-id",
    "created_at": "2026-08-01T08:00:00Z",
    "healthy_days": 7,
    "sample_count": 10080,
    "signals": {
      "resting_body_temperature_c": {"median": 36.6, "mad_scale": 0.12},
      "resting_heart_rate_bpm": {"median": 70, "mad_scale": 3.5},
      "resting_body_ambient_delta_c": {"median": 9.4, "mad_scale": 0.3},
      "daily_motion_fraction": {"median": 0.4, "mad_scale": 0.08}
    }
  },
  "observations": [
    {
      "timestamp": "2026-08-01T08:00:00Z",
      "body_temperature_c": 36.7,
      "ambient_temperature_c": 27.1,
      "heart_rate_bpm": 72,
      "motion": 0,
      "resting": true
    }
  ]
}
```

The example shows only one observation to keep it readable. A real request must
contain 12–10,000 chronological observations spanning at least 30 minutes.

Required observation fields are:

- `timestamp`, including its timezone;
- `body_temperature_c`;
- `ambient_temperature_c`;
- `heart_rate_bpm`;
- `motion`, either `0` or `1`.

`resting` is optional and accepts `true`, `false`, `0` or `1`. An explicit value
is preferred when the collector knows whether the person is resting or asleep.

The baseline is optional, but without it the change result is
`insufficient_data`. Never attach one person's baseline to another person's
request; the service checks the IDs and rejects a mismatch.

## Response without an illness model

Shortened example, with individual signal details omitted:

```json
{
  "change_assessment": {
    "status": "within_personal_baseline",
    "score": 1.2,
    "threshold": 3.5,
    "resting_window_hours": 6,
    "resting_sample_count": 120,
    "signals": {},
    "interpretation": "No configured measurement changed substantially from this person's healthy baseline.",
    "disclaimer": "Change detection only; this score is not an illness probability or diagnosis."
  },
  "prediction": null,
  "model": {
    "status": "not_loaded",
    "reason": "could not load model: ..."
  },
  "data_quality": {
    "warnings": []
  }
}
```

The score is the largest robust difference among resting body temperature,
resting heart rate, body-versus-room temperature and daily movement. A value at
or above `3.5` produces `unusual_change`. This is a conservative technical
starting threshold, not a medical emergency limit.

If fewer than 12 resting readings exist in the latest six hours, the service
uses the full submitted history and adds a warning. If the full history still
has fewer than 12, it returns `insufficient_data`.

## Response with a validated illness model

When `artifacts/model.json` is installed, `prediction` also contains separate
current-risk and future-risk results. The personal `change_assessment` remains
present. Only a model validated with real illness labels should be installed.

## Common errors

- HTTP 400: malformed JSON or content length
- HTTP 413: request larger than the configured limit
- HTTP 415: content type is not JSON
- HTTP 422: missing, unknown, out-of-range or inconsistent input

Raw request bodies are not written to application logs.
