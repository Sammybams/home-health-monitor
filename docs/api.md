# Gateway JSON API

The service listens on `127.0.0.1:8080` by default. The hardware-specific BLE
bridge posts one already-filtered wearable feature packet after each 2-5 second
sampling cycle.

## POST `/v2/packets`

Send `Content-Type: application/json`. A complete example is in
[`examples/packet.json`](../examples/packet.json).

### Required packet fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Must be `1`. |
| `subject_id` | Private person identifier. |
| `device_id` | Wearable identifier. |
| `sequence` | Non-negative device packet counter. `(device_id, sequence)` is unique. |
| `timestamp` | ISO-8601 time with timezone; stored as UTC. |
| `sample_duration_seconds` | Wearable awake/sample period from `2` through `5`. |
| `firmware_version` | Wearable firmware identity. |
| `sensor_config_id` | Exact sensor/placement configuration identity. |
| `feature_manifest_id` | Must match the installed model; current value is `features-v1`. |
| `heart_rate_bpm` | Heart rate derived by the wearable from its optical signal. |
| `spo2_percent` | Blood oxygen saturation. |
| `temperature_c` | Body/skin/surface temperature in Celsius. |
| `temperature_type` | `skin`, `core`, or `surface`. |
| `temperature_site` | Measurement position, such as `wrist`. |
| `motion_intensity` | Non-negative accelerometer-derived motion magnitude. |
| `motion` | `0` for no detected motion or `1` for motion. |
| `quality` | `ppg`, `spo2`, `temperature`, and `motion` values from `0` to `1`. |
| `wearable` | Immediate wearable assessment described below. |

The wearable object is:

```json
{
  "decision": "normal",
  "score": 0.42,
  "deviations": {
    "heart_rate_bpm": 0.42,
    "spo2_percent": -0.2,
    "temperature_c": 0.1,
    "motion_intensity": 0.05
  },
  "reason_codes": []
}
```

`decision` is `normal` or `anomaly`. An anomaly requires at least one reason
code. Deviations are the signed robust z-scores calculated by the wearable.
Unknown fields are rejected so accidental additions do not silently change the
model contract.

### Successful response

Every accepted packet returns HTTP 200 and a binary decision:

```json
{
  "decision": "anomaly",
  "triggered_by": ["gateway_baseline"],
  "reason_codes": ["spo2_percent_personal_deviation"],
  "scores": {
    "wearable": 0.8,
    "gateway_baseline_max_abs_z": 5.2,
    "gateway_model_error": 0.13
  },
  "contributing_signals": ["spo2_percent"],
  "measurements": {
    "heart_rate_bpm": 92,
    "spo2_percent": 91.5,
    "temperature_c": 34.1,
    "motion": 0
  },
  "calibration": {
    "status": "ready",
    "hours": 48.0
  },
  "timestamp": "2026-09-07T20:00:00Z"
}
```

`triggered_by` can contain:

- `wearable` — the wearable's immediate check flagged the packet;
- `gateway_baseline` — a current value is at least four robust personal scale
  units from baseline;
- `gateway_autoencoder` — the 24-hour reconstruction error is severe once or
  above threshold for two consecutive windows;
- `data_quality` — the same sensor quality remained below `0.5` for three
  recent packets.

During calibration, a normal result still has `decision: normal` and
`calibration.status: collecting`. After 48 elapsed hours, at least 80% valid
coverage, and eight low-motion hours, status becomes `ready`.

## GET `/v2/prediction?subject_id=...`

Returns the latest stored prediction for the subject. A missing `subject_id`
returns HTTP 400. A subject with no prediction returns HTTP 404.

## GET `/v2/calibration?subject_id=...`

Returns the calibration summary attached to the subject's latest prediction:

```json
{"status":"collecting","hours":12.0,"coverage":0.91}
```

or:

```json
{"status":"ready","hours":48.0}
```

## GET `/health`

```json
{
  "status": "ready",
  "database": "ready",
  "model_loaded": false,
  "model_reason": "could not load autoencoder artifact: ..."
}
```

`model_loaded: false` does not disable predictions. Wearable, quality, and
personal-baseline logic continue to run.

## HTTP errors

- `400` — malformed JSON length or missing query subject;
- `404` — unknown endpoint or no stored prediction;
- `413` — body is empty or exceeds the configured limit (64 KiB by default);
- `415` — content type is not JSON;
- `422` — packet fields, units, ranges, identifiers, or versions are invalid.

The server logs request metadata, not raw request bodies.
