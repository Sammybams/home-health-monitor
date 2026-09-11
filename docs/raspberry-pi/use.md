# Use the gateway

## What sends data to it

The wearable measures PPG/SpO2, temperature, and motion. It derives heart rate
from PPG and sends its feature packet over BLE. Hardware-specific BLE code must
convert that packet to the repository's [JSON format](../api.md).

The BLE process running on the same Pi sends each packet here:

```text
POST http://127.0.0.1:8080/v2/packets
```

Example:

```sh
curl -sS -X POST http://127.0.0.1:8080/v2/packets \
  -H 'Content-Type: application/json' \
  --data-binary @/opt/home-health-monitor/examples/packet.json
```

The important output is always one of:

```json
{"decision":"normal"}
```

```json
{"decision":"anomaly"}
```

## What happens inside the Pi

1. The packet is checked and saved in SQLite.
2. During the first 48 hours, the gateway learns that person's normal values.
3. After calibration, it checks new values against that personal baseline.
4. It also builds a rolling 24-hour window for the autoencoder.
5. The wearable, sensor-quality, personal-baseline, and autoencoder checks are
   combined into the final binary result.

There is always a prediction, including during calibration. The database and
calibration progress survive restarts.

The HTTP service listens only on `127.0.0.1`, so it is intended for a BLE
process on the same Pi. This repository does not include SMS or notification
delivery.
