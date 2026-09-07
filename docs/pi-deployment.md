# Raspberry Pi home-gateway deployment

This guide deploys only the home gateway. The separate wearable samples its
sensors, derives heart rate, performs its immediate check, and transmits a BLE
feature packet. A hardware BLE bridge converts that packet to the documented
local JSON request. This repository does not implement the wearable or SMS.

For the complete model, calibration, prediction, notebook, and deployment flow,
start with the [end-to-end guide](end-to-end.md).

## Runtime layout

```text
wearable --BLE--> hardware bridge --HTTP localhost--> gateway service
                                                     |-- SQLite history/profile
                                                     |-- int8 model (optional)
                                                     `-- normal/anomaly JSON
```

The database survives service restart and reboot. Calibration resumes from
stored packets/profile. The service automatically removes packet and event
history older than 30 days.

## Operating-system recommendation

Use a clean 64-bit Raspberry Pi OS Lite image on the Pi Zero 2 W. A Bookworm
image with Python 3.11 is a conservative runtime choice for current lightweight
TFLite wheels. Disable the desktop and unrelated services to preserve memory.

Raspberry Pi OS Bookworm and newer require `pip` packages to be installed in a
virtual environment; the included unit therefore runs
`/opt/home-health-monitor/.venv/bin/python`. See Raspberry Pi's official
[Python package guidance](https://www.raspberrypi.com/documentation/computers/os.html#use-python-on-a-raspberry-pi).

## 1. Install system packages

```sh
sudo apt update
sudo apt full-upgrade
sudo apt install --no-install-recommends \
  python3 python3-venv ca-certificates git curl
```

Create the non-login account used by `systemd`:

```sh
sudo adduser --system \
  --group \
  --no-create-home \
  --home /opt/home-health-monitor \
  home-health
```

## 2. Install the repository and runtime

```sh
sudo git clone https://github.com/Sammybams/home-health-monitor.git \
  /opt/home-health-monitor
sudo python3 -m venv /opt/home-health-monitor/.venv
sudo /opt/home-health-monitor/.venv/bin/python -m pip install --upgrade pip
sudo /opt/home-health-monitor/.venv/bin/python -m pip install \
  -e '/opt/home-health-monitor[gateway]' tflite-runtime

sudo chown -R root:home-health /opt/home-health-monitor
sudo chmod -R o-rwx /opt/home-health-monitor
sudo chmod -R g+rX /opt/home-health-monitor
```

The gateway tries the lightweight `tflite_runtime` interpreter first and full
TensorFlow second. Full TensorFlow should not be installed on the 512 MB Pi. If
the correct interpreter wheel is unavailable for the chosen OS/Python build,
leave the model absent and verify fallback operation, then select a compatible
64-bit OS/Python image or build the lightweight runtime for that image.

## 3. Install and start the service

```sh
sudo install -o root -g root -m 0644 \
  /opt/home-health-monitor/deploy/home-health-monitor.service \
  /etc/systemd/system/home-health-monitor.service
sudo systemctl daemon-reload
sudo systemctl enable --now home-health-monitor
```

The unit:

- runs as `home-health`, not root;
- listens only on `127.0.0.1:8080`;
- creates `/var/lib/home-health-monitor` through `StateDirectory`;
- stores SQLite at `/var/lib/home-health-monitor/gateway.db`;
- restarts on failure and starts after boot;
- uses a 96 MiB memory ceiling;
- applies read-only filesystem and privilege restrictions.

## 4. Verify fallback operation

The first deployment should work before a model is copied:

```sh
sudo systemctl status home-health-monitor --no-pager
curl -sS http://127.0.0.1:8080/health
```

Expected essentials:

```json
{"status":"ready","database":"ready","model_loaded":false}
```

Send a packet:

```sh
curl -sS -X POST http://127.0.0.1:8080/v2/packets \
  -H 'Content-Type: application/json' \
  --data-binary @/opt/home-health-monitor/examples/packet.json
```

It must return `decision: normal` or `decision: anomaly` even with
`model_loaded: false`.

Repeat the same request. The `(device_id, sequence)` packet is stored only once,
although the caller still receives a prediction. Confirm the latest result:

```sh
curl -sS \
  'http://127.0.0.1:8080/v2/prediction?subject_id=subject-1'
curl -sS \
  'http://127.0.0.1:8080/v2/calibration?subject_id=subject-1'
```

## 5. Connect the BLE bridge

The bridge runs on the gateway side or as an adjacent trusted process. For every
received BLE feature packet it should:

1. verify the BLE characteristic length/version;
2. map the payload exactly to the [v2 packet contract](api.md);
3. preserve device sequence and UTC timestamp;
4. POST once to `http://127.0.0.1:8080/v2/packets`;
5. keep/retry a packet locally if the gateway process is restarting;
6. pass the returned binary result to the local consumer chosen by the wider
   system.

Do not expose the gateway HTTP port on the LAN. BLE pairing, characteristic
UUIDs, byte encoding, and retry storage belong to the hardware integration
repository because they depend on the wearable firmware contract.

## 6. Let calibration complete

Calibration becomes ready after all of the following are present in SQLite:

- a 48-hour elapsed span;
- 80% valid one-minute coverage;
- eight hours of valid low-motion readings.

Check progress using `/v2/calibration`. Restart and reboot during a test to
confirm progress survives. Once ready, verify that a deliberately constructed
large personal deviation triggers `gateway_baseline` in a controlled test.

## 7. Install a trained model

For hardware integration today, the versioned demonstration files are at
`models/development-demo/model.tflite` and `model-metadata.json`. They run the
same inference contract but remain tagged `development_demo`. Use them to prove
installation, memory, latency, and end-to-end packet handling; replace them as
a pair with the selected field candidate later.

Train and select both artifacts off-device. Copy them to a temporary location:

```sh
scp artifacts/gateway/model.tflite pi@PI_ADDRESS:/tmp/model.tflite
scp artifacts/gateway/model-metadata.json \
  pi@PI_ADDRESS:/tmp/model-metadata.json
```

Install atomically controlled files and restart:

```sh
ssh pi@PI_ADDRESS 'sudo install -d -o root -g home-health -m 0750 \
  /opt/home-health-monitor/artifacts/gateway'
ssh pi@PI_ADDRESS 'sudo install -o root -g home-health -m 0640 \
  /tmp/model.tflite /opt/home-health-monitor/artifacts/gateway/model.tflite'
ssh pi@PI_ADDRESS 'sudo install -o root -g home-health -m 0640 \
  /tmp/model-metadata.json \
  /opt/home-health-monitor/artifacts/gateway/model-metadata.json'
ssh pi@PI_ADDRESS 'sudo systemctl restart home-health-monitor'
```

Call `/health` and require `model_loaded: true`. A feature-manifest, checksum,
shape, or quantization mismatch keeps it false while fallback predictions remain
available.

## 8. Measure the release gates on the Pi

Check memory:

```sh
systemctl show home-health-monitor \
  -p MemoryCurrent -p MemoryPeak -p MemoryMax -p TasksCurrent
```

Measure repeated packet requests containing a full preceding 24-hour history in
SQLite. Record median and worst-case request time. Release targets are:

```text
model.tflite < 1 MiB
service peak RSS < 96 MiB
one complete inference < 2 seconds
```

Also test missing model, corrupted metadata, duplicate packet, low-quality
packet streak, reboot during calibration, completed calibration, two persistent
model errors, and one severe model error.

## Logs and updates

```sh
sudo journalctl -u home-health-monitor -n 100 --no-pager
sudo journalctl -u home-health-monitor -f
```

Update only through a reviewed commit, then test and restart:

```sh
sudo git -C /opt/home-health-monitor pull --ff-only origin main
sudo -u home-health env \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/opt/home-health-monitor/src \
  /opt/home-health-monitor/.venv/bin/python -m unittest discover \
  -s /opt/home-health-monitor/tests -v
sudo systemctl restart home-health-monitor
curl -sS http://127.0.0.1:8080/health
```

After initial installation and every service-unit change, reboot once and
confirm the API returns without an SSH session:

```sh
sudo reboot
```
