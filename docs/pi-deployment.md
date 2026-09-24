# Raspberry Pi deployment

Use the short guides in [`docs/raspberry-pi`](raspberry-pi/README.md):

1. [Install the gateway](raspberry-pi/install.md)
2. [Connect the BLE integration and consume predictions](raspberry-pi/use.md)
3. [Verify, update, and troubleshoot it](raspberry-pi/operate.md)

The supported layout is:

```text
wearable --BLE--> local BLE process --HTTP localhost--> gateway service
                                                       |-- SQLite history
                                                       |-- personal baseline
                                                       `-- int8 model
```

The service runs as the restricted `home-health` user, listens on
`127.0.0.1:8080`, stores data at
`/var/lib/home-health-monitor/gateway.db`, starts after boot, and has a 96 MiB
memory limit. This repository contains only the home gateway; BLE firmware and
SMS delivery are outside its scope.

The installer currently activates registered V1. The 4.2 KiB V2 real-PPG model
and LiteRT-compatible runtime are committed, but V2 is deliberately not wired
to the service until the exact wearable 20-field payload is available. Once it
is, the deployment sequence is: implement the BLE-to-vector adapter, rerun the
V2 notebook on target-wearable normal data, copy model plus metadata, benchmark
under the same 96 MiB service limit, and only then change the registry default.
