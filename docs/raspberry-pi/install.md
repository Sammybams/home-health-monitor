# Install on the Pi

## 1. Prepare the Pi

Install 64-bit Raspberry Pi OS Lite and connect it to the internet. Log in by
SSH or open its terminal.

Git is normally available. If `git --version` fails, run:

```sh
sudo apt update
sudo apt install -y git
```

## 2. Install the gateway

Run these commands:

```sh
sudo git clone https://github.com/Sammybams/home-health-monitor.git \
  /opt/home-health-monitor
cd /opt/home-health-monitor
sudo ./deploy/install-pi.sh
```

The installer adds the small LiteRT runtime, installs the included development
model, creates the database service, and makes the gateway start after every
reboot. It can take several minutes on a Pi Zero 2 W.

## 3. Confirm it works

```sh
sudo /opt/home-health-monitor/deploy/verify-pi.sh
```

Successful output includes:

```text
service: running
database: ready
model: gateway-ae-cc9b07502845
model checksum: verified
sample prediction: normal
```

The prediction may also be `anomaly`; both are valid binary results.

Do not keep the command from the shared screenshot running in a terminal. That
command is useful for development, but it stops when the terminal closes. The
installer runs the gateway in the background using `systemd` and starts it
again after a reboot.
