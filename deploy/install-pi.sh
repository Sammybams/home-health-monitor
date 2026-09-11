#!/bin/sh
set -eu

INSTALL_DIR=/opt/home-health-monitor
SERVICE_NAME=home-health-monitor

usage() {
    cat <<'EOF'
Install the home gateway on a 64-bit Raspberry Pi:

  sudo git clone https://github.com/Sammybams/home-health-monitor.git /opt/home-health-monitor
  cd /opt/home-health-monitor && sudo ./deploy/install-pi.sh
  sudo ./deploy/verify-pi.sh

The installer adds the lightweight runtime, bundled development model,
SQLite service, and automatic startup. It does not install SMS or BLE code.
EOF
}

if [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi
if [ "$#" -ne 0 ]; then
    usage >&2
    exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

if [ "$REPOSITORY_DIR" != "$INSTALL_DIR" ]; then
    echo "error: clone this repository at $INSTALL_DIR before running the installer" >&2
    exit 1
fi
if [ "$(id -u)" -ne 0 ]; then
    echo "error: run this installer with sudo" >&2
    exit 1
fi
if [ "$(dpkg --print-architecture)" != "arm64" ]; then
    echo "error: this installer requires 64-bit Raspberry Pi OS (arm64)" >&2
    exit 1
fi

echo "[1/6] Installing small operating-system packages"
apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl git python3 python3-venv

echo "[2/6] Creating the restricted service account"
if ! id home-health >/dev/null 2>&1; then
    adduser --system --group --no-create-home \
        --home "$INSTALL_DIR" home-health
fi

echo "[3/6] Installing the Python gateway and LiteRT"
if [ ! -x "$INSTALL_DIR/.venv/bin/python" ]; then
    python3 -m venv "$INSTALL_DIR/.venv"
fi
"$INSTALL_DIR/.venv/bin/python" -m pip install --upgrade pip
"$INSTALL_DIR/.venv/bin/python" -m pip install -e "$INSTALL_DIR[gateway-pi]"

echo "[4/6] Installing the bundled development model"
install -d -o root -g home-health -m 0750 "$INSTALL_DIR/artifacts/gateway"
install -o root -g home-health -m 0640 \
    "$INSTALL_DIR/models/development-demo/model.tflite" \
    "$INSTALL_DIR/artifacts/gateway/model.tflite"
install -o root -g home-health -m 0640 \
    "$INSTALL_DIR/models/development-demo/model-metadata.json" \
    "$INSTALL_DIR/artifacts/gateway/model-metadata.json"

echo "[5/6] Securing files and enabling automatic startup"
chown -R root:home-health "$INSTALL_DIR"
chmod -R o-rwx "$INSTALL_DIR"
chmod -R g+rX "$INSTALL_DIR"
install -o root -g root -m 0644 \
    "$INSTALL_DIR/deploy/home-health-monitor.service" \
    "/etc/systemd/system/$SERVICE_NAME.service"
systemctl daemon-reload
systemctl enable --now "$SERVICE_NAME"

echo "[6/6] Checking the running gateway"
"$INSTALL_DIR/deploy/verify-pi.sh"

echo "Installation complete. The gateway will start automatically after reboot."
