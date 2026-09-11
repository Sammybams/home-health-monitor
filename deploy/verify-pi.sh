#!/bin/sh
set -eu

INSTALL_DIR=/opt/home-health-monitor
SERVICE_NAME=home-health-monitor
BASE_URL=http://127.0.0.1:8080

usage() {
    cat <<'EOF'
Verify the installed Raspberry Pi gateway.

Success means the system service is running, /health reports model_loaded=true,
the model checksum matches its metadata, and a sample packet returns normal or anomaly.

Run: sudo /opt/home-health-monitor/deploy/verify-pi.sh
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

PYTHON="$INSTALL_DIR/.venv/bin/python"
MODEL="$INSTALL_DIR/artifacts/gateway/model.tflite"
METADATA="$INSTALL_DIR/artifacts/gateway/model-metadata.json"

for required in "$PYTHON" "$MODEL" "$METADATA"; do
    if [ ! -e "$required" ]; then
        echo "error: missing $required" >&2
        exit 1
    fi
done

if ! systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "error: $SERVICE_NAME is not running" >&2
    systemctl status "$SERVICE_NAME" --no-pager >&2 || true
    exit 1
fi

HEALTH=$(curl --fail --silent --show-error "$BASE_URL/health")
"$PYTHON" -c '
import json, sys
health = json.loads(sys.argv[1])
if health.get("status") != "ready" or health.get("database") != "ready":
    raise SystemExit("error: gateway or database is not ready")
if health.get("model_loaded") is not True:
    reason = health.get("model_reason")
    raise SystemExit(f"error: model did not load: {reason}")
' "$HEALTH"

EXPECTED_SHA=$("$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["model_sha256"])' \
    "$METADATA")
ACTUAL_SHA=$("$PYTHON" -c \
    'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
    "$MODEL")
if [ "$EXPECTED_SHA" != "$ACTUAL_SHA" ]; then
    echo "error: installed model checksum does not match its metadata" >&2
    exit 1
fi

PREDICTION=$(curl --fail --silent --show-error \
    -X POST "$BASE_URL/v2/packets" \
    -H 'Content-Type: application/json' \
    --data-binary "@$INSTALL_DIR/examples/packet.json")
DECISION=$("$PYTHON" -c \
    'import json,sys; print(json.loads(sys.argv[1]).get("decision", ""))' \
    "$PREDICTION")
if [ "$DECISION" != "normal" ] && [ "$DECISION" != "anomaly" ]; then
    echo "error: sample request did not return a binary prediction" >&2
    exit 1
fi

MODEL_ID=$("$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["model_id"])' \
    "$METADATA")

echo "service: running"
echo "database: ready"
echo "model: $MODEL_ID"
echo "model checksum: verified"
echo "sample prediction: $DECISION"
systemctl show "$SERVICE_NAME" \
    -p MemoryCurrent -p MemoryPeak -p MemoryMax -p TasksCurrent
