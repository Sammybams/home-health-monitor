#!/bin/sh
set -eu

INSTALL_DIR=/opt/home-health-monitor
SERVICE_NAME=home-health-monitor
BASE_URL=http://127.0.0.1:8080

usage() {
    cat <<'EOF'
Verify the installed Raspberry Pi gateway.

Success means the service is running, /health reports model_loaded=true and
vector_model_loaded=true, both checksums match, and both sample routes return
normal or anomaly.

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
VECTOR_MODEL="$INSTALL_DIR/artifacts/real-ppg-v2/model.tflite"
VECTOR_METADATA="$INSTALL_DIR/artifacts/real-ppg-v2/model-metadata.json"

for required in "$PYTHON" "$MODEL" "$METADATA" "$VECTOR_MODEL" "$VECTOR_METADATA"; do
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
if health.get("vector_model_loaded") is not True:
    reason = health.get("vector_model_reason")
    raise SystemExit(f"error: vector model did not load: {reason}")
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
VECTOR_EXPECTED_SHA=$("$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["model_sha256"])' \
    "$VECTOR_METADATA")
VECTOR_ACTUAL_SHA=$("$PYTHON" -c \
    'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
    "$VECTOR_MODEL")
if [ "$VECTOR_EXPECTED_SHA" != "$VECTOR_ACTUAL_SHA" ]; then
    echo "error: installed vector model checksum does not match its metadata" >&2
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
VECTOR_PREDICTION=$(curl --fail --silent --show-error \
    -X POST "$BASE_URL/v3/intervals" \
    -H 'Content-Type: application/json' \
    --data-binary "@$INSTALL_DIR/examples/vector-interval.json")
VECTOR_DECISION=$("$PYTHON" -c \
    'import json,sys; print(json.loads(sys.argv[1]).get("decision", ""))' \
    "$VECTOR_PREDICTION")
if [ "$VECTOR_DECISION" != "normal" ] && [ "$VECTOR_DECISION" != "anomaly" ]; then
    echo "error: V2 sample request did not return a binary prediction" >&2
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
echo "vector model: real-ppg-vector-autoencoder-v2"
echo "vector model checksum: verified"
echo "vector sample prediction: $VECTOR_DECISION"
systemctl show "$SERVICE_NAME" \
    -p MemoryCurrent -p MemoryPeak -p MemoryMax -p TasksCurrent
