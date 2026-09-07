# Home Gateway Anomaly Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Raspberry Pi home gateway that accepts wearable feature packets, calibrates a personal baseline, stores a rolling history, runs long-term anomaly inference and returns a compact binary decision.

**Architecture:** The external wearable sends versioned feature packets; this repository begins at packet ingestion. The gateway validates and stores packets in SQLite, derives a robust 48-hour personal baseline, constructs 288-step daily sequences, combines the wearable flag with gateway fallback or 1D-autoencoder output, and exposes local JSON endpoints.

**Tech Stack:** Python 3.10+ standard library for runtime and tests, SQLite for durable state, optional TensorFlow for off-device Conv1D-autoencoder training, and a TensorFlow Lite interpreter for model inference.

**Spec:** `docs/superpowers/specs/2026-09-07-two-tier-anomaly-monitor-design.md`

## Global Constraints

- Implement the home gateway only; wearable firmware is outside this repository.
- Final decisions are exactly `normal` or `anomaly`.
- Heart rate is a feature derived by the wearable from its optical signal.
- SMS, cellular networking and notification delivery are excluded.
- Prediction responses contain results and reasons without repeated disclaimer text.
- Runtime must remain suitable for a 512 MB Raspberry Pi Zero 2 W.
- Health datasets and generated participant windows remain outside Git.
- Every production behavior is introduced test-first and committed separately.

---

### Task 1: Versioned wearable packet contract

**Files:**
- Create: `src/home_health_monitor/gateway/__init__.py`
- Create: `src/home_health_monitor/gateway/contracts.py`
- Create: `tests/test_gateway_contracts.py`

**Interfaces:**
- Consumes: JSON-compatible objects received from the external wearable.
- Produces: `parse_packet(payload: object) -> FeaturePacket` and immutable `FeaturePacket`, `Quality`, and `WearableAssessment` values.

- [ ] **Step 1: Write failing contract tests**

```python
def test_parse_packet_accepts_complete_v1_packet(self):
    packet = parse_packet(valid_packet())
    self.assertEqual("subject-1", packet.subject_id)
    self.assertEqual(97.2, packet.spo2_percent)
    self.assertEqual("normal", packet.wearable.decision)

def test_parse_packet_rejects_unknown_fields(self):
    payload = valid_packet()
    payload["sms"] = True
    with self.assertRaisesRegex(InputError, "unknown packet fields"):
        parse_packet(payload)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `PYTHONPATH=src python3 -m unittest tests.test_gateway_contracts -v`

Expected: import failure because `home_health_monitor.gateway.contracts` does not exist.

- [ ] **Step 3: Implement the strict schema**

```python
@dataclass(frozen=True, slots=True)
class FeaturePacket:
    schema_version: int
    subject_id: str
    device_id: str
    sequence: int
    timestamp: datetime
    sample_duration_seconds: float
    heart_rate_bpm: float
    spo2_percent: float
    temperature_c: float
    temperature_type: str
    temperature_site: str
    motion_intensity: float
    motion: int
    quality: Quality
    wearable: WearableAssessment
```

Validate finite numbers, UTC-aware timestamps, identifiers, sequence values,
`normal|anomaly`, motion `0|1`, SpO2 `0..100`, and bounded sensor values. Reject
unknown fields and unknown schema versions.

- [ ] **Step 4: Run contract tests and the full suite**

Run: `PYTHONPATH=src python3 -m unittest tests.test_gateway_contracts -v`

Run: `PYTHONPATH=src python3 -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit and push**

```bash
git add src/home_health_monitor/gateway tests/test_gateway_contracts.py
git commit -m "feat: define gateway packet contract"
git push origin main
```

### Task 2: Durable idempotent packet store

**Files:**
- Create: `src/home_health_monitor/gateway/store.py`
- Create: `tests/test_gateway_store.py`

**Interfaces:**
- Consumes: validated `FeaturePacket` objects.
- Produces: `GatewayStore`, with `add_packet`, `packets_between`, `latest_packet`, `save_calibration`, `load_calibration`, `save_event`, and `recent_events`.

- [ ] **Step 1: Write failing persistence tests**

```python
def test_duplicate_device_sequence_is_idempotent(self):
    with GatewayStore(self.path) as store:
        self.assertTrue(store.add_packet(packet(sequence=4)))
        self.assertFalse(store.add_packet(packet(sequence=4)))
        self.assertEqual(1, len(store.packets_between("subject-1", START, END)))

def test_state_survives_reopen(self):
    with GatewayStore(self.path) as store:
        store.add_packet(packet(sequence=1))
    with GatewayStore(self.path) as reopened:
        self.assertEqual(1, reopened.latest_packet("subject-1").sequence)
```

- [ ] **Step 2: Run the store tests and verify RED**

Run: `PYTHONPATH=src python3 -m unittest tests.test_gateway_store -v`

Expected: import failure because `gateway.store` does not exist.

- [ ] **Step 3: Implement SQLite storage**

Use `sqlite3`, WAL mode, foreign-key enforcement and a unique key on
`(device_id, sequence)`. Store normalized columns instead of request JSON.
Transactions commit packet, calibration and event updates atomically. Add an
indexed `(subject_id, timestamp)` lookup and an explicit retention method.

- [ ] **Step 4: Run store tests and the full suite**

Run: `PYTHONPATH=src python3 -m unittest tests.test_gateway_store -v`

Run: `PYTHONPATH=src python3 -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit and push**

```bash
git add src/home_health_monitor/gateway/store.py tests/test_gateway_store.py
git commit -m "feat: persist gateway packets"
git push origin main
```

### Task 3: Forty-eight-hour robust calibration

**Files:**
- Create: `src/home_health_monitor/gateway/calibration.py`
- Create: `tests/test_gateway_calibration.py`

**Interfaces:**
- Consumes: chronological `FeaturePacket` values for one subject.
- Produces: `build_calibration(packets, expected_interval_seconds=60) -> CalibrationResult` and `robust_deviations(packet, profile) -> dict[str, float]`.

- [ ] **Step 1: Write failing calibration tests**

```python
def test_calibration_requires_48_hours_and_80_percent_coverage(self):
    result = build_calibration(packets_for(hours=48, coverage=0.79))
    self.assertEqual("collecting", result.status)

def test_calibration_uses_median_and_mad(self):
    result = build_calibration(packets_for(hours=49, coverage=0.9, outlier=True))
    self.assertEqual("ready", result.status)
    self.assertAlmostEqual(97.0, result.profile.signals["spo2_percent"].median)
```

- [ ] **Step 2: Run calibration tests and verify RED**

Run: `PYTHONPATH=src python3 -m unittest tests.test_gateway_calibration -v`

Expected: import failure because `gateway.calibration` does not exist.

- [ ] **Step 3: Implement robust calibration**

Calculate valid coverage against the expected cadence, require 48 elapsed hours,
80 percent valid packets and eight low-motion hours, and calculate median plus
scaled MAD for heart rate, SpO2, temperature and motion intensity. Apply explicit
scale floors so constant normal signals remain usable.

- [ ] **Step 4: Run calibration tests and the full suite**

Run: `PYTHONPATH=src python3 -m unittest tests.test_gateway_calibration -v`

Run: `PYTHONPATH=src python3 -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit and push**

```bash
git add src/home_health_monitor/gateway/calibration.py tests/test_gateway_calibration.py
git commit -m "feat: calibrate personal gateway baseline"
git push origin main
```

### Task 4: Ordered 24-hour model windows

**Files:**
- Create: `src/home_health_monitor/gateway/windowing.py`
- Create: `tests/test_gateway_windowing.py`

**Interfaces:**
- Consumes: packets, calibration profile and model feature names.
- Produces: `build_window(packets, profile, end, step_minutes=5) -> ModelWindow` containing exactly 288 ordered vectors and masks.

- [ ] **Step 1: Write failing window tests**

```python
def test_window_has_288_five_minute_steps(self):
    window = build_window(self.packets, self.profile, self.end)
    self.assertEqual(288, len(window.values))

def test_missing_bins_have_zero_values_and_zero_mask(self):
    window = build_window(self.packets_without_hour_3, self.profile, self.end)
    self.assertEqual(0.0, window.values[36][0])
    self.assertEqual(0.0, window.masks[36][0])
```

- [ ] **Step 2: Run window tests and verify RED**

Run: `PYTHONPATH=src python3 -m unittest tests.test_gateway_windowing -v`

Expected: import failure because `gateway.windowing` does not exist.

- [ ] **Step 3: Implement fixed window construction**

Aggregate accepted packets into UTC-aligned five-minute bins using medians. Use
robust baseline normalization. Append quality and missingness features, represent
missing values as zero after normalization and set their mask to zero. Never
forward-fill.

- [ ] **Step 4: Run window tests and the full suite**

Run: `PYTHONPATH=src python3 -m unittest tests.test_gateway_windowing -v`

Run: `PYTHONPATH=src python3 -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit and push**

```bash
git add src/home_health_monitor/gateway/windowing.py tests/test_gateway_windowing.py
git commit -m "feat: build gateway model windows"
git push origin main
```

### Task 5: Autoencoder artifact and inference adapter

**Files:**
- Create: `src/home_health_monitor/gateway/autoencoder.py`
- Create: `tests/test_gateway_autoencoder.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `model.tflite`, `model-metadata.json`, and `ModelWindow`.
- Produces: `AutoencoderModel.load(...)`, `predict(window) -> ReconstructionResult`, and injectable `InterpreterProtocol` for runtime-independent tests.

- [ ] **Step 1: Write failing artifact tests**

```python
def test_reconstruction_result_reports_feature_errors(self):
    model = AutoencoderModel.from_interpreter(metadata(), IdentityInterpreter())
    result = model.predict(non_identity_window())
    self.assertGreater(result.overall_error, 0)
    self.assertEqual(set(FEATURES), set(result.feature_errors))

def test_loader_rejects_feature_manifest_mismatch(self):
    with self.assertRaisesRegex(ModelError, "feature manifest"):
        AutoencoderModel.load(self.model_path, mismatched_metadata_path)
```

- [ ] **Step 2: Run model tests and verify RED**

Run: `PYTHONPATH=src python3 -m unittest tests.test_gateway_autoencoder -v`

Expected: import failure because `gateway.autoencoder` does not exist.

- [ ] **Step 3: Implement the model adapter**

Validate schema, model checksum, 288-step input shape, feature order, quantization
parameters and thresholds. Resolve `tflite_runtime.interpreter.Interpreter`
first and `tensorflow.lite.Interpreter` second. Keep both optional so gateway
fallback operation remains dependency-free.

- [ ] **Step 4: Run model tests and the full suite**

Run: `PYTHONPATH=src python3 -m unittest tests.test_gateway_autoencoder -v`

Run: `PYTHONPATH=src python3 -m unittest discover -s tests -v`

Expected: all tests pass without TensorFlow installed.

- [ ] **Step 5: Commit and push**

```bash
git add pyproject.toml src/home_health_monitor/gateway/autoencoder.py tests/test_gateway_autoencoder.py
git commit -m "feat: run gateway autoencoder artifacts"
git push origin main
```

### Task 6: Binary two-tier decision engine

**Files:**
- Create: `src/home_health_monitor/gateway/engine.py`
- Create: `tests/test_gateway_engine.py`

**Interfaces:**
- Consumes: packet, store, calibration result and optional autoencoder model.
- Produces: `GatewayEngine.ingest(packet) -> GatewayDecision` with compact `to_dict()` output.

- [ ] **Step 1: Write failing decision tests**

```python
def test_wearable_anomaly_wins_immediately(self):
    decision = self.engine.ingest(packet(wearable_decision="anomaly"))
    self.assertEqual("anomaly", decision.decision)
    self.assertEqual(["wearable"], decision.triggered_by)

def test_normal_result_has_no_disclaimer(self):
    body = self.engine.ingest(packet()).to_dict()
    self.assertEqual("normal", body["decision"])
    self.assertNotIn("disclaimer", body)
```

- [ ] **Step 2: Run engine tests and verify RED**

Run: `PYTHONPATH=src python3 -m unittest tests.test_gateway_engine -v`

Expected: import failure because `gateway.engine` does not exist.

- [ ] **Step 3: Implement combination and fallback**

Apply `wearable OR gateway_model OR persistent_data_failure`. Before model
readiness, use robust personal deviations when calibration is ready. Include
only decision, trigger sources, reason codes, scores, contributing signals,
current measurements, calibration status and timestamp.

- [ ] **Step 4: Run engine tests and the full suite**

Run: `PYTHONPATH=src python3 -m unittest tests.test_gateway_engine -v`

Run: `PYTHONPATH=src python3 -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit and push**

```bash
git add src/home_health_monitor/gateway/engine.py tests/test_gateway_engine.py
git commit -m "feat: combine gateway anomaly tiers"
git push origin main
```

### Task 7: Gateway HTTP service

**Files:**
- Create: `src/home_health_monitor/gateway/server.py`
- Modify: `src/home_health_monitor/__main__.py`
- Modify: `deploy/home-health-monitor.service`
- Create: `tests/test_gateway_server.py`

**Interfaces:**
- Consumes: JSON packets through `POST /v2/packets`.
- Produces: compact decision response, `GET /v2/prediction`, `GET /v2/calibration`, and `GET /health`.

- [ ] **Step 1: Write failing endpoint tests**

```python
def test_packet_endpoint_returns_binary_prediction(self):
    status, body = request_json(self.server, "POST", "/v2/packets", valid_packet())
    self.assertEqual(200, status)
    self.assertIn(body["decision"], {"normal", "anomaly"})
    self.assertNotIn("disclaimer", body)

def test_health_reports_gateway_components(self):
    status, body = request_json(self.server, "GET", "/health")
    self.assertEqual("ready", body["status"])
    self.assertIn("model_loaded", body)
```

- [ ] **Step 2: Run server tests and verify RED**

Run: `PYTHONPATH=src python3 -m unittest tests.test_gateway_server -v`

Expected: import failure because `gateway.server` does not exist.

- [ ] **Step 3: Implement the local service**

Reuse the bounded standard-library HTTP approach, maximum body size and
localhost binding. Add explicit database and model paths. Do not add SMS,
outbound HTTP or user-facing notification code.

- [ ] **Step 4: Run server tests and the full suite**

Run: `PYTHONPATH=src python3 -m unittest tests.test_gateway_server -v`

Run: `PYTHONPATH=src python3 -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit and push**

```bash
git add deploy/home-health-monitor.service src/home_health_monitor/__main__.py src/home_health_monitor/gateway/server.py tests/test_gateway_server.py
git commit -m "feat: expose home gateway predictions"
git push origin main
```

### Task 8: Normal-only Conv1D training and export

**Files:**
- Create: `src/home_health_monitor/gateway/training.py`
- Create: `tests/test_gateway_training.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: JSON Lines normal windows grouped by subject and feature-manifest version.
- Produces: integer-quantized `model.tflite`, `model-metadata.json`, participant-held-out metrics and checksums.

- [ ] **Step 1: Write failing training-data tests**

```python
def test_split_keeps_subjects_in_one_partition(self):
    split = grouped_split(rows_for_subjects(12), seed=42)
    self.assertFalse(set(split.train_subjects) & set(split.test_subjects))

def test_loader_rejects_non_normal_training_rows(self):
    with self.assertRaisesRegex(TrainingDataError, "normal-only"):
        load_windows(path_with_decision("anomaly"))
```

- [ ] **Step 2: Run training tests and verify RED**

Run: `PYTHONPATH=src python3 -m unittest tests.test_gateway_training -v`

Expected: import failure because `gateway.training` does not exist.

- [ ] **Step 3: Implement inspectable data loading and grouped splitting**

Keep TensorFlow imports inside the train function. Validate 288-step shapes,
finite values, masks, normal-only labels, distinct subjects and feature-manifest
identity before constructing train, validation and test groups.

- [ ] **Step 4: Implement and export the small model**

Use an encoder with two small `Conv1D` layers, strided downsampling and a mirrored
decoder. Optimize reconstruction mean-squared error over observed values only.
Select persistence thresholds from held-out normal participants, convert with a
representative dataset to integer TensorFlow Lite, and write complete metadata
plus SHA-256.

- [ ] **Step 5: Run data-path tests and an optional TensorFlow smoke test**

Run: `PYTHONPATH=src python3 -m unittest tests.test_gateway_training -v`

Run when TensorFlow is installed: `PYTHONPATH=src python3 -m home_health_monitor.gateway.training --smoke-test`

Run: `PYTHONPATH=src python3 -m unittest discover -s tests -v`

Expected: all standard tests pass without TensorFlow; the smoke test exports and
reloads a tiny TFLite artifact when training dependencies are installed.

- [ ] **Step 6: Commit and push**

```bash
git add pyproject.toml src/home_health_monitor/gateway/training.py tests/test_gateway_training.py
git commit -m "feat: train gateway anomaly autoencoder"
git push origin main
```

### Task 9: Dataset audit and conversion tools

**Files:**
- Create: `src/home_health_monitor/datasets/__init__.py`
- Create: `src/home_health_monitor/datasets/audit.py`
- Create: `src/home_health_monitor/datasets/galaxyppg.py`
- Create: `src/home_health_monitor/datasets/synthetic.py`
- Create: `tests/test_dataset_tools.py`

**Interfaces:**
- Consumes: user-supplied GalaxyPPG directory or corrected synthetic CSV path.
- Produces: quality report JSON and research packets/windows outside Git.

- [ ] **Step 1: Write failing audit tests**

```python
def test_audit_detects_repeated_synthetic_cycle(self):
    report = audit_synthetic(self.duplicate_csv)
    self.assertEqual(36, report.unique_rows)
    self.assertFalse(report.training_eligible)

def test_galaxy_converter_never_labels_illness(self):
    records = list(convert_galaxy(self.fixture_root))
    self.assertNotIn("decision", records[0])
    self.assertEqual("engineering_only", records[0]["dataset_role"])
```

- [ ] **Step 2: Run dataset tests and verify RED**

Run: `PYTHONPATH=src python3 -m unittest tests.test_dataset_tools -v`

Expected: import failure because `home_health_monitor.datasets` does not exist.

- [ ] **Step 3: Implement audits and converters**

Audit duplicates, timestamp order, missing streams, sensor status and participant
coverage. Galaxy conversion derives engineering heart-rate and motion examples
but cannot emit final SpO2 model windows. Synthetic conversion requires the
corrected filename and marks every row `synthetic_demo`.

- [ ] **Step 4: Run dataset tests and full suite**

Run: `PYTHONPATH=src python3 -m unittest tests.test_dataset_tools -v`

Run: `PYTHONPATH=src python3 -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Commit and push**

```bash
git add src/home_health_monitor/datasets tests/test_dataset_tools.py
git commit -m "feat: audit anomaly research datasets"
git push origin main
```

### Task 10: Gateway documentation and Pi verification

**Files:**
- Modify: `README.md`
- Modify: `docs/api.md`
- Modify: `docs/data-sources.md`
- Modify: `docs/design.md`
- Modify: `docs/pi-deployment.md`
- Create: `docs/training.md`

**Interfaces:**
- Consumes: completed gateway behavior and verified commands.
- Produces: one consistent explanation of setup, training, operation and limits.

- [ ] **Step 1: Update all documentation around the implemented gateway**

Document the packet schema, 48-hour calibration, binary decision, model fallback,
training data format, GalaxyPPG's engineering-only role, synthetic-data warning,
SQLite paths, model installation and local endpoints. Remove the obsolete
healthy/sick logistic-model narrative and repeated response disclaimers.

- [ ] **Step 2: Run documentation and repository checks**

Run: `rg -n "SMS|current_unhealthy|unhealthy_within_horizon|diagnosis|disclaimer" README.md docs src tests`

Expected: SMS appears only in explicit out-of-scope documentation; obsolete
supervised labels and response disclaimer fields do not appear in active code or
API examples.

Run: `git diff --check`

Run: `PYTHONPATH=src python3 -m unittest discover -s tests -v`

Expected: checks and all tests pass.

- [ ] **Step 3: Verify the service on the target Pi**

Run the documented install, reboot, calibration-resume, model-missing fallback,
model-loaded, duplicate-packet and 24-hour-window scenarios. Record peak RSS and
inference latency; release only below 96 MB RSS and two seconds per inference.

- [ ] **Step 4: Commit and push documentation**

```bash
git add README.md docs
git commit -m "docs: explain home gateway operation"
git push origin main
```
