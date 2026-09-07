# Implemented gateway design

## Scope

This code is only the home gateway. The wearable firmware, sensor drivers, BLE
GATT transport, and any downstream message or notification system are separate.
The gateway accepts the wearable's feature packet, maintains local history,
runs longer-term inference, and stores/returns the decision locally.

The output is exactly a binary physiological-pattern decision: `normal` or
`anomaly`. It does not select a disease name and does not require illness labels.

## Inputs

The physical measurements are:

1. optical PPG/SpO2;
2. temperature;
3. motion.

Heart rate is derived from the optical signal on the wearable, so the gateway's
fixed numerical signal set is:

```text
heart_rate_bpm
spo2_percent
temperature_c
motion_intensity
```

The packet also contains binary motion, one quality value per signal family,
device/firmware/configuration identity, and the wearable's immediate result.
Temperature type and placement are fixed during calibration because skin,
surface, and core readings are not interchangeable.

## End-to-end sequence

1. The wearable wakes and samples for 2-5 seconds.
2. It filters raw signals, derives heart rate and compact features, and evaluates
   current robust deviations from its personal baseline.
3. It sends one versioned feature packet over BLE and returns to sleep.
4. A hardware-specific bridge posts that packet to `POST /v2/packets` on the Pi.
5. The gateway validates and inserts it into SQLite. Repeated `(device_id,
   sequence)` values do not duplicate packet history.
6. The gateway updates or loads the subject's calibration.
7. When ready, it compares the current packet with the robust baseline.
8. When a model is loaded, it creates the 24-hour model tensor and runs int8
   inference.
9. It combines all evidence with an OR rule, stores the event, and returns JSON.
10. Packet and prediction-event history older than 30 days is deleted
    automatically; the small calibration profile remains.

## Forty-eight-hour calibration

Calibration does not accept a wall-clock wait alone. It becomes ready only when
all three gates pass:

- at least 48 elapsed hours;
- at least 80% of expected one-minute positions contain a valid normal packet;
- at least eight hours of valid low-motion packets.

A packet contributes only when the wearable marked it normal and all four
quality values are at least `0.8`. Heart rate, SpO2, and temperature baselines
use low-motion readings. Motion intensity uses all accepted readings.

For each signal, the gateway stores its median and scaled median absolute
deviation (MAD). Minimum scale floors prevent a constant calibration signal
from causing division by zero. A current signed robust deviation is:

```text
(current value - personal median) / personal MAD scale
```

An absolute gateway deviation of at least `4.0` contributes a personal-baseline
anomaly. These are versioned engineering defaults in the code.

## Twenty-four-hour model tensor

The prediction timeline has 288 UTC-aligned five-minute bins. Packets in a bin
are median-aggregated. There is no forward filling: a missing signal stays zero
after normalization and its mask stays zero.

Each time step has eight values:

```text
4 personal-normalized physiological values
4 raw quality values
```

Eight matching observed/missing masks are appended, giving the model an input
shape of `1 × 288 × 16`. Signal values below their `0.8` quality gate are masked,
while the quality measurement itself remains visible to the model.

## Tiny autoencoder

The normal-only model is a small temporal convolutional autoencoder:

```text
288×16 input
  -> Conv1D(12)
  -> strided Conv1D(8)
  -> strided Conv1D(4)
  -> upsample + Conv1D(8)
  -> upsample + Conv1D(12)
  -> 8 reconstructed values
```

Training masks missing positions out of reconstruction loss. Validation people
set the 99th-percentile persistent threshold. The severe threshold starts at no
less than twice that value. The selected model is converted with representative
target-device windows to full int8 LiteRT/TFLite.

At gateway startup, the loader verifies:

- metadata schema and feature-manifest identity;
- SHA-256 of the model file;
- exact input/output shape and feature order;
- int8 scale and zero point;
- persistent/severe thresholds.

The production release gates are model size below 1 MiB, gateway peak RSS below
96 MiB, and inference below two seconds on the exact Pi image. The exporter
enforces model size; memory and latency must be measured on the target Pi.

## Decision combiner

```text
anomaly = wearable_anomaly
       OR abs(personal_z) >= 4 for any signal
       OR severe_autoencoder_error
       OR two_consecutive_high_autoencoder_errors
       OR three_consecutive_low_quality_packets
```

The response identifies every active source and contributing signal. A missing
or invalid model never removes the other prediction paths. During calibration,
the wearable and sensor-quality paths still produce a result for every packet.

## Storage and resource behavior

SQLite uses write-ahead logging, an indexed subject/timestamp lookup, and a
unique device/sequence key. It stores normalized fields rather than opaque
packet JSON. Calibration and decision event payloads survive restart. The HTTP
server is the Python standard library server and has a bounded request size;
FastAPI and a general web framework are not required.

## What is needed for a field-ready model

The code and model format are ready, but the final weights and thresholds must
come from normal data recorded by the actual wearable configuration. That data
must include enough people from the target population, stable sensor placement,
firmware/feature-manifest identity, continuous day coverage, activity variety,
and reviewed normal periods. Evaluation reports false alerts per person/day,
activity-specific errors, missing-data behavior, subgroup behavior, Pi memory,
and Pi latency.
