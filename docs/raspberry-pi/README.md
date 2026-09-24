# Raspberry Pi quick guide

This folder explains the home gateway in plain language.

1. [Install it](install.md)
2. [Connect the wearable and use predictions](use.md)
3. [Check, update, or troubleshoot it](operate.md)
4. [Understand the included model](model.md)

The Pi receives a JSON packet from the BLE integration, saves it, and always
returns either `normal` or `anomaly`. It does not train the model and it does
not send SMS messages.

Use a Raspberry Pi Zero 2 W with 512 MB RAM and **64-bit Raspberry Pi OS
Lite**. The installed V1 model is 18 KiB and the installed real-PPG V2 candidate
is 4.2 KiB. They use separate strict endpoints; V2 does not replace the complete
V1 contract until the BLE feature mapping is confirmed and the candidate is
benchmarked on the target Pi. TensorFlow is not installed on the Pi; inference
uses the smaller LiteRT runtime.
