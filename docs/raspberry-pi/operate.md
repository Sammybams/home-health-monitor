# Operate the gateway

## Check everything

```sh
sudo /opt/home-health-monitor/deploy/verify-pi.sh
```

## Service status and logs

```sh
sudo systemctl status home-health-monitor --no-pager
sudo journalctl -u home-health-monitor -n 100 --no-pager
```

## Update from GitHub

```sh
sudo git -C /opt/home-health-monitor pull --ff-only origin main
sudo /opt/home-health-monitor/deploy/install-pi.sh
```

The installer can be run again. It updates the Python package, model, and
service, then verifies the running gateway.

## Reboot test

```sh
sudo reboot
```

After reconnecting:

```sh
sudo /opt/home-health-monitor/deploy/verify-pi.sh
```

## Replace the model later

Train on a separate computer. Install `model.tflite` and
`model-metadata.json` together in:

```text
/opt/home-health-monitor/artifacts/gateway/
```

Then run:

```sh
sudo systemctl restart home-health-monitor
sudo /opt/home-health-monitor/deploy/verify-pi.sh
```

The verifier rejects a mismatched checksum or a model that did not load.
