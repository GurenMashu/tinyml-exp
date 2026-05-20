# Sensor Simulator

## Overview
- Small toolkit to generate telemetry data matching the MLP model features and run offline inference.
- Components:
  - `sim.py` — standalone Tkinter GUI that generates CSV lines in the order: `co,humidity,lpg,temp,smoke,light,motion`.
  - `telemetry_inference.py` — helpers: `load_model()`, `load_scaler()`, `predict()` for PyTorch inference.
  - `run_inference.py` — reads CSV lines from serial/PTY and runs continuous predictions using `telemetry_inference`.

### Standalone GUI mode
- `sim.py` can run without any serial or PTY connection.
- It generates varied telemetry samples internally and shows the latest line in the UI.
- Note: standalone mode only generates data. To run model inference on that generated data, you must also run `run_inference.py` on a matching serial/PTY endpoint or use the GUI in serial mode.

Run standalone:
```bash
python3 sim.py --mock
```

or simply:
```bash
python3 sim.py
```

### Serial / PTY mode
1. Create a PTY pair (Linux/macOS):

```bash
socat -d -d pty,raw,echo=0,link=/tmp/ttyV0 pty,raw,echo=0,link=/tmp/ttyV1
```

2. Start the GUI on one end:

```bash
python3 sim.py /tmp/ttyV0 --baud 115200
```

3. Run inference on the other end:

```bash
python3 run_inference.py /tmp/ttyV1 --model path/to/saved_model.pth --scaler path/to/scaler.pkl
```

### Real ESP32 mode
- If you have a physical ESP32 (or similar) connected, pass its serial port instead of a PTY path:

```bash
python3 gui.py /dev/ttyUSB0 --baud 115200
```

Notes
- `run_inference.py` requires a trained PyTorch model file (`.pth`).
- The GUI generates one sample per second by default, with smooth drift, occasional gas/smoke spikes, and motion/light events.

Files
- `sim.py` — standalone telemetry generator GUI
- `telemetry_inference.py` — model/scaler/predict helpers
- `run_inference.py` — serial consumer + inference runner

