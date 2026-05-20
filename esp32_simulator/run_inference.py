#!/usr/bin/env python3
"""
Simple runner that reads telemetry CSV lines from a serial port (or PTY)
and runs predictions using `telemetry_inference.py`.

Expected CSV format per line (from GUI):
  co,humidity,lpg,temp,smoke,light,motion

Example:
  python3 run.py /tmp/ttyV1 --model saved_telemetry_models/telemetry_mlp.pth

If you are using the GUI in mock mode connected to a PTY pair created by socat,
point the GUI at /tmp/ttyV0 and run this script on /tmp/ttyV1.
" socat -d -d pty,raw,echo=0,link=/tmp/ttyV0 pty,raw,echo=0,link=/tmp/ttyV1 "
"""

import argparse
import serial
import time
import sys
import numpy as np
from esp32_simulator import telemetry_inference as ti

CLASS_NAMES = [
    "normal_unoccupied",
    "normal_occupied",
    "warning_uncomfortable",
    "warning_poor_air",
    "critical_gas_leak",
    "critical_fire",
]


def main():
    parser = argparse.ArgumentParser(description="Read telemetry CSV from serial and run model predictions")
    parser.add_argument("port", help="Serial device (e.g. /dev/ttyUSB0 or /tmp/ttyV1)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--model", required=True, help="Path to PyTorch model .pth")
    parser.add_argument("--scaler", default=None, help="Optional scaler joblib file")
    parser.add_argument("--print-probs", action="store_true", help="Print full probability vector")
    args = parser.parse_args()

    # load model and scaler
    try:
        model = ti.load_model(args.model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    try:
        scaler = ti.load_scaler(args.scaler)
    except Exception as e:
        print(f"Scaler load/compute failed, continuing without scaler: {e}")
        scaler = None

    # open serial
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"Opened serial {args.port} @ {args.baud}")
    except Exception as e:
        print(f"Failed to open serial port {args.port}: {e}")
        sys.exit(1)

    try:
        while True:
            try:
                line = ser.readline().decode(errors='ignore').strip()
            except Exception as e:
                print(f"Serial read error: {e}")
                time.sleep(0.1)
                continue

            if not line:
                continue

            # allow lines with prefix or labels; look for first CSV-like component
            parts = [p.strip() for p in line.split(',') if p.strip() != '']
            if len(parts) < 7:
                # not a telemetry line
                print("IGNORED:", line)
                continue

            try:
                # parse according to expected order
                co = float(parts[0])
                humidity = float(parts[1])
                lpg = float(parts[2])
                temp = float(parts[3])
                smoke = float(parts[4])
                light = int(float(parts[5]))
                motion = int(float(parts[6]))

                sample = np.array([[co, humidity, lpg, temp, smoke, light, motion]], dtype=np.float32)

                probs, pred = ti.predict(model, scaler, sample)
                label = CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else str(pred)

                ts = time.strftime('%Y-%m-%d %H:%M:%S')
                if args.print_probs:
                    print(f"{ts}  -> {label}  p={probs[pred]:.3f}  probs={np.round(probs,3)}")
                else:
                    print(f"{ts}  -> {label}  p={probs[pred]:.3f}")

            except Exception as e:
                print(f"PARSE/PREDICT ERROR for line '{line}': {e}")

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()