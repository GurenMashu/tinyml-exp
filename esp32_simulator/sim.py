try:
    import serial
except ImportError:
    serial = None

import random
import time
import tkinter as tk
from tkinter import ttk, messagebox
import argparse
import threading
import sys
import os


class SensorControlPanel:

    def __init__(self, port=None, baudrate=115200, use_serial=False):

        self.running = True
        self.ser = None
        self.port = port
        self.baudrate = baudrate
        self.use_serial = use_serial

        self.sim_interval_ms = 1000
        self.generator_running = False
        self.log_to_file = False
        self.log_path = os.path.join(os.getcwd(), "esp32_simulated_data.csv")

        # Telemetry state for standalone generation
        self.state = {
            "co": 0.002,
            "humidity": 55.0,
            "lpg": 0.002,
            "temp": 24.0,
            "smoke": 0.01,
            "light": 0,
            "motion": 0,
        }

        # Serial Connection (only if requested)
        if self.use_serial and serial is not None:
            self.connect_serial()

        if not self.use_serial:
            print("[INFO] Running in standalone generator mode")

        # GUI Window
        self.root = tk.Tk()
        self.root.title("ESP32 TinyML Sensor Simulator")
        self.root.geometry("520x520")
        self.root.configure(padx=20, pady=20)

        # Variables
        # telemetry features expected by the MLP model:
        # ["co", "humidity", "lpg", "temp", "smoke", "light", "motion"]
        self.co_var = tk.DoubleVar(value=self.state["co"])
        self.humidity_var = tk.DoubleVar(value=self.state["humidity"])
        self.lpg_var = tk.DoubleVar(value=self.state["lpg"])
        self.temp_var = tk.DoubleVar(value=self.state["temp"])
        self.smoke_var = tk.DoubleVar(value=self.state["smoke"])
        self.light_var = tk.IntVar(value=self.state["light"])
        self.motion_var = tk.IntVar(value=self.state["motion"])

        # no model logic here; this GUI only generates telemetry

        # Create UI
        self.create_widgets()

        # Read Thread (only if serial enabled)
        if self.use_serial:
            self.read_thread = threading.Thread(
                target=self.read_serial,
                daemon=True
            )
            self.read_thread.start()

        # Generator Thread
        self.generator_thread = threading.Thread(
            target=self._generator_loop,
            daemon=True
        )
        self.generator_thread.start()

    # --------------------------------------------------
    # Serial Connection
    # --------------------------------------------------
    def connect_serial(self):

        try:

            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1
            )

            # STM32 reset delay
            time.sleep(2)

            print(
                f"[INFO] Connected to "
                f"{self.port} at {self.baudrate}"
            )

        except serial.SerialException as e:

            print(f"[SERIAL ERROR] {e}")
            self.running = False

        except PermissionError:

            print(
                "[ERROR] Permission denied.\n"
                "Try:\n"
                "sudo chmod 666 /dev/ttyACM0"
            )

            self.running = False

        except Exception as e:

            print(f"[UNKNOWN ERROR] {e}")
            self.running = False

    # --------------------------------------------------
    # GUI Widgets
    # --------------------------------------------------
    def create_widgets(self):

        def label_for(val, fmt):
            return fmt.format(val)

        # helper to add a labeled slider
        def make_slider(parent, text, var, frm, to, fmt):
            ttk.Label(parent, text=text).pack(anchor='w')
            lbl = ttk.Label(parent, text=fmt.format(var.get()))
            lbl.pack(anchor='e')
            def onmove(e):
                try:
                    lbl.config(text=fmt.format(var.get()))
                except Exception:
                    pass
                self.update_values()
            s = ttk.Scale(parent, from_=frm, to=to, variable=var, command=onmove)
            s.pack(fill='x', pady=5)
            return lbl

        # expose helper as instance method for later use
        self._add_slider = lambda text, var, frm, to, fmt='{:.2f}': make_slider(self.root, text, var, frm, to, fmt)
        self._add_control_buttons = lambda: self._create_control_buttons()


        # Feature controls
        self._add_slider("CO (approx)", self.co_var, 0.0, 0.02, fmt="{:.4f}")
        self._add_slider("LPG (approx)", self.lpg_var, 0.0, 0.02, fmt="{:.4f}")
        self._add_slider("Smoke (approx)", self.smoke_var, 0.0, 0.08, fmt="{:.4f}")
        self._add_slider("Temperature (°C)", self.temp_var, 10.0, 40.0, fmt="{:.1f}")
        self._add_slider("Humidity (%)", self.humidity_var, 10.0, 100.0, fmt="{:.1f}")

        ttk.Checkbutton(
            self.root,
            text="Light (on/off)",
            variable=self.light_var,
            command=self.update_values
        ).pack(anchor="w", pady=6)

        ttk.Checkbutton(
            self.root,
            text="Motion Detected",
            variable=self.motion_var,
            command=self.update_values
        ).pack(anchor="w", pady=6)

        # Generator controls
        self._add_control_buttons()

        # Generated output preview
        ttk.Label(self.root, text="Latest generated sample:").pack(anchor="w", pady=(15, 0))
        self.output_text = tk.Text(self.root, height=5, width=60, state="disabled", wrap="none")
        self.output_text.pack(fill="x", pady=5)

        # Status Label
        self.status_label = ttk.Label(
            self.root,
            text="Ready",
            foreground="green"
        )

        self.status_label.pack(side="bottom", pady=10)

    # --------------------------------------------------
    # Update Slider Labels
    # --------------------------------------------------
    def update_values(self, event=None):
        # labels are updated by slider callbacks; nothing else required here
        return

    # --------------------------------------------------
    # Control button helpers
    # --------------------------------------------------
    def _create_control_buttons(self):
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.start_btn = ttk.Button(btn_frame, text="Start", command=self.start_generator)
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_generator)
        self.stop_btn.pack(side="left", padx=5)

        self.log_var = tk.IntVar(value=0)
        ttk.Checkbutton(
            btn_frame,
            text="Log to CSV",
            variable=self.log_var,
            command=self._toggle_log
        ).pack(side="left", padx=5)

        ttk.Button(
            btn_frame,
            text="Reconnect",
            command=self.reconnect_serial
        ).pack(side="left", padx=5)

    def start_generator(self):
        if not self.generator_running:
            self.generator_running = True
            self.update_status("Generator started", "green")

    def stop_generator(self):
        if self.generator_running:
            self.generator_running = False
            self.update_status("Generator stopped", "orange")

    def _toggle_log(self):
        self.log_to_file = bool(self.log_var.get())
        self.update_status("CSV logging enabled" if self.log_to_file else "CSV logging disabled", "blue")

    def _append_output(self, text):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.configure(state="disabled")

    # --------------------------------------------------
    # Data generation
    # --------------------------------------------------
    def generate_sample(self):
        # smooth updates with random drift
        def drift(value, scale, min_v, max_v):
            value += (random.random() - 0.5) * scale
            value = max(min_v, min(max_v, value))
            return value

        # periodic cycles and noise
        self.state["co"] = drift(self.state["co"], 0.0005, 0.0001, 0.02)
        self.state["lpg"] = drift(self.state["lpg"], 0.0005, 0.0001, 0.02)
        self.state["smoke"] = drift(self.state["smoke"], 0.001, 0.0, 0.08)
        self.state["temp"] = drift(self.state["temp"], 0.4, 10.0, 40.0)
        self.state["humidity"] = drift(self.state["humidity"], 2.0, 10.0, 100.0)

        # random occupancy events
        if random.random() < 0.08:
            self.state["motion"] = 1
        elif random.random() < 0.05:
            self.state["motion"] = 0

        # light correlates with motion and some random false positives
        if self.state["motion"] == 1 or random.random() < 0.1:
            self.state["light"] = 1
        else:
            self.state["light"] = 0

        # occasional spikes for gas/smoke
        if random.random() < 0.05:
            self.state["smoke"] = min(0.08, self.state["smoke"] + random.uniform(0.01, 0.03))
            self.state["lpg"] = min(0.02, self.state["lpg"] + random.uniform(0.001, 0.005))

        # update sliders to reflect generated data
        self.co_var.set(self.state["co"])
        self.lpg_var.set(self.state["lpg"])
        self.smoke_var.set(self.state["smoke"])
        self.temp_var.set(self.state["temp"])
        self.humidity_var.set(self.state["humidity"])
        self.light_var.set(self.state["light"])
        self.motion_var.set(self.state["motion"])

        csv_line = f"{self.state['co']:.4f},{self.state['humidity']:.1f},{self.state['lpg']:.4f},{self.state['temp']:.1f},{self.state['smoke']:.4f},{self.state['light']},{self.state['motion']}"
        return csv_line

    def _generator_loop(self):
        while self.running:
            if self.generator_running:
                line = self.generate_sample()
                if self.use_serial and self.ser and self.ser.is_open:
                    try:
                        self.ser.write((line + "\n").encode("utf-8"))
                    except Exception as e:
                        self.update_status(f"Serial write failed: {e}", "red")
                else:
                    print("TX:", line)

                self._append_output(line)
                if self.log_to_file:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                time.sleep(self.sim_interval_ms / 1000.0)
            else:
                time.sleep(0.1)
    def send_data(self):

        # Build telemetry row matching model features: co,humidity,lpg,temp,smoke,light,motion
        co = float(self.co_var.get())
        lpg = float(self.lpg_var.get())
        smoke = float(self.smoke_var.get())
        temp = float(self.temp_var.get())
        humidity = float(self.humidity_var.get())
        light = int(self.light_var.get())
        motion = int(self.motion_var.get())

        data = f"{co:.4f},{humidity:.1f},{lpg:.4f},{temp:.1f},{smoke:.4f},{light},{motion}\n"

        # If serial is enabled, write; otherwise just print
        if self.use_serial and self.ser and self.ser.is_open:
            try:
                self.ser.write(data.encode("utf-8"))
            except Exception as e:
                self.update_status(f"Serial write failed: {e}", "red")
                print(f"[SERIAL WRITE ERROR] {e}")

        self.update_status(f"Sent: {data.strip()}", "blue")
        print("TX:", data.strip())

    # --------------------------------------------------
    # Read STM32 Messages
    # --------------------------------------------------
    def read_serial(self):

        while self.running:

            try:

                if (
                    self.ser and
                    self.ser.is_open and
                    self.ser.in_waiting > 0
                ):

                    line = self.ser.readline() \
                        .decode(
                            "utf-8",
                            errors="ignore"
                        ) \
                        .strip()

                    if line:
                        print("STM32:", line)

            except serial.SerialException as e:

                print(f"[READ SERIAL ERROR] {e}")

                self.update_status(
                    "Serial disconnected",
                    "red"
                )

            except UnicodeDecodeError as e:

                print(f"[DECODE ERROR] {e}")

            except Exception as e:

                print(f"[READ ERROR] {e}")

            time.sleep(0.05)

    # --------------------------------------------------
    # Reconnect Serial
    # --------------------------------------------------
    def reconnect_serial(self):

        try:

            if self.ser and self.ser.is_open:
                self.ser.close()

            self.connect_serial()

            if self.ser and self.ser.is_open:

                self.update_status(
                    "Reconnected successfully",
                    "green"
                )

        except Exception as e:

            self.update_status(
                f"Reconnect failed: {e}",
                "red"
            )

    # ------------------- model/scaler helpers -------------------
    # model/scaler/prediction are intentionally implemented in a separate module

    # --------------------------------------------------
    # Update Status Label
    # --------------------------------------------------
    def update_status(self, message, color="black"):

        try:

            self.status_label.config(
                text=message,
                foreground=color
            )

        except Exception:
            pass

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    def on_closing(self):

        self.running = False

        try:

            if self.ser and self.ser.is_open:
                self.ser.close()

        except Exception as e:

            print(f"[CLOSE ERROR] {e}")

        finally:

            print("[INFO] Application closed")

            self.root.destroy()


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":

    try:

        parser = argparse.ArgumentParser(
            description="STM32 TinyML Sensor Simulator"
        )

        parser.add_argument(
            "port",
            nargs="?",
            default=None,
            help="Optional serial port (Example: /dev/ttyACM0). Omit for standalone generation."
        )

        parser.add_argument(
            "--baud",
            type=int,
            default=115200,
            help="Baud Rate"
        )

        parser.add_argument(
            "--mock",
            action="store_true",
            help="Run standalone generator without serial"
        )

        args = parser.parse_args()

        use_serial = bool(args.port and not args.mock)
        app = SensorControlPanel(
            port=args.port,
            baudrate=args.baud,
            use_serial=use_serial
        )

        if app.running:

            app.root.protocol(
                "WM_DELETE_WINDOW",
                app.on_closing
            )

            app.root.mainloop()

        else:

            print("[ERROR] Failed to start application")
            sys.exit(1)

    except KeyboardInterrupt:

        print("\n[INFO] Interrupted by user")

    except Exception as e:

        print(f"[FATAL ERROR] {e}")
        sys.exit(1)