#!/usr/bin/env python3
"""
Live plotter for Arduino dual sensor (A0/A1) output.

Runs a tiny built-in HTTP server (Python stdlib only) that serves a web page
using Plotly.js (loaded from CDN). The page polls a /data endpoint every
200 ms and updates the charts in real time. No matplotlib / dash required.

Usage:
    python plot_sensor_live.py --port /dev/ttyACM0 [--max-points 500] [--http-port 8050]
"""

import argparse
import json
import re
import sys
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

try:
    import serial
except ImportError:
    print("Error: Missing required package. Install with:")
    print("  pip install pyserial")
    sys.exit(1)


@dataclass
class SensorData:
    """Container for sensor readings."""
    timestamp: float
    channel: str  # "A0" or "A1"
    adc_live: float
    adc_mean10: float
    d_mean: float
    voltage_live: float
    voltage_mean10: float
    resistance: Optional[float]


# Regex patterns for parsing the Arduino output
ARDUINO_OUTPUT_PATTERN = re.compile(
    r"A(?P<channel>[01])\s+ADC_live:\s+(?P<adc_live>[\d.]+)\s+\|\s+"
    r"ADC_mean10:\s+(?P<adc_mean10>[\d.]+)\s+\|\s+"
    r"dMean:\s+(?P<d_mean>[-\d.]+)\s+\|\s+"
    r"U_live:\s+(?P<u_live>[\d.]+)\s+V\s+\|\s+"
    r"U_mean10:\s+(?P<u_mean10>[\d.]+)\s+V\s+\|\s+"
    r"R~\(mean\):\s+(?P<resistance>(?:\d+(?:\.\d+)?|unendlich / ausser Bereich))"
)


class SerialReader(threading.Thread):
    """Background thread for reading serial data."""

    def __init__(self, port: str, baudrate: int = 115200, max_points: int = 500):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.max_points = max_points
        self.data_a0 = deque(maxlen=max_points)
        self.data_a1 = deque(maxlen=max_points)
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.error = None

    def run(self):
        """Background thread main loop."""
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=1)
            # Mark connected as soon as the port opens; main() polls this flag.
            self.connected = True
            time.sleep(2)  # Wait for Arduino initialization
            ser.reset_input_buffer()

            start_time = None

            while self.running:
                try:
                    if ser.in_waiting:
                        line = ser.readline().decode("utf-8", errors="ignore").strip()
                        if not line or "Start:" in line:
                            if start_time is None:
                                start_time = time.time()
                            continue

                        # Parse both A0 and A1 from the line
                        parts = line.split("||")
                        for part in parts:
                            match = ARDUINO_OUTPUT_PATTERN.search(part)
                            if not match:
                                continue

                            try:
                                timestamp = time.time() - start_time if start_time else 0

                                channel = f"A{match.group('channel')}"
                                adc_live = float(match.group('adc_live'))
                                adc_mean10 = float(match.group('adc_mean10'))
                                d_mean = float(match.group('d_mean'))
                                u_live = float(match.group('u_live'))
                                u_mean10 = float(match.group('u_mean10'))

                                resistance_str = match.group('resistance')
                                resistance = None if "unendlich" in resistance_str else float(resistance_str)

                                data = SensorData(
                                    timestamp=timestamp,
                                    channel=channel,
                                    adc_live=adc_live,
                                    adc_mean10=adc_mean10,
                                    d_mean=d_mean,
                                    voltage_live=u_live,
                                    voltage_mean10=u_mean10,
                                    resistance=resistance,
                                )

                                with self.lock:
                                    if channel == "A0":
                                        self.data_a0.append(data)
                                    else:
                                        self.data_a1.append(data)
                            except (ValueError, IndexError):
                                pass
                    else:
                        time.sleep(0.001)
                except Exception as e:
                    self.error = f"Read error: {e}"
                    break

            ser.close()
        except serial.SerialException as e:
            self.error = f"Serial connection error: {e}"
            self.connected = False
        except Exception as e:
            self.error = f"Unexpected error: {e}"

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of current data."""
        with self.lock:
            data_a0 = list(self.data_a0)
            data_a1 = list(self.data_a1)

        return {
            "a0": {
                "x": list(range(len(data_a0))),
                "adc": [d.adc_mean10 for d in data_a0],
                "voltage": [d.voltage_mean10 for d in data_a0],
                "dmean": [d.d_mean for d in data_a0],
                "resistance": [d.resistance for d in data_a0],
            },
            "a1": {
                "x": list(range(len(data_a1))),
                "adc": [d.adc_mean10 for d in data_a1],
                "voltage": [d.voltage_mean10 for d in data_a1],
                "dmean": [d.d_mean for d in data_a1],
                "resistance": [d.resistance for d in data_a1],
            },
            "count_a0": len(data_a0),
            "count_a1": len(data_a1),
            "connected": self.connected,
        }

    def stop(self):
        """Stop the reader thread."""
        self.running = False


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Arduino Sensor Live Monitor</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body { font-family: sans-serif; margin: 0; padding: 10px; background: #f5f5f5; }
        h1 { font-size: 18px; text-align: center; }
        #status { text-align: center; color: #666; margin-bottom: 8px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .cell { background: white; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.15); }
    </style>
</head>
<body>
    <h1>Arduino Sensor Live Monitor (A0 &amp; A1)</h1>
    <div id="status">Connecting…</div>
    <div class="grid">
        <div class="cell"><div id="adc"></div></div>
        <div class="cell"><div id="voltage"></div></div>
        <div class="cell"><div id="dmean"></div></div>
        <div class="cell"><div id="resistance"></div></div>
    </div>
<script>
const layout = (title, ytitle) => ({
    title: { text: title, font: { size: 14 } },
    margin: { l: 55, r: 15, t: 35, b: 35 },
    height: 320,
    xaxis: { title: 'Sample' },
    yaxis: { title: ytitle },
    legend: { orientation: 'h', y: 1.15 },
});
const cfg = { responsive: true, displayModeBar: false };

function traces(d, key, w) {
    return [
        { x: d.a0.x, y: d.a0[key], name: 'A0', mode: 'lines', line: { color: 'blue', width: w } },
        { x: d.a1.x, y: d.a1[key], name: 'A1', mode: 'lines', line: { color: 'red', width: w } },
    ];
}

Plotly.newPlot('adc', [], layout('ADC Mean (10-sample MA)', 'ADC'), cfg);
Plotly.newPlot('voltage', [], layout('Voltage Mean', 'Voltage (V)'), cfg);
Plotly.newPlot('dmean', [], layout('Delta Mean (Rate of Change)', 'ΔMean'), cfg);
Plotly.newPlot('resistance', [], layout('Estimated Resistance', 'Resistance (Ω)'), cfg);

async function refresh() {
    try {
        const resp = await fetch('/data');
        const d = await resp.json();
        Plotly.react('adc', traces(d, 'adc', 2), layout('ADC Mean (10-sample MA)', 'ADC'), cfg);
        Plotly.react('voltage', traces(d, 'voltage', 2), layout('Voltage Mean', 'Voltage (V)'), cfg);
        Plotly.react('dmean', traces(d, 'dmean', 1.5), layout('Delta Mean (Rate of Change)', 'ΔMean'), cfg);
        Plotly.react('resistance', traces(d, 'resistance', 1.5), layout('Estimated Resistance', 'Resistance (Ω)'), cfg);
        document.getElementById('status').textContent =
            `A0: ${d.count_a0} samples | A1: ${d.count_a1} samples | ${d.connected ? 'Connected' : 'Disconnected'}`;
    } catch (e) {
        document.getElementById('status').textContent = 'Connection lost: ' + e;
    }
}
setInterval(refresh, 200);
refresh();
</script>
</body>
</html>
"""


def make_handler(reader: SerialReader):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Silence request logging

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/data":
                body = json.dumps(reader.snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


def main():
    parser = argparse.ArgumentParser(
        description="Live plotter for Arduino dual sensor output (built-in web server + Plotly.js)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_sensor_live.py --port /dev/ttyACM0
  python plot_sensor_live.py --port COM3 --max-points 1000 --http-port 8060
        """,
    )
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--max-points", type=int, default=500, help="Max points to display")
    parser.add_argument("--http-port", type=int, default=8050, help="HTTP server port")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser")

    args = parser.parse_args()

    print(f"Connecting to {args.port}...")
    reader = SerialReader(args.port, args.baudrate, args.max_points)
    reader.start()

    # Wait for the reader to either connect or fail (up to 3s)
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if reader.connected or reader.error:
            break
        time.sleep(0.05)

    if not reader.connected:
        print(f"Error: Could not connect to {args.port}")
        if reader.error:
            print(f"  {reader.error}")
        sys.exit(1)

    print("Connected! Starting web server...")

    server = ThreadingHTTPServer(("127.0.0.1", args.http_port), make_handler(reader))
    url = f"http://127.0.0.1:{args.http_port}/"

    print(f"Live plot available at: {url}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url)), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.shutdown()
        reader.stop()
        reader.join(timeout=2)
        print("Stopped.")


if __name__ == "__main__":
    main()
