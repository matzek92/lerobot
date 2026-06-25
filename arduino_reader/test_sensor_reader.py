#!/usr/bin/env python3
"""
Test script for reading Arduino dual sensor (A0/A1) output over serial.

This script connects to an Arduino running arduino_a0_reader.ino and parses
the dual-channel sensor output (ADC values, voltages, resistances).

Usage:
    python test_sensor_reader.py --port /dev/ttyUSB0 --baudrate 115200 [--duration 10] [--output data.csv]
"""

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import serial
except ImportError:
    print("Error: pyserial not installed. Install it with: pip install pyserial")
    sys.exit(1)


@dataclass
class SensorReading:
    """Single sensor reading with parsed values."""
    timestamp: float
    channel: str  # "A0" or "A1"
    adc_live: float
    adc_mean10: float
    d_mean: float
    voltage_live: float
    voltage_mean10: float
    resistance: Optional[float]  # None if "unendlich / ausser Bereich"

    def __str__(self) -> str:
        r_str = f"{self.resistance:.0f} Ω" if self.resistance is not None else "∞"
        return (
            f"{self.channel} @ {self.timestamp:.3f}s | "
            f"ADC: {self.adc_live:.0f} (live) / {self.adc_mean10:.1f} (mean) | "
            f"ΔMean: {self.d_mean:.2f} | "
            f"Voltage: {self.voltage_live:.3f}V (live) / {self.voltage_mean10:.3f}V (mean) | "
            f"R: {r_str}"
        )


# Regex patterns for parsing the Arduino output
ARDUINO_OUTPUT_PATTERN = re.compile(
    r"A(?P<channel>[01])\s+ADC_live:\s+(?P<adc_live>[\d.]+)\s+\|\s+"
    r"ADC_mean10:\s+(?P<adc_mean10>[\d.]+)\s+\|\s+"
    r"dMean:\s+(?P<d_mean>[-\d.]+)\s+\|\s+"
    r"U_live:\s+(?P<u_live>[\d.]+)\s+V\s+\|\s+"
    r"U_mean10:\s+(?P<u_mean10>[\d.]+)\s+V\s+\|\s+"
    r"R~\(mean\):\s+(?P<resistance>(?:\d+(?:\.\d+)?|unendlich / ausser Bereich))"
)


def parse_sensor_line(line: str, timestamp: float) -> Optional[SensorReading]:
    """Parse a single line from Arduino output."""
    # Handle lines that might contain both A0 and A1 (separated by ||)
    parts = line.split("||")
    
    readings = []
    for part in parts:
        match = ARDUINO_OUTPUT_PATTERN.search(part)
        if not match:
            continue
        
        try:
            channel = f"A{match.group('channel')}"
            adc_live = float(match.group('adc_live'))
            adc_mean10 = float(match.group('adc_mean10'))
            d_mean = float(match.group('d_mean'))
            u_live = float(match.group('u_live'))
            u_mean10 = float(match.group('u_mean10'))
            
            resistance_str = match.group('resistance')
            if "unendlich" in resistance_str or "ausser" in resistance_str:
                resistance = None
            else:
                resistance = float(resistance_str)
            
            readings.append(
                SensorReading(
                    timestamp=timestamp,
                    channel=channel,
                    adc_live=adc_live,
                    adc_mean10=adc_mean10,
                    d_mean=d_mean,
                    voltage_live=u_live,
                    voltage_mean10=u_mean10,
                    resistance=resistance,
                )
            )
        except (ValueError, IndexError) as e:
            print(f"Warning: Failed to parse part of line: {part[:80]} ({e})")
            continue
    
    return readings


def save_to_csv(readings: list[SensorReading], output_file: Path) -> None:
    """Save readings to a CSV file."""
    import csv
    
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "channel",
                "adc_live",
                "adc_mean10",
                "d_mean",
                "voltage_live",
                "voltage_mean10",
                "resistance_ohm",
            ],
        )
        writer.writeheader()
        for r in readings:
            writer.writerow({
                "timestamp": f"{r.timestamp:.6f}",
                "channel": r.channel,
                "adc_live": f"{r.adc_live:.2f}",
                "adc_mean10": f"{r.adc_mean10:.2f}",
                "d_mean": f"{r.d_mean:.4f}",
                "voltage_live": f"{r.voltage_live:.6f}",
                "voltage_mean10": f"{r.voltage_mean10:.6f}",
                "resistance_ohm": f"{r.resistance:.2f}" if r.resistance else "inf",
            })
    
    print(f"\nData saved to {output_file}")


def print_statistics(readings: list[SensorReading]) -> None:
    """Print statistics for each channel."""
    by_channel = {}
    for r in readings:
        if r.channel not in by_channel:
            by_channel[r.channel] = []
        by_channel[r.channel].append(r)
    
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    for channel in sorted(by_channel.keys()):
        data = by_channel[channel]
        print(f"\n{channel}:")
        print(f"  Samples: {len(data)}")
        
        adc_means = [r.adc_mean10 for r in data]
        print(f"  ADC (mean10) - min: {min(adc_means):.1f}, max: {max(adc_means):.1f}, avg: {sum(adc_means) / len(adc_means):.1f}")
        
        voltages = [r.voltage_mean10 for r in data]
        print(f"  Voltage (mean) - min: {min(voltages):.4f}V, max: {max(voltages):.4f}V, avg: {sum(voltages) / len(voltages):.4f}V")
        
        resistances = [r.resistance for r in data if r.resistance is not None]
        if resistances:
            print(f"  Resistance - min: {min(resistances):.0f}Ω, max: {max(resistances):.0f}Ω, avg: {sum(resistances) / len(resistances):.0f}Ω")
        
        # Calculate delta statistics
        deltas = [r.d_mean for r in data]
        print(f"  ΔMean - min: {min(deltas):.4f}, max: {max(deltas):.4f}, avg: {sum(deltas) / len(deltas):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Test script for reading Arduino dual sensor output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Read from serial port for 10 seconds
  python test_sensor_reader.py --port /dev/ttyUSB0 --duration 10

  # Read from serial port and save to CSV
  python test_sensor_reader.py --port /dev/ttyUSB0 --duration 30 --output sensor_data.csv

  # Read from specific COM port on Windows
  python test_sensor_reader.py --port COM3 --baudrate 115200
        """
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port (default: /dev/ttyUSB0, e.g., COM3 on Windows)",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="Serial baudrate (default: 115200)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10,
        help="How long to read (in seconds, default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save data to CSV file (optional)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print all readings to console",
    )

    args = parser.parse_args()

    print(f"Connecting to {args.port} @ {args.baudrate} baud...")
    print(f"Reading for {args.duration} seconds...")
    print("=" * 80)

    try:
        ser = serial.Serial(args.port, args.baudrate, timeout=1)
        time.sleep(2)  # Wait for Arduino to initialize
        
        # Clear any buffered data
        ser.reset_input_buffer()
        
        all_readings = []
        start_time = time.time()
        line_count = 0
        read_count = 0

        while time.time() - start_time < args.duration:
            try:
                if ser.in_waiting:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    
                    line_count += 1
                    timestamp = time.time() - start_time
                    
                    # Skip header line
                    if "Start:" in line or "Messung" in line:
                        print(f"[INFO] {line}")
                        continue
                    
                    readings = parse_sensor_line(line, timestamp)
                    if readings:
                        for r in readings:
                            read_count += 1
                            all_readings.append(r)
                            if args.verbose:
                                print(r)
                    else:
                        if line and args.verbose:
                            print(f"[UNPARSED] {line[:80]}")
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"Error reading line: {e}", file=sys.stderr)
                continue

        ser.close()
        
        print("=" * 80)
        print(f"Read {line_count} lines, parsed {read_count} sensor readings")
        print(f"Total time: {time.time() - start_time:.2f}s")
        
        if all_readings:
            print_statistics(all_readings)
            
            if args.output:
                save_to_csv(all_readings, args.output)

    except serial.SerialException as e:
        print(f"Error: Could not open serial port {args.port}: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if all_readings:
            print(f"Collected {len(all_readings)} readings before interruption")
            if args.output:
                save_to_csv(all_readings, args.output)


if __name__ == "__main__":
    main()
