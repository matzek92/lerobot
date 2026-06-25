#!/usr/bin/env python3
"""
Diagnostic script for Arduino sensor reader troubleshooting.
"""

import sys
import time
from pathlib import Path

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("Error: pyserial not installed. Install it with: pip install pyserial")
    sys.exit(1)


def test_port_connection(port: str, baudrate: int = 115200) -> bool:
    """Test if we can connect to the serial port."""
    try:
        ser = serial.Serial(port, baudrate, timeout=2)
        print(f"✓ Connected to {port} @ {baudrate} baud")
        ser.close()
        return True
    except serial.SerialException as e:
        print(f"✗ Cannot connect to {port}: {e}")
        return False


def read_raw_output(port: str, baudrate: int = 115200, timeout_s: float = 5) -> list[str]:
    """Read raw output from Arduino without parsing."""
    print(f"\nReading raw output from {port} for {timeout_s}s...")
    print("-" * 80)
    
    lines = []
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for Arduino to initialize
        ser.reset_input_buffer()
        
        start = time.time()
        while time.time() - start < timeout_s:
            if ser.in_waiting:
                try:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        print(line)
                        lines.append(line)
                except Exception as e:
                    print(f"Error decoding: {e}")
            else:
                time.sleep(0.01)
        
        ser.close()
        print("-" * 80)
        return lines
    except Exception as e:
        print(f"Error reading: {e}")
        return []


def check_parsing(lines: list[str]) -> None:
    """Check if lines can be parsed."""
    import re
    
    if not lines:
        print("✗ No data to parse")
        return
    
    # Check for expected patterns
    patterns = {
        "A0 data": r"A0\s+ADC_live:",
        "A1 data": r"A1\s+ADC_live:",
        "Header line": r"Start:|Messung",
    }
    
    print(f"\nChecking {len(lines)} lines for expected patterns:")
    found = {name: False for name in patterns}
    
    for line in lines:
        for name, pattern in patterns.items():
            if re.search(pattern, line):
                found[name] = True
    
    for name, is_found in found.items():
        symbol = "✓" if is_found else "✗"
        print(f"{symbol} {name}")
    
    if not found["A0 data"] and not found["A1 data"]:
        print("\n⚠ WARNING: No sensor data patterns found!")
        print("First few lines received:")
        for line in lines[:5]:
            print(f"  {line[:100]}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Troubleshoot Arduino sensor reader")
    parser.add_argument("--port", type=str, default="/dev/ttyACM1", help="Serial port to test")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--timeout", type=float, default=5, help="Read timeout in seconds")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ARDUINO SENSOR READER DIAGNOSTICS")
    print("=" * 80)
    
    # Step 1: List all ports
    print("\n1. AVAILABLE SERIAL PORTS:")
    print("-" * 80)
    ports = list(list_ports.comports())
    if not ports:
        print("✗ No serial ports found!")
    else:
        for i, port in enumerate(ports, 1):
            print(f"{i}. {port.device}: {port.description}")
    
    # Step 2: Test connection
    print(f"\n2. TESTING CONNECTION:")
    if not test_port_connection(args.port, args.baudrate):
        print(f"\nAvailable ports:")
        for port in ports:
            print(f"  - {port.device}")
        sys.exit(1)
    
    # Step 3: Read raw output
    print(f"\n3. RAW OUTPUT TEST:")
    lines = read_raw_output(args.port, args.baudrate, args.timeout)
    
    if not lines:
        print("\n⚠ No data received! Check:")
        print("  1. Is Arduino powered on?")
        print("  2. Is the USB cable properly connected?")
        print("  3. Is the correct sketch loaded on Arduino?")
        print("  4. Check with: cat /dev/ttyACM1  (or appropriate port)")
        sys.exit(1)
    
    # Step 4: Check parsing
    print(f"\n4. DATA PARSING TEST:")
    check_parsing(lines)
    
    # Step 5: Statistics
    print(f"\n5. DATA STATISTICS:")
    print(f"Total lines received: {len(lines)}")
    a0_lines = [l for l in lines if "A0" in l and "ADC_live" in l]
    a1_lines = [l for l in lines if "A1" in l and "ADC_live" in l]
    print(f"A0 readings: {len(a0_lines)}")
    print(f"A1 readings: {len(a1_lines)}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
