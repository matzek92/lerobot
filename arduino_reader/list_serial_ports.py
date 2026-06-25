#!/usr/bin/env python3
"""
Helper script to list all available serial ports.
Useful for identifying Arduino connections.
"""

import sys
from pathlib import Path

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("Error: pyserial not installed. Install it with: pip install pyserial")
    sys.exit(1)


def main():
    print("Available Serial Ports:")
    print("=" * 80)
    
    ports = list_ports.comports()
    
    if not ports:
        print("No serial ports found!")
        return
    
    for i, port in enumerate(ports, 1):
        print(f"\n{i}. {port.device}")
        print(f"   Description: {port.description}")
        print(f"   Manufacturer: {port.manufacturer}")
        print(f"   Serial Number: {port.serial_number}")
        print(f"   Hardware ID: {port.hwid}")
    
    print("\n" + "=" * 80)
    print("\nLinux users can also use:")
    print("  dmesg | tail -20          # Check kernel messages")
    print("  ls /dev/tty*              # List all tty devices")
    print("  ls /dev/serial/by-id/     # Find by hardware ID")
    print("\nTo use with test_sensor_reader.py:")
    print("  python test_sensor_reader.py --port /dev/ttyUSB0")


if __name__ == "__main__":
    main()
