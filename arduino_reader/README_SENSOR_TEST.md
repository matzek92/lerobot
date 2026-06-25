# Arduino Sensor Reader Test Script

Python script for testing and monitoring the dual-channel Arduino sensor reader (`arduino_a0_reader.ino`).

## Features

- **Real-time serial reading**: Connect to Arduino and read sensor data at 1kHz
- **Dual-channel support**: Parse both A0 and A1 analog inputs simultaneously
- **Data parsing**: Extract ADC values, voltages, resistances, and delta measurements
- **CSV export**: Save readings to CSV for post-processing or visualization
- **Statistics**: Print min/max/average statistics for each channel
- **Verbose mode**: Optional console output for each reading

## Installation

1. Install pyserial:
```bash
pip install pyserial
```

2. Identify your Arduino serial port:
   - **Linux**: `/dev/ttyUSB0`, `/dev/ttyUSB1`, etc. (use `ls /dev/tty*`)
   - **macOS**: `/dev/tty.usbserial-*` (use `ls /dev/tty.*`)
   - **Windows**: `COM3`, `COM4`, etc. (check Device Manager)

## Usage

### Basic usage (read for 10 seconds):
```bash
python test_sensor_reader.py --port /dev/ttyUSB0
```

### Read for 30 seconds and save to CSV:
```bash
python test_sensor_reader.py --port /dev/ttyUSB0 --duration 30 --output sensor_data.csv
```

### Verbose mode (print each reading):
```bash
python test_sensor_reader.py --port /dev/ttyUSB0 --duration 10 --verbose
```

### Windows example:
```bash
python test_sensor_reader.py --port COM3 --baudrate 115200 --duration 10
```

## Command-line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | `/dev/ttyUSB0` | Serial port path |
| `--baudrate` | `115200` | Serial baudrate |
| `--duration` | `10` | Read duration in seconds |
| `--output` | None | CSV output file path (optional) |
| `--verbose` / `-v` | False | Print all readings to console |

## Output Example

```
================================================================================
A0 @ 0.045s | ADC: 512 (live) / 510.5 (mean) | ΔMean: 0.12 | Voltage: 2.499V (live) / 2.493V (mean) | R: 30245 Ω
A1 @ 0.045s | ADC: 768 (live) / 765.3 (mean) | ΔMean: -0.08 | Voltage: 3.749V (live) / 3.742V (mean) | R: 15234 Ω
...
================================================================================
STATISTICS
================================================================================

A0:
  Samples: 450
  ADC (mean10) - min: 508.2, max: 512.1, avg: 510.4
  Voltage (mean) - min: 2.4896V, max: 2.5049V, avg: 2.4947V
  Resistance - min: 29500Ω, max: 31200Ω, avg: 30124Ω
  ΔMean - min: -0.45, max: 0.52, avg: 0.02

A1:
  Samples: 450
  ADC (mean10) - min: 763.1, max: 768.9, avg: 765.6
  Voltage (mean) - min: 3.7389V, max: 3.7611V, avg: 3.7488V
  Resistance - min: 14800Ω, max: 15600Ω, avg: 15234Ω
  ΔMean - min: -0.38, max: 0.41, avg: 0.01

Data saved to sensor_data.csv
```

## CSV Output

The exported CSV contains these columns:
- `timestamp` - Time since start (seconds)
- `channel` - Sensor channel (A0 or A1)
- `adc_live` - Current ADC reading
- `adc_mean10` - 10-sample moving average
- `d_mean` - Change in moving average
- `voltage_live` - Current voltage reading
- `voltage_mean10` - Averaged voltage
- `resistance_ohm` - Estimated resistance (or "inf" if out of range)

## Troubleshooting

**"Could not open serial port"**
- Check the port number with `ls /dev/tty*` (Linux) or Device Manager (Windows)
- Ensure you have permissions: `sudo chmod 666 /dev/ttyUSB0` (Linux)

**"Failed to parse"**
- Check Arduino baud rate matches the script (`--baudrate` flag)
- Ensure Arduino sketch is running and sending data

**No data received**
- Verify Arduino is connected: `cat /dev/ttyUSB0` (should show output)
- Check for loose connections or defective USB cable

## Integration with SO Follower

This sensor reader is compatible with the SO Follower robot configuration:
- Configure in `config_so_follower.py` with `sensor_enabled=True`
- Set the serial port with `sensor_port="/dev/ttyUSB0"` (or appropriate port)
- Choose reading mode: `sensor_value_mode="mean10"` (recommended)
- Select channel(s): `sensor_channel="a0"`, `"a1"`, or `"both"`
