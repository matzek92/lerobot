# Arduino Sensor Live Plotter

Real-time visualization of Arduino dual sensor (A0/A1) data with matplotlib.

## Installation

Install matplotlib if not already present:
```bash
pip install matplotlib
```

## Usage

Basic usage:
```bash
python plot_sensor_live.py --port /dev/ttyACM0
```

Custom settings:
```bash
# Use different serial port
python plot_sensor_live.py --port /dev/ttyUSB0

# Windows example
python plot_sensor_live.py --port COM3

# Increase buffer size for longer monitoring
python plot_sensor_live.py --port /dev/ttyACM0 --max-points 1000
```

## Command-line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | `/dev/ttyACM0` | Serial port path |
| `--baudrate` | `115200` | Serial baudrate |
| `--max-points` | `500` | Maximum data points to display on graphs |

## Display

The plotter shows 4 real-time graphs:

1. **ADC Mean Values** (top) - 10-sample moving average
   - A0 in blue, A1 in red
   - Range: 0-1023

2. **Voltage Mean Values** (middle-top) - Calculated from ADC
   - A0 in blue, A1 in red
   - Range: 0-5.5V

3. **Delta Mean / Rate of Change** (bottom-left)
   - Shows how quickly values are changing
   - Useful for detecting motion

4. **Estimated Resistance** (bottom-right)
   - Calculated from ADC values
   - Only shown if in valid range

## Status Information

The bottom of the window shows:
- Number of samples collected for each channel
- Current connection status

## Tips

- **Close the window** to exit the plotter
- **Scroll/pan** in the graphs by clicking and dragging
- **Zoom** using the zoom tool in the toolbar
- **Maximize window** for better visibility with multiple graphs

## Troubleshooting

If no data appears:
1. Check the serial port: `python list_serial_ports.py`
2. Verify Arduino is sending data: `python diagnose_arduino.py --port /dev/ttyACM0`
3. Ensure correct port in `--port` argument

If the plot is choppy:
- Reduce `--max-points` for better performance
- Close other applications to free up CPU
