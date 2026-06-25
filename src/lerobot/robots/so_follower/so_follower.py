#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
import time
from functools import cached_property
from typing import TYPE_CHECKING

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _serial_available, require_package

if TYPE_CHECKING or _serial_available:
    import serial
else:
    serial = None  # type: ignore[assignment]

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_so_follower import SOFollowerRobotConfig

logger = logging.getLogger(__name__)

_SENSOR_VALUE_PATTERNS = {
    "live": re.compile(r"ADC_live:\s*([-+]?\d+(?:\.\d+)?)"),
    "mean10": re.compile(r"ADC_mean10:\s*([-+]?\d+(?:\.\d+)?)"),
    "dmean": re.compile(r"dMean:\s*([-+]?\d+(?:\.\d+)?)"),
}

_SENSOR_CHANNEL_PATTERNS = {
    "live": {
        "a0": re.compile(r"A0\s+ADC_live:\s*([-+]?\d+(?:\.\d+)?)"),
        "a1": re.compile(r"A1\s+ADC_live:\s*([-+]?\d+(?:\.\d+)?)"),
    },
    "mean10": {
        "a0": re.compile(r"A0\s+.*?ADC_mean10:\s*([-+]?\d+(?:\.\d+)?)"),
        "a1": re.compile(r"A1\s+.*?ADC_mean10:\s*([-+]?\d+(?:\.\d+)?)"),
    },
    "dmean": {
        "a0": re.compile(r"A0\s+.*?dMean:\s*([-+]?\d+(?:\.\d+)?)"),
        "a1": re.compile(r"A1\s+.*?dMean:\s*([-+]?\d+(?:\.\d+)?)"),
    },
}


def _parse_sensor_value(line: bytes, mode: str) -> float | None:
    text = line.decode("utf-8", errors="ignore").strip()
    if not text:
        return None

    pattern = _SENSOR_VALUE_PATTERNS.get(mode)
    if pattern is not None:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None

    # Fallback: first floating-point token in the line.
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_sensor_values(line: bytes, mode: str) -> dict[str, float]:
    text = line.decode("utf-8", errors="ignore").strip()
    if not text:
        return {}

    parsed_values: dict[str, float] = {}
    channel_patterns = _SENSOR_CHANNEL_PATTERNS.get(mode, {})
    for channel, pattern in channel_patterns.items():
        match = pattern.search(text)
        if match is None:
            continue
        try:
            parsed_values[channel] = float(match.group(1))
        except ValueError:
            continue

    if parsed_values:
        return parsed_values

    fallback_value = _parse_sensor_value(line, mode)
    if fallback_value is None:
        return {}

    # Backward compatibility with single-sensor Arduino output without A0/A1 prefixes.
    return {"a0": fallback_value}


class SOFollower(Robot):
    """
    Generic SO follower base implementing common functionality for SO-100/101/10X.
    Designed to be subclassed with a per-hardware-model `config_class` and `name`.
    """

    config_class = SOFollowerRobotConfig
    name = "so_follower"

    def __init__(self, config: SOFollowerRobotConfig):
        super().__init__(config)
        self.config = config
        self._sensor_serial = None
        self._sensor_last_values = {
            "a0": float(config.sensor_default_value),
            "a1": float(config.sensor_default_value),
        }

        valid_modes = set(_SENSOR_VALUE_PATTERNS)
        if self.config.sensor_value_mode not in valid_modes:
            raise ValueError(
                f"sensor_value_mode must be one of {sorted(valid_modes)}. Got '{self.config.sensor_value_mode}'."
            )

        valid_channels = {"a0", "a1", "both"}
        if self.config.sensor_channel not in valid_channels:
            raise ValueError(
                f"sensor_channel must be one of {sorted(valid_channels)}. Got '{self.config.sensor_channel}'."
            )

        if self.config.sensor_enabled and not self.config.sensor_port:
            raise ValueError("sensor_port must be set when sensor_enabled=True.")
        # choose normalization mode depending on config if available
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features = {**self._motors_ft, **self._cameras_ft}
        if self.config.sensor_enabled:
            if self.config.sensor_channel == "both":
                features[self.config.sensor_feature_name_a0] = float
                features[self.config.sensor_feature_name_a1] = float
            else:
                features[self.config.sensor_feature_name] = float
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        motors_and_cameras_connected = self.bus.is_connected and all(
            cam.is_connected for cam in self.cameras.values()
        )
        if not self.config.sensor_enabled:
            return motors_and_cameras_connected
        return motors_and_cameras_connected and self._sensor_serial is not None and self._sensor_serial.is_open

    def _connect_sensor_if_enabled(self) -> None:
        if not self.config.sensor_enabled:
            return

        require_package("pyserial", extra="pyserial-dep", import_name="serial")
        try:
            self._sensor_serial = serial.Serial(
                self.config.sensor_port,
                self.config.sensor_baud_rate,
                timeout=self.config.sensor_timeout_s,
            )
            self._sensor_serial.reset_input_buffer()
            logger.info(
                "Connected optional analog sensor on %s (%d baud)",
                self.config.sensor_port,
                self.config.sensor_baud_rate,
            )
        except Exception as exc:
            self._sensor_serial = None
            msg = (
                f"Failed to connect optional sensor on '{self.config.sensor_port}': {exc}. "
                "Continuing without sensor."
            )
            if self.config.sensor_strict:
                raise ConnectionError(msg) from exc
            logger.warning(msg)

    def _read_sensor_values(self) -> dict[str, float]:
        if not self.config.sensor_enabled or self._sensor_serial is None:
            return dict(self._sensor_last_values)

        try:
            last_values = dict(self._sensor_last_values)
            while self._sensor_serial.in_waiting > 0:
                line = self._sensor_serial.readline()
                if not line:
                    break
                parsed = _parse_sensor_values(line, self.config.sensor_value_mode)
                if parsed:
                    last_values.update(parsed)

            if last_values == self._sensor_last_values:
                line = self._sensor_serial.readline()
                if line:
                    parsed = _parse_sensor_values(line, self.config.sensor_value_mode)
                    if parsed:
                        last_values.update(parsed)

            self._sensor_last_values = last_values

            return dict(self._sensor_last_values)
        except Exception as exc:
            if self.config.sensor_strict:
                raise RuntimeError(f"Sensor read failed: {exc}") from exc
            logger.debug("Optional sensor read failed (%s), reusing last values", exc)
            return dict(self._sensor_last_values)

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self._connect_sensor_if_enabled()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        # Attempt to call record_ranges_of_motion with a reduced motor set when appropriate.
        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.bus.write("P_Coefficient", motor, 16)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                if motor == "gripper":
                    self.bus.write("Max_Torque_Limit", motor, 500)  # 50% of max torque to avoid burnout
                    self.bus.write("Protection_Current", motor, 250)  # 50% of max current to avoid burnout
                    self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        if self.config.sensor_enabled:
            sensor_values = self._read_sensor_values()
            if self.config.sensor_channel == "both":
                obs_dict[self.config.sensor_feature_name_a0] = sensor_values["a0"]
                obs_dict[self.config.sensor_feature_name_a1] = sensor_values["a1"]
            elif self.config.sensor_channel == "a1":
                obs_dict[self.config.sensor_feature_name] = sensor_values["a1"]
            else:
                obs_dict[self.config.sensor_feature_name] = sensor_values["a0"]

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            RobotAction: the action sent to the motors, potentially clipped.
        """

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        if self._sensor_serial is not None:
            try:
                self._sensor_serial.close()
            finally:
                self._sensor_serial = None

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")


SO100Follower = SOFollower
SO101Follower = SOFollower
