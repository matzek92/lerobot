#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import json
import logging
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import (
    ObservationProcessorStep,
    ProcessorStepRegistry,
)
from lerobot.robots import Robot
from lerobot.utils.constants import OBS_STATE

logger = logging.getLogger(__name__)


@dataclass
@ProcessorStepRegistry.register("joint_velocity_processor")
class JointVelocityProcessorStep(ObservationProcessorStep):
    """
    Calculates and appends joint velocity information to the observation state.

    This step computes the velocity of each joint by calculating the finite
    difference between the current and the last observed joint positions. The
    resulting velocity vector is then concatenated to the original state vector.

    Attributes:
        dt: The time step (delta time) in seconds between observations, used for
            calculating velocity.
        last_joint_positions: Stores the joint positions from the previous step
                              to enable velocity calculation.
    """

    dt: float = 0.1

    last_joint_positions: torch.Tensor | None = None

    def observation(self, observation: dict) -> dict:
        """
        Computes joint velocities and adds them to the observation state.

        Args:
            observation: The input observation dictionary, expected to contain
                         an `observation.state` key with joint positions.

        Returns:
            A new observation dictionary with the `observation.state` tensor
            extended to include joint velocities.

        Raises:
            ValueError: If `observation.state` is not found in the observation.
        """
        # Get current joint positions (assuming they're in observation.state)
        current_positions = observation.get(OBS_STATE)
        if current_positions is None:
            raise ValueError(f"{OBS_STATE} is not in observation")

        # Initialize last joint positions if not already set
        if self.last_joint_positions is None:
            self.last_joint_positions = current_positions.clone()
            joint_velocities = torch.zeros_like(current_positions)
        else:
            # Compute velocities
            joint_velocities = (current_positions - self.last_joint_positions) / self.dt

        self.last_joint_positions = current_positions.clone()

        # Extend observation with velocities
        extended_state = torch.cat([current_positions, joint_velocities], dim=-1)

        # Create new observation dict
        new_observation = dict(observation)
        new_observation[OBS_STATE] = extended_state

        return new_observation

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the time step `dt`.
        """
        return {
            "dt": self.dt,
        }

    def reset(self) -> None:
        """Resets the internal state, clearing the last known joint positions."""
        self.last_joint_positions = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the `observation.state` feature to reflect the added velocities.

        This method doubles the size of the first dimension of the `observation.state`
        shape to account for the concatenation of position and velocity vectors.

        Args:
            features: The policy features dictionary.

        Returns:
            The updated policy features dictionary.
        """
        if OBS_STATE in features[PipelineFeatureType.OBSERVATION]:
            original_feature = features[PipelineFeatureType.OBSERVATION][OBS_STATE]
            # Double the shape to account for positions + velocities
            new_shape = (original_feature.shape[0] * 2,) + original_feature.shape[1:]

            features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
                type=original_feature.type, shape=new_shape
            )
        return features


@dataclass
@ProcessorStepRegistry.register("current_processor")
class MotorCurrentProcessorStep(ObservationProcessorStep):
    """
    Reads motor currents from a robot and appends them to the observation state.

    This step queries the robot's hardware interface to get the present current
    for each motor and concatenates this information to the existing state vector.

    Attributes:
        robot: An instance of a `lerobot` Robot class that provides access to
               the hardware bus.
    """

    robot: Robot | None = None

    def observation(self, observation: dict) -> dict:
        """
        Fetches motor currents and adds them to the observation state.

        Args:
            observation: The input observation dictionary.

        Returns:
            A new observation dictionary with the `observation.state` tensor
            extended to include motor currents.

        Raises:
            ValueError: If the `robot` attribute has not been set.
        """
        # Get current values from robot state
        if self.robot is None:
            raise ValueError("Robot is not set")

        present_current_dict = self.robot.bus.sync_read("Present_Current")  # type: ignore[attr-defined]
        motor_currents = torch.tensor(
            [present_current_dict[name] for name in self.robot.bus.motors],  # type: ignore[attr-defined]
            dtype=torch.float32,
        ).unsqueeze(0)

        current_state = observation.get(OBS_STATE)
        if current_state is None:
            return observation

        extended_state = torch.cat([current_state, motor_currents], dim=-1)

        # Create new observation dict
        new_observation = dict(observation)
        new_observation[OBS_STATE] = extended_state

        return new_observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the `observation.state` feature to reflect the added motor currents.

        This method increases the size of the first dimension of the `observation.state`
        shape by the number of motors in the robot.

        Args:
            features: The policy features dictionary.

        Returns:
            The updated policy features dictionary.
        """
        if OBS_STATE in features[PipelineFeatureType.OBSERVATION] and self.robot is not None:
            original_feature = features[PipelineFeatureType.OBSERVATION][OBS_STATE]
            # Add motor current dimensions to the original state shape
            num_motors = 0
            if hasattr(self.robot, "bus") and hasattr(self.robot.bus, "motors"):  # type: ignore[attr-defined]
                num_motors = len(self.robot.bus.motors)  # type: ignore[attr-defined]

            if num_motors > 0:
                new_shape = (original_feature.shape[0] + num_motors,) + original_feature.shape[1:]
                features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
                    type=original_feature.type, shape=new_shape
                )
        return features


@dataclass
class ZmqSensorConfig:
    """Configuration for a ZMQ sensor subscription.

    The sensor publisher is expected to send JSON messages with the format::

        {"timestamp": <float>, "features": [<float>, ...]}

    Attributes:
        address: TCP address of the ZMQ publisher (e.g. ``"127.0.0.1"``).
        port: Port of the ZMQ publisher.
        feature_dim: Expected number of float values per message.
        timeout_ms: Socket receive timeout in milliseconds. If no message arrives
            within this period the last received value (or zeros) is reused.
    """

    address: str = "127.0.0.1"
    port: int = 5600
    feature_dim: int = 1
    timeout_ms: int = 100


class ZmqSensorClient:
    """Subscribes to a ZMQ PUB socket and buffers the latest feature vector.

    The publisher must send UTF-8 JSON strings of the form::

        {"timestamp": 1234.5, "features": [1.0, 2.0, 3.0]}

    A background thread continuously reads from the socket so that
    :py:meth:`read_latest` always returns without blocking.

    Args:
        config: :class:`ZmqSensorConfig` describing the connection parameters.
    """

    def __init__(self, config: ZmqSensorConfig) -> None:
        self.config = config
        self._lock: Lock = Lock()
        self._latest_features: list[float] | None = None
        self._stop_event: Event = Event()
        self._thread: Thread | None = None

    def connect(self) -> None:
        """Connect to the ZMQ publisher and start the background read thread."""
        import zmq

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.RCVTIMEO, self.config.timeout_ms)
        self._socket.setsockopt(zmq.CONFLATE, True)
        self._socket.connect(f"tcp://{self.config.address}:{self.config.port}")

        self._stop_event.clear()
        self._thread = Thread(
            target=self._read_loop,
            daemon=True,
            name=f"ZmqSensorClient_{self.config.address}:{self.config.port}",
        )
        self._thread.start()
        logger.info(
            f"ZmqSensorClient connected to tcp://{self.config.address}:{self.config.port}"
        )

    def _read_loop(self) -> None:
        import zmq

        while not self._stop_event.is_set():
            try:
                message = self._socket.recv_string()
                data = json.loads(message)
                features = data.get("features")
                if features is not None:
                    with self._lock:
                        self._latest_features = list(features)
            except zmq.Again:
                # Receive timeout – no new data; just loop again
                pass
            except json.JSONDecodeError as exc:
                logger.warning(f"ZmqSensorClient received malformed JSON: {exc}")

    def read_latest(self) -> list[float]:
        """Return the most recent feature vector, or zeros if none received yet."""
        with self._lock:
            if self._latest_features is not None:
                return list(self._latest_features)
        return [0.0] * self.config.feature_dim

    def disconnect(self) -> None:
        """Stop the background thread and close the ZMQ socket."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if hasattr(self, "_socket"):
            self._socket.close()
        if hasattr(self, "_context"):
            self._context.term()
        logger.info("ZmqSensorClient disconnected.")


@dataclass
@ProcessorStepRegistry.register("zmq_sensor_processor")
class ZmqObservationProcessorStep(ObservationProcessorStep):
    """Reads features from a ZMQ publisher and appends them to ``observation.state``.

    External programs stream additional sensor features via ZMQ in the format::

        {"timestamp": <float>, "features": [<float>, ...]}

    This step receives those features and concatenates them to the existing robot
    state vector so that downstream policies treat them identically to motor states.

    Attributes:
        zmq_configs: List of :class:`ZmqSensorConfig` instances, one per publisher.
        _clients: ZMQ client instances created during :py:meth:`connect`.
    """

    zmq_configs: list[ZmqSensorConfig] = field(default_factory=list)
    _clients: list[ZmqSensorClient] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._clients = [ZmqSensorClient(cfg) for cfg in self.zmq_configs]
        for client in self._clients:
            client.connect()

    def observation(self, observation: dict) -> dict:
        """Fetch ZMQ features and append them to ``observation.state``.

        Args:
            observation: Input observation dictionary containing ``observation.state``.

        Returns:
            Updated observation dictionary with ZMQ features appended to the state.
        """
        if not self._clients:
            return observation

        all_features: list[float] = []
        for client in self._clients:
            all_features.extend(client.read_latest())

        zmq_tensor = torch.tensor(all_features, dtype=torch.float32).unsqueeze(0)

        current_state = observation.get(OBS_STATE)
        if current_state is None:
            observation[OBS_STATE] = zmq_tensor
            return observation

        new_observation = dict(observation)
        new_observation[OBS_STATE] = torch.cat([current_state, zmq_tensor], dim=-1)
        return new_observation

    def get_config(self) -> dict[str, Any]:
        return {
            "zmq_configs": [
                {
                    "address": cfg.address,
                    "port": cfg.port,
                    "feature_dim": cfg.feature_dim,
                    "timeout_ms": cfg.timeout_ms,
                }
                for cfg in self.zmq_configs
            ]
        }

    def reset(self) -> None:
        """No state to reset between episodes."""

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Extends ``observation.state`` shape to account for the ZMQ features.

        Args:
            features: Policy features dictionary.

        Returns:
            Updated policy features dictionary.
        """
        total_zmq_dim = sum(cfg.feature_dim for cfg in self.zmq_configs)
        if total_zmq_dim == 0:
            return features

        if OBS_STATE in features[PipelineFeatureType.OBSERVATION]:
            original_feature = features[PipelineFeatureType.OBSERVATION][OBS_STATE]
            new_shape = (original_feature.shape[0] + total_zmq_dim,) + original_feature.shape[1:]
            features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
                type=original_feature.type, shape=new_shape
            )
        return features

    def disconnect(self) -> None:
        """Disconnect all ZMQ clients."""
        for client in self._clients:
            client.disconnect()
