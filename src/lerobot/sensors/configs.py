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

import abc
from dataclasses import dataclass

import draccus  # type: ignore


@dataclass(kw_only=True)
class SensorConfig(draccus.ChoiceRegistry, abc.ABC):  # type: ignore
    """Abstract base configuration for all sensor types.

    Subclasses are registered with :py:meth:`SensorConfig.register_subclass` so
    that YAML / CLI configs can specify the sensor type by name.

    Attributes:
        feature_dim: Number of scalar values produced per reading.
        optional: When ``True`` (default), a missing or unresponsive device
            only emits a warning and the sensor falls back to zeros — the
            robot/pipeline continues running.  Set to ``False`` to raise a
            :exc:`ConnectionError` when the device cannot be contacted during
            :py:meth:`Sensor.connect`.
        warmup_s: Seconds to wait for the first valid reading during
            :py:meth:`Sensor.connect`.  ``0.0`` (default) skips the warmup
            check entirely (analogous to ``camera.connect(warmup=False)``).
    """

    feature_dim: int = 1
    optional: bool = True
    warmup_s: float = 0.0

    @property
    def type(self) -> str:
        return str(self.get_choice_name(self.__class__))
