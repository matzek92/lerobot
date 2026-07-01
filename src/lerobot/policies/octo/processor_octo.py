#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_octo import OctoConfig


def make_octo_pre_post_processors(
    config: OctoConfig,
    dataset_stats=None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create minimal pre- and post-processing pipelines for OctoPolicy.

    Because Octo handles all observation normalization internally (via its own dataset
    statistics), the pre-processor only adds a batch dimension and places tensors on the
    configured device.  The post-processor moves results back to CPU.

    ``dataset_stats`` is accepted for API compatibility but intentionally ignored — Octo
    uses its own checkpoint-embedded statistics.

    Args:
        config: The ``OctoConfig`` instance for this policy.
        dataset_stats: Unused; kept for API parity with other policy processors.

    Returns:
        A ``(preprocessor, postprocessor)`` tuple of ``PolicyProcessorPipeline`` objects.
    """
    input_steps = [
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
    ]
    output_steps = [
        DeviceProcessorStep(device="cpu"),
    ]

    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=input_steps,
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=output_steps,
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor
