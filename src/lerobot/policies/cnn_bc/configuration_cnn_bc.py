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
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("cnn_bc")
@dataclass
class CNNBCConfig(PreTrainedConfig):
    """Configuration class for the CNN Behavioral Cloning policy.

    This policy uses a ResNet18 backbone to extract image features and feeds them (along with
    optional robot proprioceptive state) through fully connected layers to predict action chunks.

    The parameters you will most likely need to change are the ones which depend on the environment
    and sensors. Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - At least one key starting with "observation.images" is required as input.
        - May optionally include an "observation.state" key for proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes
            the current step only; multiple steps not supported).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of
            the policy. Must be no greater than `chunk_size`.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy.
        normalization_mapping: A dictionary that maps FeatureType strings to NormalizationMode.
        vision_backbone: Name of the torchvision ResNet backbone to use for encoding images.
            Must be one of the ResNet variants (e.g. "resnet18", "resnet34", "resnet50").
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights are used.
        hidden_dims: Sizes of the hidden layers in the fully connected head. Each element creates
            one Linear → ReLU → Dropout block.
        dropout: Dropout probability applied after each hidden layer in the FC head.
        optimizer_lr: Learning rate for all parameters except the backbone.
        optimizer_weight_decay: Weight decay for the AdamW optimizer.
        optimizer_lr_backbone: Learning rate for the backbone parameters.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"

    # Fully connected head architecture.
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.1

    # Training preset.
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. "
                f"Got {self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `n_obs_steps={self.n_obs_steps}`"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("You must provide at least one image among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
