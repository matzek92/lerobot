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

import logging
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig, DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("vision_dit")
@dataclass
class VisionDiTConfig(PreTrainedConfig):
    """Configuration for the vision-only Diffusion Transformer (DiT) policy.

    A transformer-based diffusion / flow matching policy conditioned on robot state and
    image observations only (no language). The image encoder is a freely configurable
    torchvision CNN backbone (e.g. ResNet18).
    """

    n_obs_steps: int = 2
    horizon: int = 32
    n_action_steps: int = 24

    # Objective Selection
    objective: str = "diffusion"  # "diffusion" or "flow_matching"

    # --- Diffusion-specific ---
    noise_scheduler_type: str = "DDPM"  # "DDPM" or "DDIM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"  # "epsilon" or "sample"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    num_inference_steps: int | None = None

    # --- Flow Matching-specific ---
    sigma_min: float = 0.0
    num_integration_steps: int = 100
    integration_method: str = "euler"  # "euler" or "rk4"
    timestep_sampling_strategy: str = "beta"  # "uniform" or "beta"
    timestep_sampling_s: float = 0.999
    timestep_sampling_alpha: float = 1.5
    timestep_sampling_beta: float = 1.0

    # Transformer Architecture
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    use_positional_encoding: bool = False
    timestep_embed_dim: int = 256
    use_rope: bool = True
    rope_base: float = 10000.0

    # Vision Encoder (configurable torchvision CNN)
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_separate_rgb_encoder_per_camera: bool = False
    vision_encoder_lr_multiplier: float = 0.1
    image_resize_shape: tuple[int, int] | None = None
    image_crop_shape: tuple[int, int] | None = None
    image_crop_is_random: bool = True

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Training/Optimizer
    optimizer_lr: float = 2e-5
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 0
    do_mask_loss_for_padding: bool = False

    # Auto-calculated
    drop_n_last_frames: int | None = None

    # Supported torchvision CNN backbone prefixes. The model is loaded via
    # ``torchvision.models.get_model`` and its final classification head is stripped.
    _SUPPORTED_BACKBONE_PREFIXES: tuple[str, ...] = (
        "resnet",
        "resnext",
        "wide_resnet",
        "regnet",
        "convnext",
        "efficientnet",
        "mobilenet",
        "mnasnet",
        "shufflenet",
        "densenet",
        "vgg",
        "squeezenet",
    )

    def __post_init__(self):
        super().__post_init__()

        if self.drop_n_last_frames is None:
            self.drop_n_last_frames = self.horizon - self.n_action_steps - self.n_obs_steps + 1

        self._validate()

    def _validate(self):
        if self.objective not in ["diffusion", "flow_matching"]:
            raise ValueError(f"objective must be 'diffusion' or 'flow_matching', got '{self.objective}'")

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be between 0.0 and 1.0")

        if not any(self.vision_backbone.startswith(p) for p in self._SUPPORTED_BACKBONE_PREFIXES):
            raise ValueError(
                f"vision_backbone '{self.vision_backbone}' is not in the supported torchvision "
                f"backbone families: {self._SUPPORTED_BACKBONE_PREFIXES}"
            )

        if (
            self.image_resize_shape
            and self.image_crop_shape
            and (
                self.image_crop_shape[0] > self.image_resize_shape[0]
                or self.image_crop_shape[1] > self.image_resize_shape[1]
            )
        ):
            logging.warning(
                "image_crop_shape %s must be <= image_resize_shape %s; disabling cropping.",
                self.image_crop_shape,
                self.image_resize_shape,
            )
            self.image_crop_shape = None

        if self.objective == "diffusion":
            if self.noise_scheduler_type not in ["DDPM", "DDIM"]:
                raise ValueError(
                    f"noise_scheduler_type must be 'DDPM' or 'DDIM', got {self.noise_scheduler_type}"
                )
            if self.prediction_type not in ["epsilon", "sample"]:
                raise ValueError(f"prediction_type must be 'epsilon' or 'sample', got {self.prediction_type}")
            if self.num_train_timesteps <= 0:
                raise ValueError(f"num_train_timesteps must be positive, got {self.num_train_timesteps}")
            if not (0.0 <= self.beta_start <= self.beta_end <= 1.0):
                raise ValueError(f"Invalid beta values: {self.beta_start}, {self.beta_end}")

        elif self.objective == "flow_matching":
            if not (0.0 <= self.sigma_min <= 1.0):
                raise ValueError(f"sigma_min must be in [0, 1], got {self.sigma_min}")
            if self.num_integration_steps <= 0:
                raise ValueError(f"num_integration_steps must be positive, got {self.num_integration_steps}")
            if self.integration_method not in ["euler", "rk4"]:
                raise ValueError(
                    f"integration_method must be 'euler' or 'rk4', got {self.integration_method}"
                )
            if self.timestep_sampling_strategy not in ["uniform", "beta"]:
                raise ValueError("timestep_sampling_strategy must be 'uniform' or 'beta'")
            if self.timestep_sampling_strategy == "beta":
                if not (0.0 < self.timestep_sampling_s <= 1.0):
                    raise ValueError(f"timestep_sampling_s must be in (0, 1], got {self.timestep_sampling_s}")
                if self.timestep_sampling_alpha <= 0:
                    raise ValueError("timestep_sampling_alpha must be positive")
                if self.timestep_sampling_beta <= 0:
                    raise ValueError("timestep_sampling_beta must be positive")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if self.image_crop_shape is not None:
            for key, image_ft in self.image_features.items():
                effective_h, effective_w = (
                    self.image_resize_shape
                    if self.image_resize_shape is not None
                    else (image_ft.shape[1], image_ft.shape[2])
                )
                if self.image_crop_shape[0] > effective_h or self.image_crop_shape[1] > effective_w:
                    logging.warning(
                        "image_crop_shape %s doesn't fit within effective image shape (%s, %s) for '%s'; disabling cropping.",
                        self.image_crop_shape,
                        effective_h,
                        effective_w,
                        key,
                    )
                    self.image_crop_shape = None
                    break

        if len(self.image_features) > 0:
            first_key, first_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_ft.shape:
                    raise ValueError(
                        f"Image '{key}' shape {image_ft.shape} != '{first_key}' shape {first_ft.shape}"
                    )
        else:
            raise ValueError(
                "VisionDiTConfig requires at least one image feature in `input_features`."
            )

    @property
    def is_diffusion(self) -> bool:
        return self.objective == "diffusion"

    @property
    def is_flow_matching(self) -> bool:
        return self.objective == "flow_matching"

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
