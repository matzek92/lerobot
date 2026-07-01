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

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("octo")
@dataclass
class OctoConfig(PreTrainedConfig):
    """Configuration class for the Octo generalist robot policy adapter.

    Octo (https://octo-models.github.io/) is a JAX/Flax-based transformer-diffusion policy
    from UC Berkeley, pre-trained on 800k Open X-Embodiment robot trajectories. This adapter
    wraps the JAX-based OctoModel inside LeRobot's PreTrainedPolicy interface so that it can
    be instantiated and run through the standard ``make_policy`` factory and eval loop.

    .. warning::
        Octo and JAX must be installed separately (not included in LeRobot's default deps)::

            git clone https://github.com/octo-models/octo && cd octo
            pip install -e .
            pip install -r requirements.txt
            # CPU:
            pip install "jax[cpu]"
            # GPU (CUDA 12):
            pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    .. note::
        Training / fine-tuning is **not** supported through this adapter. For fine-tuning use
        the Octo-native scripts at https://github.com/octo-models/octo.

    Args:
        octo_checkpoint: Path to the Octo model checkpoint. Supports HuggingFace Hub paths
            (e.g., ``"hf://rail-berkeley/octo-small-1.5"``) or local directory paths.
            Available checkpoints on HuggingFace:
            - ``"hf://rail-berkeley/octo-small-1.5"`` (27 M params, default)
            - ``"hf://rail-berkeley/octo-base-1.5"`` (93 M params)
        window_size: Number of consecutive observation frames to pass as history to the model.
            Octo was pre-trained with ``window_size=2``. Using ``window_size=1`` disables
            temporal context (each call is treated independently).
        n_action_steps: Number of actions from the predicted chunk to execute in the
            environment before querying the model again. Must be ≤ ``action_horizon``.
        action_horizon: Size of the action chunk predicted by the model per call.
            For the pre-trained Octo models this is **4**.
        task_text: Default language instruction for the task. Used when no task text is
            provided in the observation batch. Set to an empty string to skip language
            conditioning (goal-image mode only).
        unnorm_key: Key into ``model.dataset_statistics`` used to unnormalize predicted
            actions back to physical units. Set to ``None`` to return z-score normalized
            actions (mean 0, std 1). For the pre-trained Octo checkpoints, valid keys
            include ``"bridge_dataset"``, ``"fractal20220817_data"``, etc. Run
            ``model.get_pretty_spec()`` on the loaded model to see all available keys.
        primary_image_key: Key in the LeRobot observation batch dict for the primary
            (third-person) camera image, e.g. ``"observation.images.top"``.
            If ``None``, the first key that starts with ``"observation.images"`` is used.
        wrist_image_key: Key in the LeRobot observation batch dict for the optional wrist
            camera image, e.g. ``"observation.images.wrist"``. Set to ``None`` (default)
            when no wrist camera is available.
        image_size: ``(H, W)`` to which input images are resized before being passed to
            Octo. The pre-trained models expect **256 × 256** images.
        argmax: If ``True``, use deterministic argmax instead of stochastic diffusion
            sampling when predicting actions.
        seed: Random seed for the JAX PRNG key used during action sampling.
    """

    # Octo-specific settings
    octo_checkpoint: str = "hf://rail-berkeley/octo-small-1.5"
    window_size: int = 1
    n_action_steps: int = 1
    action_horizon: int = 4
    task_text: str = ""
    unnorm_key: str | None = None
    primary_image_key: str | None = None
    wrist_image_key: str | None = None
    image_size: tuple[int, int] = (256, 256)
    argmax: bool = False
    seed: int = 42

    # Octo handles normalization internally; keep this empty so that LeRobot's
    # NormalizerProcessorStep is not applied in the pre-processor pipeline.
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.action_horizon:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) must be ≤ "
                f"`action_horizon` ({self.action_horizon})."
            )
        if self.window_size < 1:
            raise ValueError(f"`window_size` must be ≥ 1. Got {self.window_size}.")

    def get_optimizer_preset(self) -> AdamWConfig:
        # Not used for inference-only; return a sensible default to satisfy the ABC.
        return AdamWConfig()

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        # Octo is flexible about observation keys; no hard validation required here.
        pass

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
