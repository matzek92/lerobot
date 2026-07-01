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
"""Octo policy adapter for LeRobot.

Octo (https://octo-models.github.io/) is a JAX/Flax-based generalist robot policy trained on
800k Open X-Embodiment trajectories.  This module wraps the JAX-based ``OctoModel`` so that it
conforms to LeRobot's ``PreTrainedPolicy`` interface and can be used directly from
``make_policy``, the eval loop, and the LeRobot CLI.

Because Octo is built on JAX rather than PyTorch the adapter:
- Stores the OctoModel as a non-PyTorch attribute (no ``nn.Parameter`` wrappers).
- Converts observations from PyTorch tensors to NumPy arrays on every call.
- Converts predicted actions from JAX/NumPy arrays back to PyTorch tensors.
- Overrides ``from_pretrained`` / ``_save_pretrained`` to skip safetensors I/O.
- Raises ``NotImplementedError`` for training-related methods.

Requirements (install separately)::

    git clone https://github.com/octo-models/octo && cd octo
    pip install -e .
    pip install -r requirements.txt
    pip install "jax[cpu]"   # or "jax[cuda12_pip]" for GPU
"""

import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.utils.constants import OBS_IMAGES

from ..pretrained import PreTrainedPolicy
from .configuration_octo import OctoConfig

logger = logging.getLogger(__name__)

_OCTO_INSTALL_HINT = (
    "Octo and JAX are required to use OctoPolicy. "
    "Install them with:\n"
    "  git clone https://github.com/octo-models/octo && cd octo\n"
    "  pip install -e .\n"
    "  pip install -r requirements.txt\n"
    "  pip install 'jax[cpu]'   # or 'jax[cuda12_pip]' for GPU"
)


class OctoPolicy(PreTrainedPolicy):
    """LeRobot adapter for the Octo generalist robot policy.

    This class wraps the JAX-based ``OctoModel`` and exposes the standard LeRobot
    ``PreTrainedPolicy`` interface.  It is **inference-only** — training and fine-tuning
    must be performed using the Octo-native scripts (see https://github.com/octo-models/octo).

    Typical usage via LeRobot factory::

        from lerobot.policies.factory import make_policy
        from lerobot.policies.octo import OctoConfig

        cfg = OctoConfig(
            octo_checkpoint="hf://rail-berkeley/octo-small-1.5",
            task_text="pick up the spoon",
            unnorm_key="bridge_dataset",
            primary_image_key="observation.images.top",
            n_action_steps=4,
        )
        policy = make_policy(cfg, ds_meta=...)
        action = policy.select_action(obs_batch)

    Direct instantiation::

        from lerobot.policies.octo import OctoConfig, OctoPolicy

        cfg = OctoConfig(octo_checkpoint="hf://rail-berkeley/octo-small-1.5")
        policy = OctoPolicy(cfg)
        action = policy.select_action(obs_batch)
    """

    config_class = OctoConfig
    name = "octo"

    def __init__(self, config: OctoConfig, **kwargs):
        super().__init__(config)
        self.config = config

        # JAX model handle — NOT stored as a PyTorch parameter.
        self._octo_model: Any | None = None
        self._rng: Any | None = None  # JAX PRNGKey

        # Per-key deque buffers that accumulate image frames over time.
        # Keys match Octo's expected keys: "image_primary", "image_wrist", …
        self._obs_history: dict[str, deque] = {}

        # Queue of pre-computed actions to step through between chunk re-queries.
        self._action_queue: deque = deque([], maxlen=config.n_action_steps)

        # Load the Octo JAX model.
        if config.octo_checkpoint:
            self._load_octo_model(config.octo_checkpoint)

        self.reset()

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------

    def _load_octo_model(self, checkpoint_path: str) -> None:
        """Load the Octo model from a checkpoint path.

        Args:
            checkpoint_path: HuggingFace Hub path (``"hf://…"``) or local directory.
        """
        try:
            import jax
            from octo.model.octo_model import OctoModel
        except ImportError as exc:
            raise ImportError(_OCTO_INSTALL_HINT) from exc

        logger.info("Loading Octo model from '%s' …", checkpoint_path)
        self._octo_model = OctoModel.load_pretrained(checkpoint_path)
        self._rng = jax.random.PRNGKey(self.config.seed)
        logger.info("Octo model loaded successfully.")

    # ------------------------------------------------------------------
    # PreTrainedPolicy overrides — save / load
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        pretrained_name_or_path,
        *,
        config=None,
        **kwargs,
    ) -> "OctoPolicy":
        """Load an OctoPolicy from a saved config directory.

        Unlike other LeRobot policies, OctoPolicy does **not** store its weights in
        safetensors format.  The Octo JAX checkpoint is loaded from the path stored in
        ``config.octo_checkpoint``.  This method only loads the LeRobot config; the JAX
        model is loaded inside ``__init__``.

        Args:
            pretrained_name_or_path: Path to a directory or HuggingFace repo containing a
                ``config.json`` produced by ``OctoPolicy.save_pretrained()``.
            config: Optional pre-loaded ``OctoConfig``.  If ``None`` the config is read
                from ``pretrained_name_or_path``.
        """
        from lerobot.configs import PreTrainedConfig  # avoid circular at module level

        if config is None:
            _safe_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in {
                    "force_download",
                    "resume_download",
                    "proxies",
                    "token",
                    "cache_dir",
                    "local_files_only",
                    "revision",
                }
            }
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=str(pretrained_name_or_path),
                **_safe_kwargs,
            )
        return cls(config=config)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save the OctoPolicy configuration.

        Only the LeRobot config (``config.json``) is saved here.  The Octo JAX checkpoint
        is **not** copied — it is referenced by ``config.octo_checkpoint`` and loaded on
        demand.  To persist the checkpoint locally, run ``model.save_pretrained(…)`` on
        the underlying ``self._octo_model`` directly.
        """
        self.config._save_pretrained(save_directory)

    # ------------------------------------------------------------------
    # PreTrainedPolicy interface — training (not supported)
    # ------------------------------------------------------------------

    def get_optim_params(self) -> dict:
        raise NotImplementedError(
            "OctoPolicy is inference-only.  For fine-tuning use the Octo-native scripts: "
            "https://github.com/octo-models/octo"
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        raise NotImplementedError(
            "OctoPolicy does not support training via the LeRobot forward pass.  "
            "For fine-tuning use the Octo-native scripts: https://github.com/octo-models/octo"
        )

    # ------------------------------------------------------------------
    # PreTrainedPolicy interface — reset & inference
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset episode state.

        Clears the action queue and the per-key observation history buffers.  Call this
        at the start of every episode (or whenever the environment is reset).
        """
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._obs_history = {}

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Return one action to run in the environment.

        Maintains an internal action queue.  When the queue is empty it calls
        ``predict_action_chunk`` and refills the queue with up to ``n_action_steps``
        actions from the predicted chunk.

        Args:
            batch: Preprocessed observation dict from the LeRobot eval loop.

        Returns:
            Tensor of shape ``(batch_size, action_dim)``.
        """
        self.eval()

        if len(self._action_queue) == 0:
            # actions: (batch_size, action_horizon, action_dim)
            actions = self.predict_action_chunk(batch)
            # Keep only the first n_action_steps per chunk.
            actions = actions[:, : self.config.n_action_steps]
            # Transpose → (n_action_steps, batch_size, action_dim) for queue iteration.
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions using the Octo JAX model.

        Converts the LeRobot observation batch to Octo's expected NumPy format, runs the
        JAX model, and converts the result back to a PyTorch tensor.

        Args:
            batch: Preprocessed observation dict from the LeRobot eval loop.  Expected to
                contain image tensors of shape ``(batch_size, C, H, W)``.

        Returns:
            Tensor of shape ``(batch_size, action_horizon, action_dim)``.
        """
        if self._octo_model is None:
            raise RuntimeError(
                "Octo model not loaded.  Set ``config.octo_checkpoint`` to a valid "
                "checkpoint path and re-instantiate the policy."
            )

        try:
            import jax
        except ImportError as exc:
            raise ImportError(_OCTO_INSTALL_HINT) from exc

        # -- Build Octo-formatted observation dict --
        primary_key = self._resolve_primary_image_key(batch)
        wrist_key = self.config.wrist_image_key

        octo_obs = self._build_octo_observation(batch, primary_key, wrist_key)

        # -- Build task dict from text instruction --
        task = self._build_octo_task(batch)

        # -- Advance JAX PRNG key --
        self._rng, sample_rng = jax.random.split(self._rng)

        # -- Gather unnormalization statistics (if requested) --
        unnorm_stats = None
        if self.config.unnorm_key is not None:
            try:
                unnorm_stats = self._octo_model.dataset_statistics[self.config.unnorm_key][
                    "action"
                ]
            except KeyError as exc:
                available = list(self._octo_model.dataset_statistics.keys())
                raise KeyError(
                    f"unnorm_key '{self.config.unnorm_key}' not found in Octo dataset_statistics. "
                    f"Available keys: {available}"
                ) from exc

        # -- Run JAX inference --
        actions_jax = self._octo_model.sample_actions(
            octo_obs,
            task,
            unnormalization_statistics=unnorm_stats,
            rng=sample_rng,
            argmax=self.config.argmax,
        )

        # -- Convert JAX → PyTorch: (batch_size, action_horizon, action_dim) --
        actions_np = np.asarray(actions_jax)
        return torch.from_numpy(actions_np).float()

    # ------------------------------------------------------------------
    # Private helpers: observation conversion
    # ------------------------------------------------------------------

    def _resolve_primary_image_key(self, batch: dict[str, Tensor]) -> str | None:
        """Determine which batch key maps to Octo's ``image_primary`` slot."""
        cfg_key = self.config.primary_image_key
        if cfg_key and cfg_key in batch:
            return cfg_key
        # Auto-detect: use the first key that looks like an image observation.
        for key in batch:
            if key.startswith(OBS_IMAGES) or (
                "image" in key.lower() and isinstance(batch[key], Tensor) and batch[key].ndim >= 3
            ):
                return key
        return None

    def _build_octo_observation(
        self,
        batch: dict[str, Tensor],
        primary_key: str | None,
        wrist_key: str | None,
    ) -> dict:
        """Build the Octo observation dict from a LeRobot batch.

        Each image tensor ``(B, C, H, W)`` is converted to ``(B, H, W, C)`` uint8 NumPy
        and pushed into an observation history deque of length ``window_size``.

        Returns a dict ready to pass to ``OctoModel.sample_actions``, with shapes:
        - ``"image_primary"``: ``(B, window_size, H, W, C)``
        - ``"image_wrist"`` (optional): ``(B, window_size, H, W, C)``
        - ``"timestep_pad_mask"``: ``(B, window_size)`` bool
        """
        batch_size = _get_batch_size(batch)
        window_size = self.config.window_size
        octo_obs: dict[str, Any] = {}

        # Primary camera
        if primary_key and primary_key in batch:
            img_np = _tensor_to_octo_image(batch[primary_key], self.config.image_size)
            self._push_history("image_primary", img_np)
            octo_obs["image_primary"] = self._history_to_array(
                "image_primary", batch_size, window_size
            )

        # Wrist camera (optional)
        if wrist_key and wrist_key in batch:
            wrist_np = _tensor_to_octo_image(batch[wrist_key], self.config.image_size)
            self._push_history("image_wrist", wrist_np)
            octo_obs["image_wrist"] = self._history_to_array(
                "image_wrist", batch_size, window_size
            )

        # timestep_pad_mask: True = valid, False = padding (unavailable history).
        any_key = "image_primary" if "image_primary" in octo_obs else next(iter(octo_obs), None)
        n_available = len(self._obs_history.get(any_key or "", []))
        timestep_pad_mask = np.zeros((batch_size, window_size), dtype=bool)
        if n_available > 0:
            # Most-recent `n_available` slots are valid (right-aligned).
            start = max(0, window_size - n_available)
            timestep_pad_mask[:, start:] = True
        octo_obs["timestep_pad_mask"] = timestep_pad_mask

        return octo_obs

    def _build_octo_task(self, batch: dict[str, Tensor]) -> dict:
        """Build the Octo task dict from the batch or the config default."""
        batch_size = _get_batch_size(batch)

        # Support an optional "task.language_instruction" key in the batch.
        raw = batch.get("task.language_instruction", None)
        if raw is not None:
            if isinstance(raw, (list, tuple)):
                texts = [str(t) for t in raw]
            elif isinstance(raw, Tensor):
                texts = [str(t.item()) for t in raw]
            else:
                texts = [str(raw)] * batch_size
        elif self.config.task_text:
            texts = [self.config.task_text] * batch_size
        else:
            texts = [""] * batch_size

        return self._octo_model.create_tasks(texts=texts)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Private helpers: history management
    # ------------------------------------------------------------------

    def _push_history(self, key: str, img_np: np.ndarray) -> None:
        """Push a new frame into the history buffer for *key*."""
        if key not in self._obs_history:
            self._obs_history[key] = deque(maxlen=self.config.window_size)
        self._obs_history[key].append(img_np)

    def _history_to_array(self, key: str, batch_size: int, window_size: int) -> np.ndarray:
        """Return a ``(B, T, H, W, C)`` array from the deque for *key*.

        When fewer than ``window_size`` frames are available, the oldest frame is repeated
        to fill the leading (padding) positions.
        """
        history = list(self._obs_history.get(key, []))

        if not history:
            # No frames at all: return zeros.
            h, w = self.config.image_size
            return np.zeros((batch_size, window_size, h, w, 3), dtype=np.uint8)

        # Pad at the front by repeating the earliest available frame.
        while len(history) < window_size:
            history = [history[0]] + history

        # Take the last `window_size` frames.
        history = history[-window_size:]

        # Each element is ``(B, H, W, C)``; stack along a new time axis.
        return np.stack(history, axis=1)  # → (B, T, H, W, C)


# ------------------------------------------------------------------
# Module-level helpers (pure functions, no OctoPolicy state)
# ------------------------------------------------------------------


def _get_batch_size(batch: dict[str, Any]) -> int:
    """Return the batch size inferred from the first tensor in *batch*."""
    for v in batch.values():
        if isinstance(v, Tensor) and v.ndim >= 1:
            return v.shape[0]
    return 1


def _tensor_to_octo_image(img_tensor: Tensor, target_size: tuple[int, int]) -> np.ndarray:
    """Convert a LeRobot image tensor to an Octo-compatible NumPy array.

    Args:
        img_tensor: Float or uint8 tensor of shape ``(B, C, H, W)``.
        target_size: ``(H, W)`` to resize to (e.g. ``(256, 256)``).

    Returns:
        NumPy array of shape ``(B, H, W, C)`` with dtype ``uint8`` and values in ``[0, 255]``.
    """
    # Move to CPU and convert to NumPy.
    img_np = img_tensor.cpu().numpy()  # (B, C, H, W)

    # CHW → HWC
    img_np = img_np.transpose(0, 2, 3, 1)  # (B, H, W, C)

    # Float [0, 1] → uint8 [0, 255]
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

    # Resize if necessary.
    target_h, target_w = target_size
    if img_np.shape[1] != target_h or img_np.shape[2] != target_w:
        try:
            import cv2  # opencv is a common transitive dep; try it first
            batch_size = img_np.shape[0]
            resized = np.empty((batch_size, target_h, target_w, img_np.shape[3]), dtype=np.uint8)
            for i in range(batch_size):
                resized[i] = cv2.resize(
                    img_np[i], (target_w, target_h), interpolation=cv2.INTER_LANCZOS4
                )
            img_np = resized
        except ImportError:
            # Fall back to a simple slice/repeat resize using NumPy if cv2 is absent.
            from PIL import Image  # pillow is always available in LeRobot

            batch_size = img_np.shape[0]
            resized = np.empty((batch_size, target_h, target_w, img_np.shape[3]), dtype=np.uint8)
            for i in range(batch_size):
                pil_img = Image.fromarray(img_np[i])
                resized[i] = np.array(pil_img.resize((target_w, target_h), Image.LANCZOS))
            img_np = resized

    return img_np  # (B, H, W, C) uint8
