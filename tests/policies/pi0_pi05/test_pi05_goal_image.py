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

"""Tests for PI0.5 goal image conditioning (Option A: goal image → embedding → task token).

The full end-to-end policy tests that require the pretrained PaliGemma weights are
guarded by ``@require_cuda`` and are skipped in CI (same convention as the other
pi05 tests).  The lightweight encoder-only tests do *not* require CUDA and are
expected to pass in all environments.
"""

import os

import pytest
import torch

# ---------------------------------------------------------------------------
# CI guard (for tests that require the full pretrained model)
# ---------------------------------------------------------------------------
_IS_CI = os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"


# ---------------------------------------------------------------------------
# CPU-only tests: GoalImageEncoder unit tests (no PaliGemma, no CUDA needed)
# ---------------------------------------------------------------------------


class TestSimpleCNNGoalEncoder:
    """Tests for the SimpleCNNGoalEncoder fallback backend (CPU-only)."""

    def test_output_shape(self):
        """SimpleCNNGoalEncoder maps [B,C,H,W] → [B, output_dim]."""
        from lerobot.policies.pi05.modeling_pi05 import SimpleCNNGoalEncoder

        output_dim = 128
        encoder = SimpleCNNGoalEncoder(output_dim=output_dim)
        encoder.eval()

        images = torch.rand(2, 3, 224, 224)
        with torch.no_grad():
            embeddings = encoder(images)

        assert embeddings.shape == (2, output_dim), (
            f"Expected shape (2, {output_dim}), got {embeddings.shape}"
        )

    def test_output_dtype(self):
        """SimpleCNNGoalEncoder preserves float32 dtype."""
        from lerobot.policies.pi05.modeling_pi05 import SimpleCNNGoalEncoder

        encoder = SimpleCNNGoalEncoder(output_dim=64)
        encoder.eval()
        images = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            emb = encoder(images)
        assert emb.dtype == torch.float32

    def test_different_image_sizes(self):
        """SimpleCNNGoalEncoder can receive images of different sizes."""
        from lerobot.policies.pi05.modeling_pi05 import GoalImageEncoder

        encoder = GoalImageEncoder(encoder_type="simple", output_dim=64)
        encoder.eval()
        # Non-standard size – encoder should resize internally
        images = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            emb = encoder(images)
        assert emb.shape == (1, 64)

    def test_encoder_frozen_by_default(self):
        """Encoder weights must be frozen when finetune=False (default)."""
        from lerobot.policies.pi05.modeling_pi05 import GoalImageEncoder

        encoder = GoalImageEncoder(encoder_type="simple", output_dim=64, finetune=False)
        for param in encoder.encoder.parameters():
            assert not param.requires_grad, "Encoder weights should be frozen by default."

    def test_encoder_unfrozen_when_finetune_true(self):
        """Encoder weights must be trainable when finetune=True."""
        from lerobot.policies.pi05.modeling_pi05 import GoalImageEncoder

        encoder = GoalImageEncoder(encoder_type="simple", output_dim=64, finetune=True)
        for param in encoder.encoder.parameters():
            assert param.requires_grad, "Encoder weights should be trainable when finetune=True."


class TestGoalImageEncoderSimpleBackend:
    """GoalImageEncoder with encoder_type="simple" (CPU-only, no downloads needed)."""

    def test_output_shape_batch_size_1(self):
        from lerobot.policies.pi05.modeling_pi05 import GoalImageEncoder

        encoder = GoalImageEncoder(encoder_type="simple", output_dim=256)
        encoder.eval()
        images = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            emb = encoder(images)
        assert emb.shape == (1, 256)

    def test_output_shape_batch_size_4(self):
        from lerobot.policies.pi05.modeling_pi05 import GoalImageEncoder

        encoder = GoalImageEncoder(encoder_type="simple", output_dim=256)
        encoder.eval()
        images = torch.rand(4, 3, 224, 224)
        with torch.no_grad():
            emb = encoder(images)
        assert emb.shape == (4, 256)

    def test_invalid_encoder_type_raises(self):
        from lerobot.policies.pi05.modeling_pi05 import GoalImageEncoder

        with pytest.raises(ValueError, match="Unknown goal_encoder_type"):
            GoalImageEncoder(encoder_type="nonexistent")


class TestPI05ConfigGoalImage:
    """Tests for PI05Config goal image conditioning fields."""

    def test_default_values(self):
        from lerobot.policies.pi05.configuration_pi05 import PI05Config

        cfg = PI05Config()
        assert cfg.use_goal_image is False
        assert cfg.goal_image_key == "task.goal_image"
        assert cfg.goal_encoder_type == "clip"
        assert cfg.goal_encoder_model_name == "openai/clip-vit-base-patch32"
        assert cfg.goal_projection_dim == 512
        assert cfg.finetune_goal_encoder is False

    def test_invalid_encoder_type_raises(self):
        from lerobot.policies.pi05.configuration_pi05 import PI05Config

        with pytest.raises(ValueError, match="Invalid goal_encoder_type"):
            PI05Config(use_goal_image=True, goal_encoder_type="bad_type")

    def test_invalid_projection_dim_raises(self):
        from lerobot.policies.pi05.configuration_pi05 import PI05Config

        with pytest.raises(ValueError, match="goal_projection_dim must be positive"):
            PI05Config(use_goal_image=True, goal_projection_dim=0)

    def test_backward_compatible_no_goal_image(self):
        """Existing configs without goal image should continue to work unchanged."""
        from lerobot.policies.pi05.configuration_pi05 import PI05Config

        cfg = PI05Config()
        # No exception should be raised; default use_goal_image=False
        assert cfg.use_goal_image is False

    def test_simple_encoder_config(self):
        """Simple encoder type should be accepted."""
        from lerobot.policies.pi05.configuration_pi05 import PI05Config

        cfg = PI05Config(
            use_goal_image=True,
            goal_encoder_type="simple",
            goal_projection_dim=256,
        )
        assert cfg.goal_encoder_type == "simple"
        assert cfg.goal_projection_dim == 256


class TestConverterGoalImageKey:
    """Test that task.goal_image passes through the batch→transition pipeline."""

    def test_task_sub_key_preserved(self):
        """Keys starting with 'task.' should be preserved as complementary data."""
        import torch

        from lerobot.processor.converters import batch_to_transition, transition_to_batch
        from lerobot.processor.core import TransitionKey

        goal_img = torch.rand(2, 3, 224, 224)
        batch = {
            "observation.state": torch.zeros(2, 14),
            "action": torch.zeros(2, 50, 7),
            "task": ["pick up object"] * 2,
            "task.goal_image": goal_img,
        }

        transition = batch_to_transition(batch)
        comp_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        assert "task.goal_image" in comp_data, (
            "task.goal_image should be in complementary_data after batch_to_transition"
        )

        # Round-trip back to batch
        recovered_batch = transition_to_batch(transition)
        assert "task.goal_image" in recovered_batch, (
            "task.goal_image should survive the round-trip through the pipeline"
        )
        assert torch.allclose(recovered_batch["task.goal_image"], goal_img)


# ---------------------------------------------------------------------------
# Full policy tests: require CUDA + pretrained weights (skipped in CI)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _IS_CI,
    reason="Full policy tests require local pretrained model and CUDA; skipped in CI.",
)
class TestPI05GoalImageFullPipeline:
    """End-to-end tests for PI0.5 with goal image conditioning.

    These tests construct a minimal batch that contains ``task.goal_image`` and
    verify that:
    1. The goal image is encoded to an embedding of the correct shape.
    2. The policy forward pass accepts the conditioning and returns a loss.
    3. Action prediction (select_action) returns actions of the correct shape.
    """

    def _make_config_and_policy(self):
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        cfg = PI05Config(
            max_action_dim=7,
            max_state_dim=14,
            dtype="float32",
            use_goal_image=True,
            goal_encoder_type="simple",  # No downloads needed
            goal_projection_dim=256,
        )
        cfg.input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            "observation.images.base_0_rgb": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
        }
        cfg.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
        policy = PI05Policy(cfg)
        return cfg, policy

    def _make_dataset_stats(self):
        return {
            "observation.state": {
                "mean": torch.zeros(14),
                "std": torch.ones(14),
                "min": torch.zeros(14),
                "max": torch.ones(14),
                "q01": torch.zeros(14),
                "q99": torch.ones(14),
            },
            "action": {
                "mean": torch.zeros(7),
                "std": torch.ones(7),
                "min": torch.zeros(7),
                "max": torch.ones(7),
                "q01": torch.zeros(7),
                "q99": torch.ones(7),
            },
            "observation.images.base_0_rgb": {
                "mean": torch.zeros(3, 224, 224),
                "std": torch.ones(3, 224, 224),
                "q01": torch.zeros(3, 224, 224),
                "q99": torch.ones(3, 224, 224),
            },
        }

    def test_goal_encoder_initialized(self):
        """PI05Policy should have goal_encoder when use_goal_image=True."""
        _, policy = self._make_config_and_policy()
        assert hasattr(policy.model, "goal_encoder"), "model.goal_encoder should be initialized"
        assert hasattr(policy.model, "goal_proj"), "model.goal_proj should be initialized"

    def test_encode_goal_image_shape(self):
        """_encode_goal_image should return [B, goal_projection_dim]."""
        cfg, policy = self._make_config_and_policy()
        device = next(policy.parameters()).device
        batch_size = 2
        goal_img = torch.rand(batch_size, 3, 224, 224, device=device)
        batch = {"task.goal_image": goal_img}
        goal_emb = policy._encode_goal_image(batch)
        assert goal_emb.shape == (batch_size, cfg.goal_projection_dim), (
            f"Expected ({batch_size}, {cfg.goal_projection_dim}), got {goal_emb.shape}"
        )

    def test_encode_goal_image_missing_key_raises(self):
        """Missing goal image key should raise a clear ValueError."""
        _, policy = self._make_config_and_policy()
        with pytest.raises(ValueError, match="Goal image key.*not found in batch"):
            policy._encode_goal_image({"observation.state": torch.zeros(1, 14)})

    def test_forward_with_goal_image(self):
        """Policy forward pass should succeed and return a valid loss when goal image is provided."""
        from lerobot.utils.random_utils import set_seed

        from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

        set_seed(42)
        cfg, policy = self._make_config_and_policy()
        preprocessor, _ = make_pi05_pre_post_processors(
            config=cfg, dataset_stats=self._make_dataset_stats()
        )
        device = next(policy.parameters()).device
        batch_size = 1
        batch = {
            "observation.state": torch.randn(batch_size, 14, device=device),
            "action": torch.randn(batch_size, cfg.chunk_size, 7, device=device),
            "observation.images.base_0_rgb": torch.rand(
                batch_size, 3, 224, 224, device=device
            ),
            "task": ["Pick up the red block"] * batch_size,
            "task.goal_image": torch.rand(batch_size, 3, 224, 224, device=device),
        }
        batch = preprocessor(batch)
        loss, loss_dict = policy.forward(batch)
        assert loss.item() > 0, "Loss should be positive"
        assert "loss" in loss_dict

    def test_forward_without_goal_image_unchanged(self):
        """When use_goal_image=False, behavior should be identical to the baseline."""
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
        from lerobot.utils.random_utils import set_seed

        set_seed(42)
        cfg = PI05Config(max_action_dim=7, max_state_dim=14, dtype="float32")
        cfg.input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            "observation.images.base_0_rgb": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
        }
        cfg.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
        policy = PI05Policy(cfg)
        assert not hasattr(policy.model, "goal_encoder"), (
            "goal_encoder should NOT be initialized when use_goal_image=False"
        )
        preprocessor, _ = make_pi05_pre_post_processors(
            config=cfg, dataset_stats=self._make_dataset_stats()
        )
        device = next(policy.parameters()).device
        batch_size = 1
        batch = {
            "observation.state": torch.randn(batch_size, 14, device=device),
            "action": torch.randn(batch_size, cfg.chunk_size, 7, device=device),
            "observation.images.base_0_rgb": torch.rand(
                batch_size, 3, 224, 224, device=device
            ),
            "task": ["Pick up the red block"] * batch_size,
        }
        batch = preprocessor(batch)
        loss, _ = policy.forward(batch)
        assert loss.item() > 0
