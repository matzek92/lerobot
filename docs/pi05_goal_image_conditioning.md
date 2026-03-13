# Pi0.5 Goal Image Conditioning

This document explains how to enable and use per-episode **goal image conditioning**
for the `lerobot/pi05_base` policy (Option A implementation).

## Overview

Goal image conditioning allows pi0.5 to receive an additional image that
represents the desired final state of the task.  The goal image is encoded by a
lightweight **CLIP** (or simple CNN) vision encoder and injected into the prefix
sequence as an extra token alongside the regular observation images and the
language task description.

```
goal image  →  GoalImageEncoder (frozen CLIP)  →  goal_proj (linear)  →  goal token
                                                                              ↓
                                              [img tokens | lang tokens | goal token] → PI05 model
```

Gradients flow only through `goal_proj` (and optionally through the encoder when
`finetune_goal_encoder=True`).  The core PI05 weights are unchanged.

---

## Dataset schema

Add a per-episode goal image to your dataset under the key **`task.goal_image`**
(or a key of your choice; see [configuration](#configuration)).

```python
# HuggingFace dataset feature definition
from datasets import Image

features = {
    # … existing features …
    "task.goal_image": Image(),
}
```

The image must be available at every timestep sample of an episode.  You can
achieve this by either:

- repeating it in the per-step record (simplest), or
- fetching the last frame of each episode and storing it once per episode in a
  side-file, then looking it up in your data-loading pipeline.

Images are expected to be in **RGB, [0, 1] float32** (or **[0, 255] uint8**,
which is automatically rescaled) with shape `(C, H, W)` in CHW format.
Any spatial size is accepted—images are resized to 224 × 224 inside the encoder.

---

## Configuration

Enable goal conditioning by adding the following fields to your policy config
(Python API or JSON config file):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_goal_image` | `bool` | `False` | Enable goal image conditioning |
| `goal_image_key` | `str` | `"task.goal_image"` | Batch key for the goal image tensor |
| `goal_encoder_type` | `str` | `"clip"` | Encoder backend: `"clip"` or `"simple"` |
| `goal_encoder_model_name` | `str` | `"openai/clip-vit-base-patch32"` | HuggingFace model name (CLIP only) |
| `goal_projection_dim` | `int` | `512` | Embedding dimension output by the encoder |
| `finetune_goal_encoder` | `bool` | `False` | Unfreeze encoder weights during training |

All other config fields remain **unchanged**, so existing configurations
continue to work without modification.

---

## Training

### Command-line (minimal example)

```bash
lerobot-train \
  --policy.type pi05 \
  --policy.use_goal_image true \
  --policy.goal_image_key task.goal_image \
  --policy.goal_encoder_type clip \
  --policy.goal_encoder_model_name openai/clip-vit-base-patch32 \
  --policy.goal_projection_dim 512 \
  --dataset.repo_id <HF_USER>/<DATASET_WITH_GOAL_IMAGES> \
  --output-dir runs/pi05_goal_conditioned
```

### Python API

```python
from lerobot.policies.pi05 import PI05Config, PI05Policy, make_pi05_pre_post_processors
from lerobot.configs.types import FeatureType, PolicyFeature

config = PI05Config(
    use_goal_image=True,
    goal_image_key="task.goal_image",
    goal_encoder_type="clip",
    goal_encoder_model_name="openai/clip-vit-base-patch32",
    goal_projection_dim=512,
    finetune_goal_encoder=False,  # keep encoder frozen
)

config.input_features = {
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
    "observation.images.main": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
}
config.output_features = {
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
}

policy = PI05Policy(config)
preprocessor, postprocessor = make_pi05_pre_post_processors(config=config, dataset_stats=...)
```

During training, the batch dict must contain the goal image under the configured key:

```python
batch = {
    "observation.state":         ...,   # [B, state_dim]
    "observation.images.main":   ...,   # [B, 3, 224, 224]
    "action":                    ...,   # [B, chunk_size, action_dim]
    "task":                      [...], # list[str]
    "task.goal_image":           ...,   # [B, 3, 224, 224]  ← NEW
}
batch = preprocessor(batch)
loss, loss_dict = policy.forward(batch)
```

### Using the simple CNN encoder (no CLIP download)

Set `goal_encoder_type="simple"` to use the built-in lightweight CNN fallback.
This is useful for quick prototyping or in environments without internet access.
The projection dimension defaults to `goal_projection_dim` in this case too.

```bash
lerobot-train \
  --policy.type pi05 \
  --policy.use_goal_image true \
  --policy.goal_encoder_type simple \
  --policy.goal_projection_dim 256 \
  ...
```

---

## Inference / rollout

Pass the goal image in the observation dict at every timestep:

```python
import torch
from PIL import Image

goal_image = Image.open("goal_state.png").convert("RGB")
goal_tensor = torch.tensor(np.array(goal_image), dtype=torch.float32).permute(2, 0, 1) / 255.0

observation = {
    "observation.state":       ...,
    "observation.images.main": ...,
    "task":                    ["place object in bin"],
    "task.goal_image":         goal_tensor,  # [3, H, W] – batch dim added automatically
}
observation = preprocessor(observation)
action = policy.select_action(observation)
action = postprocessor(action)
```

The policy adds the batch dimension automatically when the goal image is 3-D
(`[C, H, W]`), so no manual unsqueeze is needed.

---

## Error handling

If `use_goal_image=True` but the goal image key is absent from the batch, a clear
error is raised:

```
ValueError: Goal image key 'task.goal_image' not found in batch.
Available keys: ['action', 'observation.images.main', 'observation.state', 'task', ...].
Ensure the dataset provides goal images under this key, or disable goal conditioning
by setting use_goal_image=False in the policy config.
```

---

## Notes

* When `use_goal_image=False` (the default), **behaviour is identical** to the
  original pi0.5 implementation.  No extra computation or memory is used.
* The encoder weights are downloaded via the standard HuggingFace cache
  (`~/.cache/huggingface/hub/`).  Set `HF_HOME` to customise the cache location.
* The goal token is appended **after** the language tokens in the prefix
  sequence so that its position is stable regardless of the number of cameras.
* Only the `goal_proj` layer (and optionally the encoder) receives gradient
  updates during training.  The core PaliGemma + action-expert weights are
  controlled by the existing `freeze_vision_encoder` / `train_expert_only` flags.
