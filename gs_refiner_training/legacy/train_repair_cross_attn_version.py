#!/usr/bin/env python
# coding=utf-8
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

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from torchvision.utils import save_image
import itertools

from diffusers import ModelMixin, ConfigMixin
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from einops import rearrange

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModel
from diffusers import StableUnCLIPImg2ImgPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer
from diffusers import DDPMScheduler, PNDMScheduler
from diffusers.models.embeddings import get_timestep_embedding
import torch
import numpy as np
from tutorial_dataset import MyDataset

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

if is_torch2_available():
    from self_mutual_attention import SelfMutualAttnProcessor2_0 as SelfMutualAttnProcessor
    from diffusers.models.attention_processor import AttnProcessor2_0 as AttnProcessor

if is_wandb_available():
    import wandb

import os
from typing import Sequence
from torch.utils.data import BatchSampler, Sampler, Dataset, RandomSampler


logger = get_logger(__name__, log_level="INFO")

# DATASET_NAME_MAPPING = {
#     "lambdalabs/naruto-blip-captions": ("image", "text"),
# }


class AspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 aspect_ratios: dict,
                 drop_last: bool = False,
                 config=None,
                 valid_num=0,   # take as valid aspect-ratio when sample number >= valid_num
                 **kwargs) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last
        self.ratio_nums_gt = kwargs.get('ratio_nums', None)
        self.config = config
        assert self.ratio_nums_gt
        # buckets for each aspect ratio
        self._aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratios.keys()}
        self.current_available_bucket_keys =  [str(k) for k, v in self.ratio_nums_gt.items() if v >= valid_num]
        logger.warning(f"Using valid_num={valid_num} in config file. Available {len(self.current_available_bucket_keys)} aspect_ratios: {self.current_available_bucket_keys}")

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_info = self.dataset.get_data_info(idx)
            height, width =  data_info['height'], data_info['width']
            ratio =  width / height
            # find the closest aspect ratio
            closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
            if closest_ratio not in self.current_available_bucket_keys:
                continue
            bucket = self._aspect_ratio_buckets[closest_ratio]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        # yield the rest data and reset the buckets
        for bucket in self._aspect_ratio_buckets.values():
            while len(bucket) > 0:
                if len(bucket) <= self.batch_size:
                    if not self.drop_last:
                        yield bucket[:]
                    bucket = []
                else:
                    yield bucket[:self.batch_size]
                    bucket = bucket[self.batch_size:]


def save_model_card(
    args,
    repo_id: str,
    images: list = None,
    repo_folder: str = None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    model_description = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{args.validation_prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_description += wandb_info

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=args.pretrained_model_name_or_path,
        model_description=model_description,
        inference=True,
    )

    tags = ["stable-diffusion", "stable-diffusion-diffusers", "text-to-image", "diffusers", "diffusers-training"]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))

class PromptReplacer:
    def __init__(self, src_latents):
        self.src_latents = src_latents
    def replace_the_prompt_callback(self, idx, t, latents):
        latents_rearrange = rearrange(latents, '(b n) c h w -> b n c h w', n=2)
        _, tgt = latents_rearrange.chunk(2, dim=1)
        combined_latents = rearrange(torch.concat([self.src_latents, tgt], dim=1) , 'b n c h w -> (b n) c h w')
        latents[...] = combined_latents[...]

@torch.no_grad()
def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch,
                   feature_extractor, image_encoder, image_normalizer, image_noising_scheduler, scheduler,
                   train_dataloader, generator):
    logger.info("Running validation... ")

    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        feature_extractor=accelerator.unwrap_model(feature_extractor),
        image_encoder=accelerator.unwrap_model(image_encoder),
        image_normalizer=accelerator.unwrap_model(image_normalizer),
        image_noising_scheduler=accelerator.unwrap_model(image_noising_scheduler),
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        scheduler=accelerator.unwrap_model(scheduler),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    # random select a batch from trainingloder
    for batch in train_dataloader:
        # 0. get latents 
        imgs_prompt = batch["prompt_img"].to(device = accelerator.device, dtype=vae.dtype)[0].unsqueeze(0)
        imgs_render = batch["render"].to(device = accelerator.device, dtype=vae.dtype)[0].unsqueeze(0)
        concat_batch_img = torch.cat([imgs_prompt, imgs_render], dim=0)
        concat_latents = pipeline.vae.encode(concat_batch_img).latent_dist.mode() * vae.config.scaling_factor
        prompt_latents, render_latents = concat_latents.chunk(2)

        # 1. get image embedding
        imgs_prompt_pil = batch["prompt_pil"][0]
        cond_img = pipeline.feature_extractor(images=imgs_prompt_pil,
                                        return_tensors="pt").pixel_values.to(accelerator.device)
        image_embeds = pipeline.image_encoder(cond_img).image_embeds

        # 2. get text_embedding
        pos_prompt = ["best quality, high quality"] * image_embeds.shape[0]
        negative_prompt = ["low quality, bad quality"] * image_embeds.shape[0]

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = pipeline.encode_prompt(
                pos_prompt,
                num_images_per_prompt=1,
                device=image_embeds.device,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=True,
            )

        # 3. Sample noise that we'll add to the latents
        src_latents = prompt_latents.unsqueeze(1)
        noise_latents = randn_tensor(src_latents.shape,
                               generator=generator,
                               device=src_latents.device,
                               dtype=src_latents.dtype)
        tgt_latents = noise_latents

        combined_init_latents = rearrange(torch.cat([src_latents, tgt_latents], dim=1), 'b n c h w -> (b n) c h w')

        prompt_embeds_ = prompt_embeds_.unsqueeze(1)
        combined_prompt_embeds_ = rearrange(torch.cat([prompt_embeds_] * 2, dim=1), 'b n c d -> (b n) c d')

        negative_prompt_embeds_ = negative_prompt_embeds_.unsqueeze(1)
        combined_negative_prompt_embeds_ = rearrange(torch.cat([negative_prompt_embeds_] * 2, dim=1), 'b n c d -> (b n) c d')

        image_embeds = image_embeds.unsqueeze(1)
        combined_image_embeds = rearrange(torch.cat([image_embeds] * 2, dim=1), 'b n d -> (b n) d')

        render_latents = render_latents.unsqueeze(1)
        combined_render_latents = rearrange(torch.cat([render_latents] * 2, dim=1), 'b n c h w -> (b n) c h w')

        prompt_replacer = PromptReplacer(src_latents)

        # 4. cfg guidance
        combined_prompt_embeds_ = torch.cat([combined_negative_prompt_embeds_,
                                             combined_prompt_embeds_])
        noise_level = torch.tensor([0], device=accelerator.device)
        combined_image_embeds = pipeline._encode_image(
            image = None,
            device = accelerator.device,
            batch_size = combined_prompt_embeds_.shape[0],
            num_images_per_prompt = 1,
            do_classifier_free_guidance = True,
            noise_level = noise_level,
            generator = generator,
            image_embeds = combined_image_embeds,
        )
        pipeline.scheduler.set_timesteps(num_inference_steps=20, device=accelerator.device)
        timesteps = pipeline.scheduler.timesteps

        latents = combined_init_latents

        for i, t in enumerate(pipeline.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2)
            pos_tgt_t = t.unsqueeze(0)
            pos_src_t = torch.zeros_like(pos_tgt_t)
            pos_combined_t = torch.cat([pos_src_t, pos_tgt_t], dim=0)
            combined_t = torch.cat([pos_combined_t] * 2)

            noise_pred = pipeline.unet(
                latent_model_input,
                combined_t,
                encoder_hidden_states=combined_prompt_embeds_,
                class_labels=combined_image_embeds,
                cross_attention_kwargs={'concat_conds': combined_render_latents},
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 10.0 * (noise_pred_text - noise_pred_uncond)

            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            prompt_replacer.replace_the_prompt_callback(i, t, latents)

        image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        image = pipeline.image_processor.postprocess(image, output_type="pil")

        pipeline.maybe_free_model_hooks()
        result = image

        image_gt = batch['gt'][0].unsqueeze(0)
        images.append({'result': result, 'gt': image_gt})
        os.makedirs('image_log_together', exist_ok=True)
        os.makedirs('image_log_together/train_{}'.format(epoch), exist_ok=True)
        result[0].save('image_log_together/train_{}/output-prompt-{:06d}_e-{:06d}_b-{:06d}.png'.format(epoch, 0, epoch, 0))
        result[1].save('image_log_together/train_{}/output-{:06d}_e-{:06d}_b-{:06d}.png'.format(epoch, 0, epoch, 0))
        save_image(image_gt * 0.5 + 0.5, 'image_log_together/train_{}/gt-{:06d}_e-{:06d}_b-{:06d}.png'.format(epoch, 0, epoch, 0))
        save_image(imgs_render * 0.5 + 0.5, 'image_log_together/train_{}/input-{:06d}_e-{:06d}_b-{:06d}.png'.format(epoch, 0, epoch, 0))
        batch["prompt_pil"][0].save('image_log_together/train_{}/condition-{:06d}_e-{:06d}_b-{:06d}.png'.format(epoch, 0, epoch, 0))
        

        break

#     for i in range(len(args.validation_prompts)):
#         if torch.backends.mps.is_available():
#             autocast_ctx = nullcontext()
#         else:
#             autocast_ctx = torch.autocast(accelerator.device.type)

#         with autocast_ctx:
#             image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]

#         images.append(image)

#     for tracker in accelerator.trackers:
#         if tracker.name == "tensorboard":
#             np_images = np.stack([np.asarray(img) for img in images])
#             tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
#         elif tracker.name == "wandb":
#             tracker.log(
#                 {
#                     "validation": [
#                         wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
#                         for i, image in enumerate(images)
#                     ]
#                 }
#             )
#         else:
#             logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    # parser.add_argument(
    #     "--pretrained_unet_path",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="Path to pretrained UNet model.",
    # )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd21-unclip-model-finetuned-together",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=20,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--dream_training",
        action="store_true",
        help=(
            "Use the DREAM training method, which makes training more efficient and accurate at the ",
            "expense of doing an extra forward pass. See: https://arxiv.org/abs/2312.00210",
        ),
    )
    parser.add_argument(
        "--dream_detail_preservation",
        type=float,
        default=1.0,
        help="Dream detail preservation factor p (should be greater than 0; default=1.0, as suggested in the paper)",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--offload_ema", action="store_true", help="Offload EMA model to CPU during training step.")
    parser.add_argument("--foreach_ema", action="store_true", help="Use faster foreach implementation of EMAModel.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_classifier_free_guidance",
        type=bool,
        default=True,
        help="Whether to use classifier free guidance for the diffusion model.",
    )
    parser.add_argument(
        "--condition_dropout_prob",
        type=float,
        default=0.05,
        help="The dropout probability for the conditioning layers.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.05,
        help="The dropout probability for the conditioning layers.",
    )

    parser.add_argument(
        "--pretrained_ip_adpater_path",
        type=str,
        default=None,
        help="Path to the pretrained IP Adapter model.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

def get_null_context(tokenizer, text_encoder):
    prompt = ""
    with torch.no_grad():
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_inputs_ids
        )[0]
    return prompt_embeds




def main():
    args = parse_args()
   
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    else:
        generator = None

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder")
    image_normalizer = StableUnCLIPImageNormalizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_normalizer")
    image_noising_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_noising_scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    noise_scheduler = PNDMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")


    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in
    
    unet_original_forward = unet.forward

    def hooked_original_forward(sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs['cross_attention_kwargs'] = {}
        return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

    unet.forward = hooked_original_forward

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()
    image_encoder.requires_grad_(False)

    # ip-adapter
    # image_proj_model = ImageProjModel(
    #     cross_attention_dim=unet.config.cross_attention_dim,
    #     clip_embeddings_dim=image_encoder.config.projection_dim,
    #     clip_extra_context_tokens=4,
    # )
    # init adapter modules
    attn_procs = {}
    for name in unet.attn_processors.keys():
        if name.endswith("attn1.processor"):
            # self-attention -> tgt_src_cross-attention + src_self-attention
            attn_procs[name] = SelfMutualAttnProcessor()
        elif name.endswith("attn2.processor"):
            # cross-attention
            attn_procs[name] = AttnProcessor()
    unet.set_attn_processor(attn_procs)

    nulltxt_embeds = get_null_context(tokenizer, text_encoder)

    # if args.enable_xformers_memory_efficient_attention:
    #     if is_xformers_available():
    #         import xformers

    #         xformers_version = version.parse(xformers.__version__)
    #         if xformers_version == version.parse("0.0.16"):
    #             logger.warning(
    #                 "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
    #             )
    #         unet.enable_xformers_memory_efficient_attention()
    #     else:
    #         raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    if isinstance(model, UNet2DConditionModel):
                        model.save_pretrained(os.path.join(output_dir, "unet"), safe_serialization=False)
                    else:
                        raise ValueError(f"Model {model} not supported for saving")
                    # elif isinstance(model, IPAdapter):
                    #     state_dict = {"image_proj":model.image_proj_model.state_dict(), "adapter_modules":model.adapter_modules.state_dict()}
                    #     os.makedirs(os.path.join(output_dir, "ip_adapter"), exist_ok=True)
                    #     torch.save(state_dict, os.path.join(output_dir, "ip_adapter", "diffusion_pytorch_model.bin"))
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop(0)

        def load_model_hook(models, input_dir):

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                if isinstance(model, UNet2DConditionModel):
                    # load diffusers style into model
                    model_path = os.path.join(input_dir, "unet", "diffusion_pytorch_model.bin")
                    loaded_state_dict = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(loaded_state_dict, strict=True)

                    del loaded_state_dict
                else:
                    raise ValueError(f"Model {model} not supported for loading")
                # elif isinstance(model, IPAdapter):
                #     model_path = os.path.join(input_dir, "ip_adapter", "diffusion_pytorch_model.bin")
                #     model.load_from_checkpoint(model_path)


        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # Combine the parameters
    params_to_opt = itertools.chain(unet.parameters())

    optimizer = optimizer_cls(
        params_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    def collate_fn(examples):
        render = torch.stack([example["render"] for example in examples])
        render = render.to(memory_format=torch.contiguous_format).float()
        prompt_img = torch.stack([example["prompt_img"] for example in examples])
        prompt_img = prompt_img.to(memory_format=torch.contiguous_format).float()
        gt = torch.stack([example["gt"] for example in examples])
        gt = gt.to(memory_format=torch.contiguous_format).float()
        prompt_pil = [example["prompt_pil"] for example in examples]
        return {"render": render, "prompt_img": prompt_img, 'gt': gt, 'prompt_pil': prompt_pil}
    

    train_dataset = MyDataset('train', generator=generator)
    # val_dataset = MyDataset('val', generator=generator)


    # DataLoaders creation:
    # batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(train_dataset),
    #                                         dataset=train_dataset,
    #                                         aspect_ratios=train_dataset.aspect_ratios,
    #                                          batch_size=args.train_batch_size, 
    #                                          drop_last=True,
    #                                          ratio_nums=train_dataset.ratio_nums,
    #                                          )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    val_dataloader = train_dataloader

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    if accelerator.is_main_process:
        log_validation(
            vae,
            text_encoder,
            tokenizer,
            unet,
            args,
            accelerator,
            weight_dtype,
            global_step,
            feature_extractor,
            image_encoder,
            image_normalizer,
            image_noising_scheduler,
            noise_scheduler,
            val_dataloader,
            generator,
        )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 0. Get the concat latent embeddings
                imgs_prompt = batch["prompt_img"].to(device = accelerator.device, dtype=vae.dtype)
                imgs_render = batch["render"].to(device = accelerator.device, dtype=vae.dtype)
                imgs_gt = batch["gt"].to(device = accelerator.device, dtype=vae.dtype)
                concat_batch_img = torch.cat([imgs_prompt, imgs_render, imgs_gt], dim=0)
                concat_latents = vae.encode(concat_batch_img).latent_dist.mode() * vae.config.scaling_factor
                prompt_latents, render_latents, gt_latents = concat_latents.chunk(3)

                # 1. Get the image class embeddings
                imgs_prompt_pil = batch["prompt_pil"]
                cond_img = feature_extractor(images=imgs_prompt_pil,
                                             return_tensors="pt").pixel_values.to(accelerator.device)
                image_embeds_ori = image_encoder(cond_img).image_embeds
                noise_level = torch.tensor([0], device=accelerator.device)
                img_embed_noise = randn_tensor(
                    image_embeds_ori.shape, device=image_embeds_ori.device,dtype=image_embeds_ori.dtype
                )
                noise_level = torch.tensor([noise_level] * image_embeds_ori.shape[0], device=image_embeds_ori.device)
                image_normalizer.to(image_embeds_ori.device)
                image_embeds = image_normalizer.scale(image_embeds_ori)
                image_embeds = image_noising_scheduler.add_noise(image_embeds, timesteps=noise_level, noise=img_embed_noise)
                image_embeds = image_normalizer.unscale(image_embeds)
                noise_level = get_timestep_embedding(
                    timesteps=noise_level,
                    embedding_dim=image_embeds.shape[-1],
                    flip_sin_to_cos=True,
                    downscale_freq_shift=0
                )
                noise_level = noise_level.to(image_embeds.dtype)
                image_embeds = torch.cat([image_embeds, noise_level], dim=1)

                # 2. Get the text embeddings
                encoder_hidden_states = torch.concat([nulltxt_embeds] * len(imgs_prompt_pil), dim=0).to(accelerator.device)

                # 3. Cfg for the noise scheduler
                if args.use_classifier_free_guidance and args.condition_dropout_prob > 0:
                    random_p = torch.rand(gt_latents.shape[0], device=gt_latents.device, generator=generator)
                    
                    # 3.1. Mask for render_latents
                    image_mask_dtype = render_latents.dtype
                    image_mask = 1 - (
                        torch.logical_or(
                            torch.logical_or(
                                (random_p < args.condition_dropout_prob),
                                torch.logical_and(random_p >= 3 * args.condition_dropout_prob, random_p < 5 * args.condition_dropout_prob)
                            ),
                            torch.logical_and(random_p >= 6 * args.condition_dropout_prob, random_p < 7 * args.condition_dropout_prob)
                        )
                    ).to(image_mask_dtype)
                    image_mask = image_mask.reshape(gt_latents.shape[0], 1, 1, 1)
                    render_latents = render_latents * image_mask
                    
                    # 3.2. Mask for image_embeds
                    clip_mask_dtype = image_embeds.dtype
                    clip_mask = 1 - (
                        torch.logical_or(
                            torch.logical_or(
                                torch.logical_and(random_p >= args.condition_dropout_prob, random_p < 2 * args.condition_dropout_prob),
                                torch.logical_and(random_p >= 3 * args.condition_dropout_prob, random_p < 4 * args.condition_dropout_prob)
                            ),
                            torch.logical_and(random_p >= 5 * args.condition_dropout_prob, random_p < 7 * args.condition_dropout_prob)
                        )
                    ).to(clip_mask_dtype)
                    clip_mask = clip_mask.reshape(gt_latents.shape[0], 1)
                    image_embeds = image_embeds * clip_mask
                    
                    # 3.3. Mask for image_embeds_ori
                    prompt_mask_dtype = prompt_latents.dtype
                    prompt_mask = 1 - (
                        torch.logical_or(
                            torch.logical_and(random_p >= 2 * args.condition_dropout_prob, random_p < 3 * args.condition_dropout_prob),
                            torch.logical_and(random_p >= 4 * args.condition_dropout_prob, random_p < 7 * args.condition_dropout_prob)
                        )
                    ).to(prompt_mask_dtype)
                    prompt_mask = prompt_mask.reshape(gt_latents.shape[0], 1, 1, 1)
                    prompt_latents = prompt_latents * prompt_mask

                # 4. Sample noise that we'll add to the latents
                src_latents = prompt_latents.unsqueeze(1)
                tgt_latents = gt_latents.unsqueeze(1)
                combined_clean_latents = rearrange(torch.cat([src_latents, tgt_latents], dim=1), 'b n c h w -> (b n) c h w')
                bsz, cdim, _, _ = combined_clean_latents.shape

                noise = torch.randn_like(combined_clean_latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (bsz, cdim, 1, 1), device=combined_clean_latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                
                # 5. Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (gt_latents.shape[0],), device=gt_latents.device)
                timesteps = timesteps.long()
                timesteps = timesteps.unsqueeze(1)
                combined_timesteps = rearrange(torch.cat([torch.zeros_like(timesteps), timesteps], dim=1), 'b n -> (b n)')

                # 6. Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(combined_clean_latents, new_noise, combined_timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(combined_clean_latents, noise, combined_timesteps)

                # 7. Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=combined_clean_latents.device, dtype=combined_clean_latents.dtype)
                    combined_timesteps = combined_timesteps.to(device=combined_clean_latents.device)
                    sqrt_alpha_prod = alphas_cumprod[combined_timesteps] ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                    while len(sqrt_alpha_prod.shape) < len(combined_clean_latents.shape):
                        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                    
                    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[combined_timesteps]) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                    while len(sqrt_one_minus_alpha_prod.shape) < len(combined_clean_latents.shape):
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                    
                    velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * combined_clean_latents
                    target = velocity
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # 8. rearrange the latents, timesteps and encoder_hidden_states for the model    
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
                combined_encoder_hidden_states = rearrange(torch.cat([encoder_hidden_states] * 2, dim=1), 'b n c d -> (b n) c d')

                image_embeds = image_embeds.unsqueeze(1)
                combined_image_embeds = rearrange(torch.cat([image_embeds] * 2, dim=1), 'b n c -> (b n) c')

                render_latents = render_latents.unsqueeze(1)
                combined_render_latents = rearrange(torch.cat([render_latents] * 2, dim=1), 'b n c h w -> (b n) c h w')

                # 9. Predict the noise residual and compute loss
                model_pred = unet(noisy_latents,
                                  combined_timesteps, 
                                  encoder_hidden_states=combined_encoder_hidden_states,
                                  class_labels=combined_image_embeds,
                                  cross_attention_kwargs={'concat_conds': combined_render_latents},
                                  return_dict=False)[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if epoch % args.validation_epochs == 0:
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    unet,
                    args,
                    accelerator,
                    weight_dtype,
                    global_step,
                    feature_extractor,
                    image_encoder,
                    image_normalizer,
                    image_noising_scheduler,
                    noise_scheduler,
                    val_dataloader,
                    generator,
                )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
        if args.checkpoints_total_limit is not None:
            checkpoints = os.listdir(args.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
            if len(checkpoints) >= args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
