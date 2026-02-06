# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
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

"""SD3 Pipeline with dynamic resolution support for DeepGen model."""

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3IPAdapterMixin, SD3LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """Calculate timestep shift based on image sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """Retrieve timesteps from scheduler."""
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusion3Pipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin, SD3IPAdapterMixin):
    """SD3 Pipeline with dynamic resolution support for DeepGen.

    This pipeline supports:
    - Dynamic input/output resolutions
    - Conditional image inputs for editing
    - Pre-computed prompt embeddings from the DeepGen connector

    Args:
        transformer: SD3 Transformer model
        scheduler: Flow matching scheduler
        vae: VAE for encoding/decoding
        text_encoder: CLIP text encoder (optional, not used with DeepGen)
        tokenizer: CLIP tokenizer (optional)
        text_encoder_2: Second CLIP encoder (optional)
        tokenizer_2: Second tokenizer (optional)
        text_encoder_3: T5 encoder (optional)
        tokenizer_3: T5 tokenizer (optional)
        image_encoder: Image encoder for IP-Adapter (optional)
        feature_extractor: Feature extractor for IP-Adapter (optional)
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->image_encoder->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection = None,
        tokenizer: CLIPTokenizer = None,
        text_encoder_2: CLIPTextModelWithProjection = None,
        tokenizer_2: CLIPTokenizer = None,
        text_encoder_3: T5EncoderModel = None,
        tokenizer_3: T5TokenizerFast = None,
        image_encoder: SiglipVisionModel = None,
        feature_extractor: SiglipImageProcessor = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        self.patch_size = (
            self.transformer.config.patch_size if hasattr(self, "transformer") and self.transformer is not None else 2
        )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """Prepare initial noise latents."""
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        cond_latents: Optional[List[torch.FloatTensor]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        mu: Optional[float] = None,
    ):
        """Generate images using the SD3 pipeline.

        This method is designed to work with pre-computed embeddings from DeepGen.

        Args:
            prompt: Text prompt (not used when prompt_embeds is provided)
            height: Output image height
            width: Output image width
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            generator: Random generator for reproducibility
            latents: Pre-generated noise latents
            cond_latents: Conditional image latents for editing
            prompt_embeds: Pre-computed prompt embeddings from DeepGen connector
            negative_prompt_embeds: Pre-computed negative embeddings
            pooled_prompt_embeds: Pre-computed pooled embeddings
            negative_pooled_prompt_embeds: Pre-computed negative pooled embeddings
            output_type: Output format ("pil", "latent", etc.)
            return_dict: Whether to return a dataclass

        Returns:
            Generated images or latents
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale

        # Determine batch size
        if prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        elif prompt is not None:
            batch_size = len(prompt) if isinstance(prompt, list) else 1
        else:
            batch_size = 1

        device = self._execution_device

        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        self._num_timesteps = len(timesteps)

        # Prepare latents
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype if prompt_embeds is not None else torch.float32,
            device,
            generator,
            latents,
        )

        # Convert latents to list format for dynamic resolution support
        latents_list = [latents[i] for i in range(latents.shape[0])]

        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand timestep
            timestep = t.expand(batch_size)

            # Prepare model inputs
            if self.do_classifier_free_guidance:
                # Concatenate conditional and unconditional
                latent_model_input = latents_list + latents_list
                cond_input = cond_latents + cond_latents if cond_latents is not None else None
                encoder_hidden_states = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
                pooled_projections = torch.cat([pooled_prompt_embeds, negative_pooled_prompt_embeds], dim=0)
                timestep = torch.cat([timestep, timestep], dim=0)
            else:
                latent_model_input = latents_list
                cond_input = cond_latents
                encoder_hidden_states = prompt_embeds
                pooled_projections = pooled_prompt_embeds

            # Predict noise
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                cond_hidden_states=cond_input,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )[0]

            # CFG
            if self.do_classifier_free_guidance:
                noise_pred_cond = noise_pred[:batch_size]
                noise_pred_uncond = noise_pred[batch_size:]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Scheduler step
            if isinstance(noise_pred, list):
                latents_list = [
                    self.scheduler.step(np, t, lat, return_dict=False)[0]
                    for np, lat in zip(noise_pred, latents_list)
                ]
            else:
                latents = self.scheduler.step(noise_pred, t, torch.stack(latents_list), return_dict=False)[0]
                latents_list = [latents[i] for i in range(latents.shape[0])]

            # Callback
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                if callback_outputs is not None:
                    latents_list = callback_outputs.pop("latents", latents_list)

            if XLA_AVAILABLE:
                xm.mark_step()

        # Stack latents
        latents = torch.stack(latents_list)

        if output_type == "latent":
            return StableDiffusion3PipelineOutput(images=latents)

        # Decode latents
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        images = self.vae.decode(latents, return_dict=False)[0]
        images = self.image_processor.postprocess(images, output_type=output_type)

        if not return_dict:
            return (images,)

        return StableDiffusion3PipelineOutput(images=images)
