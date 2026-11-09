from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
import torch
from diffusers import LDMSuperResolutionPipeline
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion_superresolution import (
    preprocess,
)
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from source.masks import mask_to_tensor


MODEL_ID = "CompVis/ldm-super-resolution-4x-openimages"


@dataclass
class RestorationResult:
    image: Image.Image
    seconds: float


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(device: str | None = None) -> LDMSuperResolutionPipeline:
    device = device or select_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = LDMSuperResolutionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(leave=False)
    return pipeline


def pil_to_model_tensor(image: Image.Image, device, dtype) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)[None]
    return tensor.to(device=device, dtype=dtype)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu().clamp(-1, 1) / 2.0 + 0.5
    array = (image[0].permute(1, 2, 0).numpy() * 255.0).round()
    return Image.fromarray(array.astype(np.uint8))


def bicubic_baseline(low_res: Image.Image, output_size: tuple[int, int]) -> Image.Image:
    return low_res.resize(output_size, Image.Resampling.BICUBIC)


def blend_images(restored: Image.Image, baseline: Image.Image, mask: Image.Image) -> Image.Image:
    return Image.composite(restored, baseline, mask.convert("L"))


@torch.inference_mode()
def restore_full(
    pipeline: LDMSuperResolutionPipeline,
    low_res: Image.Image,
    steps: int,
    seed: int,
) -> Image.Image:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return pipeline(
        low_res,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]


@torch.inference_mode()
def restore_masked(
    pipeline: LDMSuperResolutionPipeline,
    low_res: Image.Image,
    mask: Image.Image,
    steps: int,
    seed: int,
) -> Image.Image:
    device = pipeline.device
    dtype = pipeline.unet.dtype
    image = preprocess(low_res).to(device=device, dtype=dtype)

    batch_size, _, height, width = image.shape
    latent_channels = pipeline.unet.config.in_channels // 2
    shape = (batch_size, latent_channels, height, width)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    source_noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    pipeline.scheduler.set_timesteps(steps, device=device)
    latents = latents * pipeline.scheduler.init_noise_sigma

    output_size = (width * 4, height * 4)
    baseline = bicubic_baseline(low_res, output_size)
    baseline_tensor = pil_to_model_tensor(baseline, device, dtype)

    # Кодируем bicubic-версию в то же пространство, где идёт диффузия
    source_latents = pipeline.vqvae.encode(baseline_tensor).latents
    latent_mask = mask_to_tensor(mask, (width, height), device, dtype)

    extra_kwargs = {}
    if "eta" in inspect.signature(pipeline.scheduler.step).parameters:
        extra_kwargs["eta"] = 0.0

    for timestep in pipeline.progress_bar(pipeline.scheduler.timesteps):
        timestep_batch = timestep.expand(batch_size)
        source_at_t = pipeline.scheduler.add_noise(
            source_latents,
            source_noise,
            timestep_batch,
        )

        # Вне маски возвращаем зашумлённый исходник на каждом шаге
        latents = latent_mask * latents + (1.0 - latent_mask) * source_at_t
        model_input = torch.cat([latents, image], dim=1)
        model_input = pipeline.scheduler.scale_model_input(model_input, timestep)
        noise_prediction = pipeline.unet(model_input, timestep).sample
        latents = pipeline.scheduler.step(
            noise_prediction,
            timestep,
            latents,
            **extra_kwargs,
        ).prev_sample

    latents = latent_mask * latents + (1.0 - latent_mask) * source_latents
    decoded = pipeline.vqvae.decode(latents, return_dict=False)[0]
    restored = tensor_to_pil(decoded)

    # Финальное смешивание точно сохраняет пиксели вне маски
    return blend_images(restored, baseline, mask.resize(output_size))

