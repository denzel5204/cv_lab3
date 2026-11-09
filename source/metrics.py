import numpy as np
import torch
from PIL import Image

from source.masks import mask_to_array


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def masked_psnr(reference: Image.Image, result: Image.Image, mask: Image.Image) -> float:
    reference_array = image_array(reference)
    result_array = image_array(result)
    weights = mask_to_array(mask.resize(reference.size))[:, :, None]
    denominator = max(float(weights.sum()) * 3.0, 1.0)
    mse = float((((reference_array - result_array) ** 2) * weights).sum() / denominator)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def outside_mae(baseline: Image.Image, result: Image.Image, mask: Image.Image) -> float:
    baseline_array = image_array(baseline)
    result_array = image_array(result)
    weights = 1.0 - mask_to_array(mask.resize(baseline.size))[:, :, None]
    denominator = max(float(weights.sum()) * 3.0, 1.0)
    return float((np.abs(baseline_array - result_array) * weights).sum() / denominator)


def to_tensor(image: Image.Image, device: str) -> torch.Tensor:
    array = image_array(image)
    return torch.from_numpy(array).permute(2, 0, 1)[None].to(device)


def load_lpips(device: str):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    return LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(device)


def load_clip_iqa(device: str):
    from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment

    metric = CLIPImageQualityAssessment(
        model_name_or_path="openai/clip-vit-base-patch32"
    )
    # Тензоры уже находятся в диапазоне от 0 до 1
    metric.processor.image_processor.do_rescale = False
    return metric.to(device)


@torch.inference_mode()
def calculate_lpips(
    metric,
    reference: Image.Image,
    result: Image.Image,
    device: str,
) -> float:
    reference_tensor = to_tensor(reference, device)
    result_tensor = to_tensor(result, device)
    value = metric(result_tensor * 2.0 - 1.0, reference_tensor * 2.0 - 1.0)
    metric.reset()
    return float(value.item())


@torch.inference_mode()
def calculate_clip_iqa(metric, result: Image.Image, device: str) -> float:
    value = metric(to_tensor(result, device))
    metric.reset()
    return float(value.item())

