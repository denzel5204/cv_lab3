import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_mask(name: str, size: tuple[int, int]) -> Image.Image:
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    if name == "zero":
        return mask
    if name == "full":
        return Image.new("L", size, 255)
    if name == "center":
        draw.rectangle(
            (width // 4, height // 4, 3 * width // 4, 3 * height // 4),
            fill=255,
        )
        return mask
    if name == "two_regions":
        draw.ellipse((40, 55, 230, 245), fill=255)
        draw.rectangle((300, 280, 480, 450), fill=255)
        return mask
    if name == "soft_center":
        draw.rectangle(
            (width // 4, height // 4, 3 * width // 4, 3 * height // 4),
            fill=255,
        )
        return mask.filter(ImageFilter.GaussianBlur(radius=12))

    raise ValueError(f"Неизвестная маска: {name}")


def mask_to_array(mask: Image.Image) -> np.ndarray:
    return np.asarray(mask, dtype=np.float32) / 255.0


def mask_to_tensor(
    mask: Image.Image,
    size: tuple[int, int],
    device,
    dtype,
):
    import torch

    resized = mask.resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None, None].to(device=device, dtype=dtype)

