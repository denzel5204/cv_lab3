from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter


LOW_RES_SIZE = 128


def bicubic_downscale(image: Image.Image) -> Image.Image:
    return image.resize(
        (LOW_RES_SIZE, LOW_RES_SIZE),
        Image.Resampling.BICUBIC,
    )


def unknown_degradation(image: Image.Image, seed: int = 42) -> Image.Image:
    rng = np.random.default_rng(seed)
    low_res = bicubic_downscale(image).filter(ImageFilter.GaussianBlur(radius=1.2))

    pixels = np.asarray(low_res, dtype=np.float32)
    noise = rng.normal(0.0, 6.0, pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    noisy = Image.fromarray(pixels)

    # Повторное JPEG-сжатие создаёт неизвестную комбинацию искажений
    buffer = BytesIO()
    noisy.save(buffer, format="JPEG", quality=45)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def make_degradation(image: Image.Image, name: str, seed: int = 42) -> Image.Image:
    if name == "bicubic":
        return bicubic_downscale(image)
    if name == "unknown":
        return unknown_degradation(image, seed)
    raise ValueError(f"Неизвестный вид искажения: {name}")

