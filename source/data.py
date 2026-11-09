from pathlib import Path
from urllib.request import urlopen

from PIL import Image


IMAGE_SIZE = 512
EXAMPLE_URLS = {
    "astronaut": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/astronaut.png",
    "coffee": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/coffee.png",
    "chelsea": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/chelsea.png",
}


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def prepare_example_images(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for name, url in EXAMPLE_URLS.items():
        path = output_dir / f"{name}.png"
        if path.exists():
            paths.append(path)
            continue

        with urlopen(url) as response:
            image = Image.open(response).convert("RGB")
        image = center_crop_square(image).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
        )
        image.save(path)
        paths.append(path)

    return paths


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")

