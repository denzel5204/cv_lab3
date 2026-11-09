from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image


def save_comparison(
    reference: Image.Image,
    low_res: Image.Image,
    mask: Image.Image,
    outputs: dict[str, Image.Image],
    path: Path,
) -> None:
    columns = 3 + len(outputs)
    figure, axes = plt.subplots(1, columns, figsize=(4 * columns, 4))
    items = [
        ("Эталон", reference),
        ("Ухудшенное", low_res.resize(reference.size)),
        ("Маска", mask),
        *outputs.items(),
    ]

    for axis, (title, image) in zip(axes, items):
        axis.imshow(image, cmap="gray" if image.mode == "L" else None)
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_metric_plot(results: list[dict], path: Path) -> None:
    combined = [row for row in results if row["mask"] == "center"]
    images = sorted({row["image"] for row in combined})
    methods = sorted({row["method"] for row in combined})
    degradations = ["bicubic", "unknown"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    metrics = [
        ("psnr_mask", "PSNR внутри маски, дБ\nбольше лучше", True),
        ("lpips", "LPIPS\nменьше лучше", False),
        ("clip_iqa", "CLIP-IQA\nбольше лучше", True),
    ]
    width = 0.35
    positions = list(range(len(images)))
    method_labels = {
        "latent": "маска в сэмплинге",
        "pixel": "склейка",
    }
    degradation_labels = {
        "bicubic": "Bicubic x4",
        "unknown": "Размытие, шум и JPEG",
    }
    for row_index, degradation in enumerate(degradations):
        current = [row for row in combined if row["degradation"] == degradation]
        for axis, (column, title, _) in zip(axes[row_index], metrics):
            for method_index, method in enumerate(methods):
                values = []
                for image in images:
                    selected = [
                        row[column]
                        for row in current
                        if row["image"] == image and row["method"] == method
                    ]
                    values.append(sum(selected) / len(selected))
                offsets = [
                    position + (method_index - (len(methods) - 1) / 2) * width
                    for position in positions
                ]
                axis.bar(offsets, values, width=width, label=method_labels[method])
            axis.set_title(f"{title}\n{degradation_labels[degradation]}")
            axis.set_xticks(positions, images, rotation=15)
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            axis.legend(title="")

    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)

