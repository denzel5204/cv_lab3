import argparse
import csv
import gc
import json
import time
from pathlib import Path

from source.data import load_rgb, prepare_example_images
from source.degradations import make_degradation
from source.masked_sampler import (
    bicubic_baseline,
    blend_images,
    load_pipeline,
    restore_full,
    restore_masked,
    select_device,
)
from source.masks import create_mask
from source.metrics import masked_psnr, outside_mae


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--images", default="astronaut,coffee,chelsea")
    parser.add_argument("--degradations", default="bicubic,unknown")
    parser.add_argument(
        "--masks",
        default="zero,full,center,two_regions,soft_center",
    )
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def evaluate(
    reference,
    baseline,
    result,
    mask,
) -> dict[str, float]:
    return {
        "psnr_mask": masked_psnr(reference, result, mask),
        "outside_mae": outside_mae(baseline, result, mask),
        "lpips": float("nan"),
        "clip_iqa": float("nan"),
    }


def main() -> None:
    args = parse_args()
    input_dir = ROOT / "data" / "examples"
    output_dir = ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = prepare_example_images(input_dir)
    if args.smoke:
        image_paths = image_paths[:1]
        degradations = ["bicubic"]
        mask_names = ["zero", "full", "center"]
    else:
        selected_images = {name.strip() for name in args.images.split(",")}
        image_paths = [path for path in image_paths if path.stem in selected_images]
        degradations = [name.strip() for name in args.degradations.split(",")]
        mask_names = [name.strip() for name in args.masks.split(",")]

    if not image_paths:
        raise ValueError("Не выбрано ни одного изображения")

    device = select_device()
    print(f"Устройство: {device}")
    pipeline = load_pipeline(device)
    records = []

    for image_path in image_paths:
        reference = load_rgb(image_path)
        for degradation_index, degradation in enumerate(degradations):
            low_res = make_degradation(reference, degradation, args.seed + degradation_index)
            baseline = bicubic_baseline(low_res, reference.size)

            start = time.perf_counter()
            full_result = restore_full(pipeline, low_res, args.steps, args.seed)
            full_seconds = time.perf_counter() - start

            for mask_name in mask_names:
                mask = create_mask(mask_name, reference.size)
                pixel_result = blend_images(full_result, baseline, mask)

                if mask_name == "zero":
                    latent_result = baseline.copy()
                    latent_seconds = 0.0
                elif mask_name == "full":
                    latent_result = full_result.copy()
                    latent_seconds = full_seconds
                else:
                    start = time.perf_counter()
                    latent_result = restore_masked(
                        pipeline,
                        low_res,
                        mask,
                        args.steps,
                        args.seed,
                    )
                    latent_seconds = time.perf_counter() - start

                methods = {
                    "pixel": (pixel_result, full_seconds),
                    "latent": (latent_result, latent_seconds),
                }
                for method, (result, seconds) in methods.items():
                    file_stem = f"{image_path.stem}_{degradation}_{mask_name}_{method}"
                    result.save(output_dir / f"{file_stem}.png")
                    metrics = evaluate(
                        reference,
                        baseline,
                        result,
                        mask,
                    )
                    records.append(
                        {
                            "image": image_path.stem,
                            "degradation": degradation,
                            "mask": mask_name,
                            "method": method,
                            "steps": args.steps,
                            "seconds": seconds,
                            **metrics,
                        }
                    )

    # Освобождаем диффузионную модель перед отдельным расчётом метрик
    del pipeline
    gc.collect()

    metrics_path = output_dir / "metrics.csv"
    if args.append and metrics_path.exists():
        with metrics_path.open(encoding="utf-8", newline="") as file:
            previous = list(csv.DictReader(file))
        new_keys = {
            (row["image"], row["degradation"], row["mask"], row["method"])
            for row in records
        }
        previous = [
            row
            for row in previous
            if (row["image"], row["degradation"], row["mask"], row["method"])
            not in new_keys
        ]
        records = previous + records

    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "run.json").write_text(
        json.dumps(
            {
                "device": device,
                "steps": args.steps,
                "seed": args.seed,
                "smoke": args.smoke,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for record in records:
        print(record)


if __name__ == "__main__":
    main()

