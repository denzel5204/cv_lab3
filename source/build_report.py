import csv
import json
import math
from pathlib import Path

from source.data import load_rgb
from source.degradations import make_degradation
from source.masks import create_mask
from source.plots import save_comparison, save_metric_plot


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    (ROOT / "images").mkdir(parents=True, exist_ok=True)
    with (ROOT / "results" / "metrics.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    run_info = json.loads((ROOT / "results" / "run.json").read_text(encoding="utf-8"))
    seed = int(run_info["seed"])
    input_dir = ROOT / "data" / "examples"

    comparison_keys = sorted(
        {
            (row["image"], row["degradation"], row["mask"])
            for row in rows
            if row["mask"] in {"center", "soft_center"}
        }
    )
    for image_name, degradation, mask_name in comparison_keys:
        reference = load_rgb(input_dir / f"{image_name}.png")
        degradation_index = 0 if degradation == "bicubic" else 1
        low_res = make_degradation(reference, degradation, seed + degradation_index)
        mask = create_mask(mask_name, reference.size)
        prefix = f"{image_name}_{degradation}_{mask_name}"
        save_comparison(
            reference,
            low_res,
            mask,
            {
                "Склейка": load_rgb(ROOT / "results" / f"{prefix}_pixel.png"),
                "Маска в сэмплинге": load_rgb(
                    ROOT / "results" / f"{prefix}_latent.png"
                ),
            },
            ROOT / "images" / f"{prefix}.png",
        )

    numeric_rows = []
    for row in rows:
        numeric_rows.append(
            {
                **row,
                "psnr_mask": float(row["psnr_mask"]),
                "outside_mae": float(row["outside_mae"]),
                "lpips": float(row["lpips"]),
                "clip_iqa": float(row["clip_iqa"]),
            }
        )

    if any(not math.isnan(row["lpips"]) for row in numeric_rows):
        save_metric_plot(numeric_rows, ROOT / "images" / "metrics.png")


if __name__ == "__main__":
    main()
