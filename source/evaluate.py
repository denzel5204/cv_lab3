import argparse
import csv
import gc
from pathlib import Path

from source.data import load_rgb
from source.masked_sampler import select_device
from source.metrics import (
    calculate_clip_iqa,
    calculate_lpips,
    load_clip_iqa,
    load_lpips,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--skip-clip", action="store_true")
    return parser.parse_args()


def result_path(row: dict[str, str]) -> Path:
    name = (
        f"{row['image']}_{row['degradation']}_"
        f"{row['mask']}_{row['method']}.png"
    )
    return ROOT / "results" / name


def main() -> None:
    args = parse_args()
    metrics_path = ROOT / "results" / "metrics.csv"
    with metrics_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    device = select_device()
    input_dir = ROOT / "data" / "examples"

    if not args.skip_lpips:
        lpips = load_lpips(device)
        for row in rows:
            reference = load_rgb(input_dir / f"{row['image']}.png")
            result = load_rgb(result_path(row))
            row["lpips"] = calculate_lpips(lpips, reference, result, device)
        del lpips
        gc.collect()

    if not args.skip_clip:
        clip_iqa = load_clip_iqa(device)
        for row in rows:
            result = load_rgb(result_path(row))
            row["clip_iqa"] = calculate_clip_iqa(clip_iqa, result, device)
        del clip_iqa
        gc.collect()

    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
