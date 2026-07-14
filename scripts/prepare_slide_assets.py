#!/usr/bin/env python3
"""Colab実行結果から発表用の軽量図版を生成する。図中の文字はASCIIのみ。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCALE = 2
BG = (248, 250, 252)
INK = (24, 35, 48)
MUTED = (93, 105, 119)
GRID = (218, 224, 231)
GREEN = (118, 185, 0)
ORANGE = (237, 129, 43)
BLUE = (57, 106, 177)
TEAL = (40, 150, 150)
RED = (207, 65, 65)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ["C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold
        else ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size * SCALE)
    return ImageFont.load_default()


def canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width * SCALE, height * SCALE), BG)
    return image, ImageDraw.Draw(image)


def finish(image: Image.Image, path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((width, height), Image.Resampling.LANCZOS).save(path, optimize=True)


def project(points: np.ndarray) -> np.ndarray:
    """車体点群を読みやすい擬似等角投影へ変換する。"""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.column_stack((x - 0.42 * y, z + 0.18 * y))


def map_to_box(values: np.ndarray, box: tuple[int, int, int, int], pad: float = 0.05) -> np.ndarray:
    x0, y0, x1, y1 = (v * SCALE for v in box)
    lo = values.min(axis=0)
    hi = values.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    lo -= pad * span
    hi += pad * span
    span = hi - lo
    sx = (x1 - x0) / span[0]
    sy = (y1 - y0) / span[1]
    scale = min(sx, sy)
    used = span * scale
    origin = np.array([x0 + (x1 - x0 - used[0]) / 2, y0 + (y1 - y0 - used[1]) / 2])
    mapped = origin + (values - lo) * scale
    mapped[:, 1] = y1 - (mapped[:, 1] - y0)
    return mapped


def circle(draw: ImageDraw.ImageDraw, xy: np.ndarray, radius: int, fill: tuple[int, int, int], outline=None) -> None:
    x, y = float(xy[0]), float(xy[1])
    r = radius * SCALE
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=2 * SCALE)


def make_stencil_zoom(npz_path: Path, output_path: Path) -> None:
    data = np.load(npz_path, allow_pickle=False)
    points = np.asarray(data["case_43_points"], dtype=np.float64)
    reference = np.asarray(data["case_43_reference"], dtype=np.float64).reshape(-1)
    prediction = np.asarray(data["case_43_prediction"], dtype=np.float64).reshape(-1)
    errors = np.abs(prediction - reference)
    query_idx = int(np.argmax(errors))
    distances = np.linalg.norm(points - points[query_idx], axis=1)
    neighbor_idx = np.argsort(distances)[1:4]
    local_idx = np.argsort(distances)[:280]

    image, draw = canvas(1600, 900)
    draw.text((70 * SCALE, 42 * SCALE), "Surface stencil: global position and local neighborhood", font=font(30, True), fill=INK)
    draw.text((70 * SCALE, 86 * SCALE), "Case 43 | query point selected at maximum absolute pressure error", font=font(17), fill=MUTED)

    left_box = (70, 155, 790, 745)
    right_box = (875, 155, 1530, 745)
    for box in (left_box, right_box):
        draw.rounded_rectangle(tuple(v * SCALE for v in box), radius=18 * SCALE, fill=(255, 255, 255), outline=GRID, width=2 * SCALE)

    overview = map_to_box(project(points), (100, 205, 760, 690))
    clipped = np.clip(errors / max(np.percentile(errors, 98), 1e-9), 0, 1)
    order = np.argsort(clipped)
    for idx in order:
        intensity = float(clipped[idx])
        color = (
            int(185 + 60 * intensity),
            int(198 - 120 * intensity),
            int(210 - 130 * intensity),
        )
        circle(draw, overview[idx], 2, color)
    for idx in neighbor_idx:
        draw.line((overview[query_idx][0], overview[query_idx][1], overview[idx][0], overview[idx][1]), fill=ORANGE, width=3 * SCALE)
        circle(draw, overview[idx], 6, ORANGE, (255, 255, 255))
    circle(draw, overview[query_idx], 8, GREEN, (255, 255, 255))
    draw.text((110 * SCALE, 175 * SCALE), "Whole surface", font=font(21, True), fill=INK)
    draw.text((110 * SCALE, 698 * SCALE), "Color intensity: absolute pressure error", font=font(15), fill=MUTED)

    local_points = points[local_idx]
    local_projection = map_to_box(project(local_points), (915, 225, 1490, 675), pad=0.18)
    local_lookup = {int(global_idx): i for i, global_idx in enumerate(local_idx)}
    for xy in local_projection:
        circle(draw, xy, 3, (178, 188, 199))
    qxy = local_projection[local_lookup[query_idx]]
    label_offsets = [(52, -62), (72, -12), (52, 42)]
    for rank, (idx, offset) in enumerate(zip(neighbor_idx, label_offsets), start=1):
        nxy = local_projection[local_lookup[int(idx)]]
        draw.line((qxy[0], qxy[1], nxy[0], nxy[1]), fill=ORANGE, width=4 * SCALE)
        circle(draw, nxy, 9, ORANGE, (255, 255, 255))
        label_xy = np.array([nxy[0] + offset[0] * SCALE, nxy[1] + offset[1] * SCALE])
        draw.line((nxy[0], nxy[1], label_xy[0] - 6 * SCALE, label_xy[1] + 9 * SCALE), fill=ORANGE, width=2 * SCALE)
        draw.text((label_xy[0], label_xy[1]), f"N{rank}", font=font(15, True), fill=INK)
    circle(draw, qxy, 12, GREEN, (255, 255, 255))
    query_label = np.array([qxy[0] - 22 * SCALE, qxy[1] + 58 * SCALE])
    draw.line((qxy[0], qxy[1], query_label[0] + 30 * SCALE, query_label[1] - 7 * SCALE), fill=GREEN, width=2 * SCALE)
    draw.text((query_label[0], query_label[1]), "Query", font=font(16, True), fill=INK)
    draw.text((915 * SCALE, 175 * SCALE), "Local zoom", font=font(21, True), fill=INK)
    draw.text((915 * SCALE, 698 * SCALE), "Query + 3 nearest surface neighbors", font=font(15), fill=MUTED)

    legend_y = 815
    circle(draw, np.array([105 * SCALE, legend_y * SCALE]), 8, GREEN)
    draw.text((125 * SCALE, (legend_y - 13) * SCALE), "Query point", font=font(16), fill=INK)
    circle(draw, np.array([330 * SCALE, legend_y * SCALE]), 8, ORANGE)
    draw.text((350 * SCALE, (legend_y - 13) * SCALE), "Stencil neighbors", font=font(16), fill=INK)
    draw.text((820 * SCALE, (legend_y - 13) * SCALE), "Teaching point: DoMINo combines global geometry with local surface context.", font=font(16), fill=INK)
    finish(image, output_path, 1600, 900)


def read_metrics(csv_path: Path) -> list[dict[str, float]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "case": float(row["case_id"]),
            "relative_l2": float(row["relative_l2"]),
            "area_l2": float(row["area_weighted_relative_l2"]),
            "correlation": float(row["correlation"]),
        }
        for row in rows
    ]


def bar(draw: ImageDraw.ImageDraw, x: int, base_y: int, width: int, value: float, maximum: float, color, label: str) -> None:
    height = int(330 * value / maximum)
    draw.rounded_rectangle(
        (x * SCALE, (base_y - height) * SCALE, (x + width) * SCALE, base_y * SCALE),
        radius=7 * SCALE,
        fill=color,
    )
    value_text = f"{value:.3f}"
    bbox = draw.textbbox((0, 0), value_text, font=font(18, True))
    draw.text(((x + width / 2) * SCALE - (bbox[2] - bbox[0]) / 2, (base_y - height - 34) * SCALE), value_text, font=font(18, True), fill=INK)
    bbox = draw.textbbox((0, 0), label, font=font(15))
    draw.text(((x + width / 2) * SCALE - (bbox[2] - bbox[0]) / 2, (base_y + 15) * SCALE), label, font=font(15), fill=MUTED)


def make_metrics_summary(csv_path: Path, output_path: Path) -> None:
    rows = read_metrics(csv_path)
    image, draw = canvas(1600, 780)
    draw.text((70 * SCALE, 42 * SCALE), "Held-out test metrics", font=font(30, True), fill=INK)
    draw.text((70 * SCALE, 86 * SCALE), "Two unseen DrivAerML shapes after 120 minutes on Tesla T4", font=font(17), fill=MUTED)

    panels = [(70, 145, 1000, 665), (1040, 145, 1530, 665)]
    for panel in panels:
        draw.rounded_rectangle(tuple(v * SCALE for v in panel), radius=18 * SCALE, fill=(255, 255, 255), outline=GRID, width=2 * SCALE)
    draw.text((105 * SCALE, 175 * SCALE), "Relative L2 error (lower is better)", font=font(21, True), fill=INK)
    draw.text((1075 * SCALE, 175 * SCALE), "Correlation (higher is better)", font=font(21, True), fill=INK)

    base_y = 570
    for y_tick in (0.0, 0.2, 0.4, 0.6):
        y = base_y - int(330 * y_tick / 0.6)
        draw.line((110 * SCALE, y * SCALE, 965 * SCALE, y * SCALE), fill=GRID, width=SCALE)
        draw.text((76 * SCALE, (y - 10) * SCALE), f"{y_tick:.1f}", font=font(13), fill=MUTED)
    positions = [205, 385, 625, 805]
    labels = ["C43 RelL2", "C43 Area", "C63 RelL2", "C63 Area"]
    values = [rows[0]["relative_l2"], rows[0]["area_l2"], rows[1]["relative_l2"], rows[1]["area_l2"]]
    colors = [BLUE, TEAL, BLUE, TEAL]
    for x, label, value, color in zip(positions, labels, values, colors):
        bar(draw, x, base_y, 100, value, 0.6, color, label)

    corr_base = 570
    for y_tick in (0.0, 0.4, 0.8, 1.0):
        y = corr_base - int(330 * y_tick)
        draw.line((1080 * SCALE, y * SCALE, 1490 * SCALE, y * SCALE), fill=GRID, width=SCALE)
        draw.text((1048 * SCALE, (y - 10) * SCALE), f"{y_tick:.1f}", font=font(13), fill=MUTED)
    for x, row, color in zip((1145, 1340), rows, (GREEN, ORANGE)):
        bar(draw, x, corr_base, 105, row["correlation"], 1.0, color, f"Case {int(row['case'])}")

    avg_rel = float(np.mean([row["relative_l2"] for row in rows]))
    avg_corr = float(np.mean([row["correlation"] for row in rows]))
    draw.text((70 * SCALE, 710 * SCALE), f"Mean RelL2: {avg_rel:.3f}    Mean correlation: {avg_corr:.3f}", font=font(19, True), fill=INK)
    draw.text((790 * SCALE, 710 * SCALE), "Pattern agreement is visible, but local pressure accuracy remains limited.", font=font(17), fill=MUTED)
    finish(image, output_path, 1600, 780)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/run_20260712/outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/run_20260712"))
    args = parser.parse_args()
    make_stencil_zoom(args.input_dir / "domino_surface_predictions.npz", args.output_dir / "domino_local_stencil_zoom.png")
    make_metrics_summary(args.input_dir / "domino_surface_metrics.csv", args.output_dir / "domino_metrics_summary.png")
    print(f"Generated slide assets in {args.output_dir}")


if __name__ == "__main__":
    main()
