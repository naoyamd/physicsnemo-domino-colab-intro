"""LLMが判定しやすい自己完結型DoMINO quick check Colabを生成する。

既存の教材Notebookに埋め込まれたDrivAerML派生データからrun 105だけを
取り出し、通信・認証・手動設定を必要としない短時間Notebookを構築する。
"""

from __future__ import annotations

import ast
import base64
import io
import json
import textwrap
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = PROJECT_ROOT / "notebooks" / "domino_surface_only_colab.ipynb"
OUTPUT = PROJECT_ROOT / "notebooks" / "domino_quickcheck_colab.ipynb"
CASE_ID = 105


def _source(text: str) -> list[str]:
    value = textwrap.dedent(text).strip("\n") + "\n"
    return value.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def _embedded_payload_from_source() -> bytes:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "FALLBACK_B64 =" in "".join(cell["source"])
    )
    assignment = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "FALLBACK_B64"
            for target in node.targets
        )
    )
    return base64.b64decode(ast.literal_eval(assignment.value))


def build_case_payload() -> str:
    """run 105の4配列だけを決定的なNPZとして再圧縮する。"""
    required = [
        f"case_{CASE_ID}_points.npy",
        f"case_{CASE_ID}_normals.npy",
        f"case_{CASE_ID}_areas.npy",
        f"case_{CASE_ID}_pressure.npy",
    ]
    source_buffer = io.BytesIO(_embedded_payload_from_source())
    output_buffer = io.BytesIO()
    with zipfile.ZipFile(source_buffer) as source_archive:
        missing = sorted(set(required) - set(source_archive.namelist()))
        if missing:
            raise RuntimeError(f"Source fallback is missing: {missing}")
        with zipfile.ZipFile(
            output_buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as output_archive:
            for name in required:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                output_archive.writestr(info, source_archive.read(name))
    return base64.b64encode(output_buffer.getvalue()).decode("ascii")


def build_notebook(payload_b64: str) -> dict:
    payload_cell = """
        CASE_ID = 105
        FALLBACK_CASE_B64 = __PAYLOAD__

        with np.load(
            io.BytesIO(base64.b64decode(FALLBACK_CASE_B64)), allow_pickle=False
        ) as archive:
            raw = {
                key: np.asarray(archive[f"case_{CASE_ID}_{key}"], dtype=np.float32)
                for key in ("points", "normals", "areas", "pressure")
            }

        assert raw["points"].shape == raw["normals"].shape == (4096, 3)
        assert raw["areas"].shape == raw["pressure"].shape == (4096,)
        assert all(np.isfinite(values).all() for values in raw.values())
        assert np.all(raw["areas"] > 0)

        # CPUでも短時間で通るよう、実データから512点だけを使用する。
        points = raw["points"][:512].copy()
        normals = raw["normals"][:512].copy()
        areas = raw["areas"][:512].copy()
        pressure = raw["pressure"][:512].copy()

        center = 0.5 * (points.min(axis=0) + points.max(axis=0))
        scale = 0.55 * float(np.max(points.max(axis=0) - points.min(axis=0)))
        points = ((points - center) / scale).astype(np.float32)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        areas = np.maximum(areas / scale**2, 1e-10).astype(np.float32)

        tree = cKDTree(points)
        _, neighbor_idx = tree.query(points, k=4)
        axis = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        _, nearest = tree.query(grid.reshape(-1, 3), k=1)
        delta = grid.reshape(-1, 3) - points[nearest]
        sdf = np.sum(delta * normals[nearest], axis=1).reshape(8, 8, 8).astype(np.float32)

        case = {
            "points": points,
            "normals": normals,
            "areas": areas,
            "pressure": pressure,
            "neighbor_idx": neighbor_idx[:, 1:].astype(np.int32),
            "geometry": points[:256],
            "grid": grid,
            "sdf": sdf,
        }
        print("Embedded real-data case:", CASE_ID)
        print("Surface points used:", len(points))
    """.replace("__PAYLOAD__", repr(payload_b64))

    cells = [
        markdown(
            """
            <a href="https://colab.research.google.com/github/naoyamd/physicsnemo-domino-colab-intro/blob/main/notebooks/domino_quickcheck_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

            # DoMINO Quick Check：Colab + LLM向け疎通確認

            このNotebookは、PhysicsNeMoの公式 `DoMINO` クラスにDrivAerML由来の実データを入力し、**forward → backward → 1 optimizer step** が通ることだけを短時間で検証します。

            使い方は **`ランタイム` → `すべてのセルを実行`** だけです。設定変更、Google Drive、Hugging Face、APIキーは不要です。GPUがあれば自動利用し、なければCPUで実行します。

            最終セルに次の2点があれば合格です。

            1. `quickcheck_result.json` の `status` が `pass`
            2. 最終行が `DOMINO_QUICKCHECK: PASS`

            > これは実装経路の疎通確認です。予測精度、CFDの妥当性、フルスケールDoMINoの再現は評価しません。
            """
        ),
        code(
            """
            import time

            QUICKCHECK_STARTED = time.perf_counter()
            %pip install -q "nvidia-physicsnemo==2.1.1" "scipy>=1.12" "matplotlib>=3.8"
            """
        ),
        code(
            """
            import base64
            import io
            import json
            import platform
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import physicsnemo
            import torch
            from scipy.spatial import cKDTree

            from physicsnemo.models.domino.config import Config, DEFAULT_MODEL_PARAMS
            from physicsnemo.models.domino.model import DoMINO

            SEED = 2026
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            runtime_root = (
                Path("/content") if Path("/content").exists() else Path.cwd() / "outputs"
            )
            OUTPUT_DIR = runtime_root / "domino_quickcheck"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            print("Python:", sys.version.split()[0])
            print("Torch:", torch.__version__)
            print("PhysicsNeMo:", getattr(physicsnemo, "__version__", "unknown"))
            print("Device:", DEVICE)
            if DEVICE.type == "cuda":
                print("GPU:", torch.cuda.get_device_name(0))
            """
        ),
        code(payload_cell),
        code(
            """
            def model_config():
                cfg = Config(json.loads(json.dumps(DEFAULT_MODEL_PARAMS)))
                cfg.model_type = "surface"
                cfg.interp_res = [8, 8, 8]
                cfg.geometry_encoding_type = "both"
                cfg.num_neighbors_surface = 4
                cfg.use_surface_normals = True
                cfg.use_surface_area = True
                cfg.encode_parameters = False
                cfg.geometry_rep.geo_conv.base_neurons = 8
                cfg.geometry_rep.geo_conv.base_neurons_in = 1
                cfg.geometry_rep.geo_conv.base_neurons_out = 1
                cfg.geometry_rep.geo_conv.surface_radii = [0.2]
                cfg.geometry_rep.geo_conv.surface_neighbors_in_radius = [4]
                cfg.geometry_rep.geo_conv.volume_radii = [0.2]
                cfg.geometry_rep.geo_conv.volume_neighbors_in_radius = [4]
                cfg.geometry_rep.geo_processor.base_filters = 2
                cfg.geometry_rep.geo_processor.surface_sdf_scaling_factor = [0.1]
                cfg.geometry_local.surface_radii = [0.4]
                cfg.geometry_local.surface_neighbors_in_radius = [4]
                cfg.geometry_local.volume_radii = [0.4]
                cfg.geometry_local.volume_neighbors_in_radius = [4]
                cfg.nn_basis_functions.base_layer = 16
                cfg.nn_basis_functions.num_modes = 2
                cfg.position_encoder.base_neurons = 16
                cfg.position_encoder.num_modes = 2
                cfg.aggregation_model.base_layer = 16
                return cfg

            def tensor(value):
                return torch.as_tensor(
                    value, dtype=torch.float32, device=DEVICE
                ).unsqueeze(0)

            rng = np.random.default_rng(SEED)
            query = np.sort(rng.choice(len(case["points"]), size=16, replace=False))
            neighbors = case["neighbor_idx"][query]
            pressure_mean = float(case["pressure"].mean())
            pressure_std = max(float(case["pressure"].std()), 1e-6)

            inputs = {
                "geometry_coordinates": tensor(case["geometry"]),
                "surf_grid": tensor(case["grid"]),
                "sdf_surf_grid": tensor(case["sdf"]),
                "pos_surface_center_of_mass": tensor(case["points"][query]),
                "surface_mesh_centers": tensor(case["points"][query]),
                "surface_mesh_neighbors": tensor(case["points"][neighbors]),
                "surface_normals": tensor(case["normals"][query]),
                "surface_neighbors_normals": tensor(case["normals"][neighbors]),
                "surface_areas": tensor(case["areas"][query]),
                "surface_neighbors_areas": tensor(case["areas"][neighbors]),
                "global_params_values": torch.ones((1, 1, 1), device=DEVICE),
                "global_params_reference": torch.ones((1, 1, 1), device=DEVICE),
            }
            target = tensor((case["pressure"][query] - pressure_mean) / pressure_std)

            model = DoMINO(
                input_features=3,
                output_features_vol=None,
                output_features_surf=1,
                global_features=1,
                model_parameters=model_config(),
            ).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            trainable_parameters = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            print("Official class:", f"{DoMINO.__module__}.{DoMINO.__name__}")
            print("Trainable parameters:", trainable_parameters)
            """
        ),
        markdown(
            """
            ## 実行と機械判定

            以降は、出力shape、損失、勾配、パラメータ更新、更新後予測がすべて正常な場合だけ `PASS` を出します。数値の大小は予測精度を意味しません。
            """
        ),
        code(
            """
            model.train()
            trainable = [p for p in model.parameters() if p.requires_grad]
            before = [p.detach().clone() for p in trainable]
            optimizer.zero_grad(set_to_none=True)

            _, prediction_before = model(inputs)
            loss_before = torch.mean((prediction_before.squeeze(-1) - target) ** 2)
            loss_before.backward()

            gradients = [p.grad for p in trainable if p.grad is not None]
            gradients_finite = bool(gradients) and all(
                torch.isfinite(gradient).all().item() for gradient in gradients
            )
            gradient_energy = sum(
                (torch.sum(gradient.detach() ** 2) for gradient in gradients),
                torch.zeros((), device=DEVICE),
            )
            gradient_norm = float(torch.sqrt(gradient_energy).cpu())
            optimizer.step()
            parameter_delta_energy = sum(
                (
                    torch.sum((parameter.detach() - initial) ** 2)
                    for parameter, initial in zip(trainable, before)
                ),
                torch.zeros((), device=DEVICE),
            )
            parameter_delta = float(torch.sqrt(parameter_delta_energy).cpu())

            model.eval()
            with torch.no_grad():
                _, prediction_after = model(inputs)
                loss_after = torch.mean((prediction_after.squeeze(-1) - target) ** 2)

            expected_shape = (1, len(query), 1)
            assertions = {
                "output_shape_ok": tuple(prediction_before.shape) == expected_shape,
                "forward_finite": bool(torch.isfinite(prediction_before).all().item()),
                "loss_finite": bool(torch.isfinite(loss_before).item()),
                "gradients_finite": gradients_finite,
                "gradient_norm_finite": bool(torch.isfinite(gradient_energy).item()),
                "gradient_nonzero": gradient_norm > 0.0,
                "parameters_finite": all(
                    torch.isfinite(parameter).all().item() for parameter in trainable
                ),
                "parameter_delta_finite": bool(
                    torch.isfinite(parameter_delta_energy).item()
                ),
                "parameter_updated": parameter_delta > 0.0,
                "post_update_finite": bool(torch.isfinite(prediction_after).all().item()),
            }
            failed = [name for name, passed in assertions.items() if not passed]
            if failed:
                raise RuntimeError(f"Quick check failed: {failed}")

            reference = target.squeeze(0).detach().cpu().numpy()
            predicted = prediction_after.squeeze().detach().cpu().numpy()
            figure, axis = plt.subplots(figsize=(6.5, 4.0))
            axis.plot(reference, "o-", label="standardized reference", linewidth=1.4)
            axis.plot(predicted, "s-", label="prediction after one step", linewidth=1.2)
            axis.set_xlabel("Query point index")
            axis.set_ylabel("Standardized pressure")
            axis.set_title("DoMINO quick-check output (not an accuracy result)")
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout()
            figure_path = OUTPUT_DIR / "domino_quickcheck_prediction.png"
            figure.savefig(figure_path, dpi=160, bbox_inches="tight")
            plt.show()

            result = {
                "schema_version": 1,
                "status": "pass",
                "test": "official-domino-forward-backward-update",
                "official_class": f"{DoMINO.__module__}.{DoMINO.__name__}",
                "data": {
                    "source": "DrivAerML-derived embedded real data",
                    "license": "CC BY-SA 4.0",
                    "case_id": CASE_ID,
                    "surface_points_used": len(case["points"]),
                    "query_points": len(query),
                },
                "environment": {
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "physicsnemo": getattr(physicsnemo, "__version__", "unknown"),
                    "device": str(DEVICE),
                },
                "checks": assertions,
                "measurements": {
                    "output_shape": list(prediction_before.shape),
                    "loss_before": float(loss_before.detach().cpu()),
                    "loss_after": float(loss_after.detach().cpu()),
                    "gradient_norm": gradient_norm,
                    "parameter_delta": parameter_delta,
                    "trainable_parameters": trainable_parameters,
                    "elapsed_seconds_including_install": time.perf_counter() - QUICKCHECK_STARTED,
                },
                "artifacts": {
                    "result_json": str(OUTPUT_DIR / "quickcheck_result.json"),
                    "prediction_png": str(figure_path),
                },
            }
            result_path = OUTPUT_DIR / "quickcheck_result.json"
            result_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )

            print(
                "DOMINO_QUICKCHECK_RESULT="
                + json.dumps(result, ensure_ascii=False, allow_nan=False)
            )
            print("Result file:", result_path)
            print("DOMINO_QUICKCHECK: PASS")
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook(build_case_payload())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
