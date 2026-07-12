"""surface-only DoMINo入門用の自己完結Colabノートブックを生成する。

生成ノートブックには、CC BY-SA 4.0のDrivAerML派生データから抽出した
小さな実データfallbackを埋め込む。通常実行では、より多くのsurface-only
ケースを実行時に取得する。
"""

from __future__ import annotations

import base64
import io
import json
import textwrap
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "notebooks" / "domino_surface_only_colab.ipynb"
DATASET_REPO = "EmmiAI/DrivAerML_subsampled_10x"
FALLBACK_CASE_IDS = (105, 130, 202)
FALLBACK_POINTS = 4096
REQUIRED_FILES = {
    "points": "surface_position_vtp.pt",
    "normals": "surface_normal_vtp.pt",
    "areas": "surface_area_vtp.pt",
    "pressure": "surface_pressure.pt",
}


def _source(text: str) -> list[str]:
    value = textwrap.dedent(text).strip("\n") + "\n"
    return value.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict:
    text = text.replace("{{", "{").replace("}}", "}")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def build_fallback_payload() -> str:
    arrays: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(20260712)

    for case_id in FALLBACK_CASE_IDS:
        loaded: dict[str, np.ndarray] = {}
        for key, filename in REQUIRED_FILES.items():
            path = hf_hub_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                filename=f"run_{case_id}/{filename}",
                local_dir="/tmp/domino_surface_fallback_source",
            )
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            loaded[key] = tensor.detach().cpu().numpy()

        n = len(loaded["points"])
        indices = rng.choice(n, size=min(FALLBACK_POINTS, n), replace=False)
        for key, values in loaded.items():
            arrays[f"case_{case_id}_{key}"] = np.asarray(
                values[indices], dtype=np.float32
            )

    arrays["case_ids"] = np.asarray(FALLBACK_CASE_IDS, dtype=np.int32)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_notebook(fallback_b64: str) -> dict:
    cells = [
        markdown(
            """
            <a href="https://colab.research.google.com/github/naoyamd/physicsnemo-domino-colab-intro/blob/main/notebooks/domino_surface_only_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

            # PhysicsNeMo DoMINo入門：車両表面圧力をColabで学習する

            このノートブックは、PhysicsNeMoの公式 `DoMINO` クラスを **surface-only** モードで実際に学習します。DrivAerML由来の3次元車両表面点群から表面圧力を予測し、正解・予測・誤差を同じ車体上で可視化します。

            - 通常実行：12〜16形状のsurfaceデータ、約2時間の学習
            - fallback：ノート内に収録した3形状の実データで短時間確認
            - 対象外：volume場、フルスケール再現、NIM呼び出し、工学的な精度保証

            > DoMINoはCFDソルバーではなく、CFD結果から写像を学ぶサロゲートです。このノートの結果は教材用の縮小実験です。
            """
        ),
        markdown(
            """
            ## 実験の流れ

            1. GPUとPhysicsNeMoの確認
            2. surface位置・法線・面積・圧力の取得
            3. 点群から正規化座標、近傍ステンシル、近似SDFを作成
            4. surface-only DoMINoをwall-clock制御で学習
            5. 未学習形状で推論し、3D圧力分布と誤差を保存

            最初は `RUN_MODE="fallback"`、`TRAIN_MINUTES=3` で全セルを通してください。その後 `RUN_MODE="full"`、`TRAIN_MINUTES=120` に変更して本実験を行います。
            """
        ),
        code(
            """
            import platform, shutil, sys

            print("Python:", sys.version.split()[0])
            print("Platform:", platform.platform())
            print("Disk free [GB]:", round(shutil.disk_usage("/content").free / 1e9, 1))
            if sys.version_info < (3, 12):
                raise RuntimeError("このノートブックはPython 3.12以上を前提とします。")

            """
        ),
        code(
            """
            %pip install -q "nvidia-physicsnemo==2.1.1" "huggingface-hub>=0.30" "scipy>=1.12" "matplotlib>=3.8" "pandas>=2.2"
            """
        ),
        code(
            """
            import base64
            import io
            import json
            import math
            import os
            import random
            import time
            from contextlib import nullcontext
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import torch
            from huggingface_hub import HfApi, hf_hub_download
            from scipy.spatial import cKDTree

            from physicsnemo.models.domino.config import Config, DEFAULT_MODEL_PARAMS
            from physicsnemo.models.domino.model import DoMINO

            if not torch.cuda.is_available():
                raise RuntimeError("GPUランタイムが必要です。Colabのランタイム設定でGPUを選択してください。")

            DEVICE = torch.device("cuda")
            print("Torch:", torch.__version__)
            print("PhysicsNeMo DoMINO:", DoMINO.__module__)
            print("GPU:", torch.cuda.get_device_name(0))
            """
        ),
        markdown(
            """
            ## 設定

            `fallback` はノート内の実データ3形状を使うため、データ取得の通信が不要です。`full` は公開Hugging Faceデータセットからsurfaceファイルだけを取得します。Google Driveを使うと、前処理結果とcheckpointを再利用できます。
            """
        ),
        code(
            """
            #@title 実行設定
            RUN_MODE = "fallback"  #@param ["fallback", "full"]
            TRAIN_MINUTES = 3  #@param {type:"integer"}
            USE_GOOGLE_DRIVE = True  #@param {type:"boolean"}
            RESUME = True  #@param {type:"boolean"}

            SEED = 2026
            FULL_CASE_COUNT = 14
            POINTS_PER_CASE = 8192
            GEOMETRY_POINTS = 4096
            QUERY_POINTS = 1024
            GRID_RES = 24
            STENCIL_POINTS = 4  # center + 3 neighbors
            AMP = True
            CHECKPOINT_EVERY_MIN = 30
            LOG_EVERY_SEC = 60

            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)

            drive_root = None
            if USE_GOOGLE_DRIVE:
                try:
                    from google.colab import drive
                    drive.mount("/content/drive")
                    drive_root = Path("/content/drive/MyDrive/physicsnemo_domino_surface")
                except Exception as exc:
                    print("Driveを使用しません:", type(exc).__name__, exc)

            ROOT = drive_root or Path("/content/physicsnemo_domino_surface")
            CACHE_DIR = ROOT / "cache"
            CHECKPOINT_DIR = ROOT / "checkpoints"
            OUTPUT_DIR = ROOT / "outputs"
            RAW_DIR = Path("/content/domino_surface_raw")
            for path in (CACHE_DIR, CHECKPOINT_DIR, OUTPUT_DIR, RAW_DIR):
                path.mkdir(parents=True, exist_ok=True)

            print("Run mode:", RUN_MODE)
            print("Training budget [min]:", TRAIN_MINUTES)
            print("Workspace:", ROOT)
            """
        ),
        markdown(
            """
            ## データ取得とfallback

            主データは `EmmiAI/DrivAerML_subsampled_10x` のsurface位置、法線、面積、圧力です。元データと派生データのライセンスはCC BY-SA 4.0です。`.pt` は `weights_only=True` で読み込みます。

            埋め込みfallbackはrun 105、130、202から各4,096点を固定seedで抽出した実データです。合成圧力は使用しません。
            """
        ),
        code(
            f"""
            FALLBACK_B64 = {fallback_b64!r}
            DATASET_REPO = "EmmiAI/DrivAerML_subsampled_10x"
            REQUIRED_FILES = {{
                "points": "surface_position_vtp.pt",
                "normals": "surface_normal_vtp.pt",
                "areas": "surface_area_vtp.pt",
                "pressure": "surface_pressure.pt",
            }}

            def load_embedded_fallback():
                payload = base64.b64decode(FALLBACK_B64.encode("ascii"))
                archive = np.load(io.BytesIO(payload), allow_pickle=False)
                cases = []
                for case_id in archive["case_ids"].astype(int).tolist():
                    case = {{"case_id": int(case_id)}}
                    case.update({{
                        key: np.asarray(
                            archive[f"case_{{case_id}}_{{key}}"], dtype=np.float32
                        )
                        for key in REQUIRED_FILES
                    }})
                    cases.append(case)
                return cases

            def discover_full_case_ids(count):
                files = HfApi().list_repo_files(DATASET_REPO, repo_type="dataset")
                available = {{}}
                for filename in files:
                    parts = filename.split("/")
                    if len(parts) != 2 or not parts[0].startswith("run_"):
                        continue
                    try:
                        case_id = int(parts[0].split("_")[1])
                    except ValueError:
                        continue
                    available.setdefault(case_id, set()).add(parts[1])
                required = set(REQUIRED_FILES.values())
                case_ids = sorted(k for k, names in available.items() if required <= names)
                rng = np.random.default_rng(SEED)
                rng.shuffle(case_ids)
                if len(case_ids) < count:
                    raise RuntimeError(f"必要なsurfaceファイルを持つケースが{{len(case_ids)}}件しかありません。")
                return case_ids[:count]

            def download_raw_case(case_id):
                loaded = {{"case_id": int(case_id)}}
                for key, filename in REQUIRED_FILES.items():
                    path = hf_hub_download(
                        repo_id=DATASET_REPO,
                        repo_type="dataset",
                        filename=f"run_{{case_id}}/{{filename}}",
                        local_dir=RAW_DIR,
                        token=os.environ.get("HF_TOKEN") or None,
                    )
                    tensor = torch.load(path, map_location="cpu", weights_only=True)
                    loaded[key] = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                return loaded

            def load_raw_cases():
                if RUN_MODE == "fallback":
                    return load_embedded_fallback(), "embedded-real-data"
                try:
                    case_ids = discover_full_case_ids(FULL_CASE_COUNT)
                    print("Selected case IDs:", case_ids)
                    return [download_raw_case(case_id) for case_id in case_ids], "huggingface"
                except Exception as exc:
                    print("Full data取得に失敗したため、実データfallbackへ切り替えます。")
                    print(type(exc).__name__, exc)
                    return load_embedded_fallback(), "embedded-real-data-fallback"

            raw_cases, data_source_mode = load_raw_cases()
            print("Data source mode:", data_source_mode)
            print("Cases:", [c["case_id"] for c in raw_cases])
            print("Points in first raw case:", len(raw_cases[0]["points"]))
            """
        ),
        markdown(
            """
            ## 前処理：点群、local stencil、近似SDF

            各車両を中心化し、最大寸法で `[-1, 1]` 付近へ正規化します。近傍ステンシルはsurface点群上のk近傍から作ります。SDFは最寄りsurface点とその法線から符号付き距離を近似します。

            これはColab教材向けの簡略化です。公式の大規模前処理を再現するものではありません。
            """
        ),
        code(
            """
            def regular_grid(resolution):
                axis = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
                return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)

            def save_case_npz(path, case):
                tmp = path.with_suffix(path.suffix + ".tmp")
                with open(tmp, "wb") as handle:
                    np.savez_compressed(handle, **case)
                os.replace(tmp, path)

            def load_case_npz(path):
                with np.load(path, allow_pickle=False) as archive:
                    return {{key: archive[key] for key in archive.files}}

            def preprocess_case(raw):
                case_id = int(raw["case_id"])
                requested = min(POINTS_PER_CASE, len(raw["points"]))
                cache_path = CACHE_DIR / f"case_{{case_id}}_n{{requested}}_g{{GRID_RES}}_v2.npz"
                if cache_path.exists():
                    return load_case_npz(cache_path)

                points = np.asarray(raw["points"], dtype=np.float32)
                normals = np.asarray(raw["normals"], dtype=np.float32)
                areas = np.asarray(raw["areas"], dtype=np.float32).reshape(-1)
                pressure = np.asarray(raw["pressure"], dtype=np.float32).reshape(-1)
                valid = (
                    np.isfinite(points).all(axis=1)
                    & np.isfinite(normals).all(axis=1)
                    & np.isfinite(areas)
                    & np.isfinite(pressure)
                    & (areas > 0)
                )
                points, normals, areas, pressure = (
                    points[valid], normals[valid], areas[valid], pressure[valid]
                )

                rng = np.random.default_rng(SEED + case_id)
                take = rng.choice(len(points), size=min(POINTS_PER_CASE, len(points)), replace=False)
                points, normals, areas, pressure = (
                    points[take], normals[take], areas[take], pressure[take]
                )

                pmin, pmax = points.min(axis=0), points.max(axis=0)
                center = 0.5 * (pmin + pmax)
                scale = float(0.55 * np.max(pmax - pmin))
                points_n = ((points - center) / scale).astype(np.float32)
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
                areas_n = np.maximum(areas / (scale * scale), 1e-10).astype(np.float32)

                tree = cKDTree(points_n)
                _, neighbor_idx = tree.query(points_n, k=STENCIL_POINTS)
                neighbor_idx = np.asarray(neighbor_idx[:, 1:], dtype=np.int32)

                grid = regular_grid(GRID_RES)
                grid_flat = grid.reshape(-1, 3)
                _, nearest = tree.query(grid_flat, k=1)
                delta = grid_flat - points_n[nearest]
                signed = np.sum(delta * normals[nearest], axis=1)
                sdf = signed.reshape(GRID_RES, GRID_RES, GRID_RES).astype(np.float32)

                geo_count = min(GEOMETRY_POINTS, len(points_n))
                geo_idx = rng.choice(len(points_n), size=geo_count, replace=False)
                case = {{
                    "case_id": np.asarray(case_id, dtype=np.int32),
                    "points": points_n.astype(np.float32),
                    "normals": normals.astype(np.float32),
                    "areas": areas_n,
                    "pressure": pressure.astype(np.float32),
                    "neighbor_idx": neighbor_idx,
                    "geometry": points_n[geo_idx].astype(np.float32),
                    "grid": grid.astype(np.float32),
                    "sdf": sdf,
                    "physical_center": center.astype(np.float32),
                    "physical_scale": np.asarray(scale, dtype=np.float32),
                }}
                save_case_npz(cache_path, case)
                return case

            cases = [preprocess_case(raw) for raw in raw_cases]
            del raw_cases

            rng = np.random.default_rng(SEED)
            order = np.arange(len(cases))
            rng.shuffle(order)
            cases = [cases[i] for i in order]
            if len(cases) >= 6:
                train_cases, val_cases, test_cases = cases[:-4], cases[-4:-2], cases[-2:]
            else:
                train_cases, val_cases, test_cases = cases[:-1], [], cases[-1:]

            train_pressure = np.concatenate([c["pressure"] for c in train_cases])
            PRESSURE_MEAN = float(train_pressure.mean())
            PRESSURE_STD = float(max(train_pressure.std(), 1e-6))
            del train_pressure

            print("Split:", {{
                "train": [int(c["case_id"]) for c in train_cases],
                "val": [int(c["case_id"]) for c in val_cases],
                "test": [int(c["case_id"]) for c in test_cases],
            }})
            print("Train pressure mean/std [dataset units]:", PRESSURE_MEAN, PRESSURE_STD)
            print("Grid:", cases[0]["grid"].shape, "points:", cases[0]["points"].shape)
            """
        ),
        markdown(
            """
            ## 小型surface-only DoMINo

            モデル本体はPhysicsNeMo 2.1.1の `physicsnemo.models.domino.model.DoMINO` です。surfaceとvolumeを同時に扱わず、surface pressure 1変数だけを予測します。global表現はSDFとsurface点群の両方、local表現は2スケールを使用します。
            """
        ),
        code(
            """
            def tiny_domino_config():
                cfg = Config(json.loads(json.dumps(DEFAULT_MODEL_PARAMS)))
                cfg.model_type = "surface"
                cfg.interp_res = [GRID_RES, GRID_RES, GRID_RES]
                cfg.geometry_encoding_type = "both"
                cfg.solution_calculation_mode = "two-loop"
                cfg.num_neighbors_surface = STENCIL_POINTS
                cfg.use_surface_normals = True
                cfg.use_surface_area = True
                cfg.encode_parameters = False
                cfg.combine_volume_surface = False

                cfg.geometry_rep.geo_conv.base_neurons = 16
                cfg.geometry_rep.geo_conv.base_neurons_in = 1
                cfg.geometry_rep.geo_conv.base_neurons_out = 1
                cfg.geometry_rep.geo_conv.surface_hops = 1
                cfg.geometry_rep.geo_conv.surface_radii = [0.08, 0.20]
                cfg.geometry_rep.geo_conv.surface_neighbors_in_radius = [8, 16]
                cfg.geometry_rep.geo_conv.volume_radii = [0.08]
                cfg.geometry_rep.geo_conv.volume_neighbors_in_radius = [8]
                cfg.geometry_rep.geo_processor.base_filters = 4
                cfg.geometry_rep.geo_processor.self_attention = False
                cfg.geometry_rep.geo_processor.cross_attention = False
                cfg.geometry_rep.geo_processor.surface_sdf_scaling_factor = [0.03, 0.10]

                cfg.geometry_local.surface_radii = [0.12, 0.30]
                cfg.geometry_local.surface_neighbors_in_radius = [8, 16]
                cfg.geometry_local.volume_radii = [0.12]
                cfg.geometry_local.volume_neighbors_in_radius = [8]
                cfg.nn_basis_functions.base_layer = 64
                cfg.nn_basis_functions.num_modes = 3
                cfg.position_encoder.base_neurons = 64
                cfg.position_encoder.num_modes = 3
                cfg.aggregation_model.base_layer = 64
                cfg.parameter_model.base_layer = 32
                return cfg

            MODEL_CFG = tiny_domino_config()
            model = DoMINO(
                input_features=3,
                output_features_vol=None,
                output_features_surf=1,
                global_features=1,
                model_parameters=MODEL_CFG,
            ).to(DEVICE)

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Model:", model.__class__.__module__ + "." + model.__class__.__name__)
            print("Mode: surface-only; output: pressure")
            print("Trainable parameters:", f"{{trainable:,}}")
            """
        ),
        code(
            """
            def tensor(x, add_batch=True):
                value = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
                return value.unsqueeze(0) if add_batch else value

            def make_batch(case, query_count, rng):
                n = len(case["points"])
                q = rng.choice(n, size=min(query_count, n), replace=False)
                neighbors = case["neighbor_idx"][q]
                inputs = {{
                    "geometry_coordinates": tensor(case["geometry"]),
                    "surf_grid": tensor(case["grid"]),
                    "sdf_surf_grid": tensor(case["sdf"]),
                    "pos_surface_center_of_mass": tensor(case["points"][q]),
                    "surface_mesh_centers": tensor(case["points"][q]),
                    "surface_mesh_neighbors": tensor(case["points"][neighbors]),
                    "surface_normals": tensor(case["normals"][q]),
                    "surface_neighbors_normals": tensor(case["normals"][neighbors]),
                    "surface_areas": tensor(case["areas"][q]),
                    "surface_neighbors_areas": tensor(case["areas"][neighbors]),
                    "global_params_values": torch.ones((1, 1, 1), device=DEVICE),
                    "global_params_reference": torch.ones((1, 1, 1), device=DEVICE),
                }}
                target = (case["pressure"][q] - PRESSURE_MEAN) / PRESSURE_STD
                return inputs, tensor(target), q

            smoke_rng = np.random.default_rng(SEED)
            smoke_inputs, smoke_target, _ = make_batch(train_cases[0], min(128, QUERY_POINTS), smoke_rng)
            model.train()
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                _, smoke_prediction = model(smoke_inputs)
                smoke_loss = torch.mean((smoke_prediction.squeeze(-1) - smoke_target) ** 2)
            smoke_loss.backward()
            model.zero_grad(set_to_none=True)
            print("Smoke output:", tuple(smoke_prediction.shape))
            print("Smoke loss:", float(smoke_loss.detach().cpu()))
            print("Peak GPU memory [GB]:", round(torch.cuda.max_memory_allocated() / 1e9, 2))
            """
        ),
        markdown(
            """
            ## Wall-clock制御学習

            epoch数ではなく経過時間で停止します。Drive利用時は30分ごとにatomic checkpointとmanifestを保存します。再開時は、それまでの累積学習時間から続行します。
            """
        ),
        code(
            """
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
            scaler = torch.amp.GradScaler("cuda", enabled=AMP)
            checkpoint_tag = "full" if data_source_mode == "huggingface" else "fallback"
            checkpoint_path = CHECKPOINT_DIR / f"domino_surface_{{checkpoint_tag}}_g{{GRID_RES}}.pt"
            manifest_path = CHECKPOINT_DIR / f"manifest_{{checkpoint_tag}}.json"
            history = []
            step = 0
            elapsed_before = 0.0

            def atomic_json(path, payload):
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
                os.replace(tmp, path)

            def save_checkpoint(total_elapsed):
                payload = {{
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "elapsed_sec": total_elapsed,
                    "history": history,
                    "pressure_mean": PRESSURE_MEAN,
                    "pressure_std": PRESSURE_STD,
                    "train_case_ids": [int(c["case_id"]) for c in train_cases],
                    "grid_res": GRID_RES,
                    "query_points": QUERY_POINTS,
                }}
                tmp = checkpoint_path.with_suffix(".pt.tmp")
                torch.save(payload, tmp)
                os.replace(tmp, checkpoint_path)
                atomic_json(manifest_path, {{
                    "status": "running",
                    "step": step,
                    "elapsed_min": total_elapsed / 60,
                    "updated_unix": time.time(),
                    "checkpoint": str(checkpoint_path),
                }})

            if RESUME and checkpoint_path.exists():
                saved = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
                if saved.get("grid_res") == GRID_RES:
                    model.load_state_dict(saved["model"])
                    optimizer.load_state_dict(saved["optimizer"])
                    scaler.load_state_dict(saved["scaler"])
                    step = int(saved.get("step", 0))
                    elapsed_before = float(saved.get("elapsed_sec", 0.0))
                    history = list(saved.get("history", []))
                    print(f"Resume: step={{step}}, cumulative={{elapsed_before/60:.1f}} min")
                else:
                    print("Checkpointのgrid設定が異なるため、resumeしません。")

            budget_sec = max(1, TRAIN_MINUTES * 60)
            train_start = time.perf_counter()
            last_log = train_start
            last_checkpoint_total = elapsed_before
            train_rng = np.random.default_rng(SEED + step)
            model.train()

            while elapsed_before + (time.perf_counter() - train_start) < budget_sec:
                case = train_cases[int(train_rng.integers(0, len(train_cases)))]
                inputs, target, _ = make_batch(case, QUERY_POINTS, train_rng)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                    _, prediction = model(inputs)
                    prediction = prediction.squeeze(-1)
                    loss = torch.mean((prediction - target) ** 2)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                step += 1

                now = time.perf_counter()
                total_elapsed = elapsed_before + (now - train_start)
                if now - last_log >= LOG_EVERY_SEC or step == 1:
                    row = {{
                        "step": step,
                        "elapsed_min": total_elapsed / 60,
                        "loss": float(loss.detach().cpu()),
                        "case_id": int(case["case_id"]),
                    }}
                    history.append(row)
                    print(row)
                    last_log = now
                    atomic_json(manifest_path, {{
                        "status": "running",
                        **row,
                        "updated_unix": time.time(),
                    }})
                if total_elapsed - last_checkpoint_total >= CHECKPOINT_EVERY_MIN * 60:
                    save_checkpoint(total_elapsed)
                    last_checkpoint_total = total_elapsed

            total_elapsed = elapsed_before + (time.perf_counter() - train_start)
            save_checkpoint(total_elapsed)
            atomic_json(manifest_path, {{
                "status": "training_complete",
                "step": step,
                "elapsed_min": total_elapsed / 60,
                "updated_unix": time.time(),
                "checkpoint": str(checkpoint_path),
            }})
            print(f"Training complete: {{step}} steps, {{total_elapsed/60:.1f}} cumulative min")
            """
        ),
        markdown(
            """
            ## 未学習形状で評価

            pressureは学習ケースだけで求めた平均・標準偏差で標準化し、表示時に元のデータ単位へ戻します。指標はrelative L2、面積重み付きrelative L2、MAE、RMSE、相関係数です。
            """
        ),
        code(
            """
            @torch.no_grad()
            def predict_case(case, chunk=QUERY_POINTS):
                model.eval()
                outputs = []
                eval_rng = np.random.default_rng(SEED + int(case["case_id"]))
                indices = np.arange(len(case["points"]))
                for start in range(0, len(indices), chunk):
                    q = indices[start : start + chunk]
                    neighbors = case["neighbor_idx"][q]
                    inputs = {{
                        "geometry_coordinates": tensor(case["geometry"]),
                        "surf_grid": tensor(case["grid"]),
                        "sdf_surf_grid": tensor(case["sdf"]),
                        "pos_surface_center_of_mass": tensor(case["points"][q]),
                        "surface_mesh_centers": tensor(case["points"][q]),
                        "surface_mesh_neighbors": tensor(case["points"][neighbors]),
                        "surface_normals": tensor(case["normals"][q]),
                        "surface_neighbors_normals": tensor(case["normals"][neighbors]),
                        "surface_areas": tensor(case["areas"][q]),
                        "surface_neighbors_areas": tensor(case["areas"][neighbors]),
                        "global_params_values": torch.ones((1, 1, 1), device=DEVICE),
                        "global_params_reference": torch.ones((1, 1, 1), device=DEVICE),
                    }}
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                        _, pred = model(inputs)
                    outputs.append(pred.squeeze().float().cpu().numpy())
                standardized = np.concatenate(outputs)
                return standardized * PRESSURE_STD + PRESSURE_MEAN

            def metrics(case, pred):
                true = case["pressure"].astype(np.float64)
                pred = pred.astype(np.float64)
                area = case["areas"].astype(np.float64)
                err = pred - true
                rel_l2 = np.linalg.norm(err) / max(np.linalg.norm(true), 1e-12)
                aw_rel_l2 = math.sqrt(np.sum(area * err**2) / max(np.sum(area * true**2), 1e-12))
                corr = np.corrcoef(true, pred)[0, 1] if np.std(pred) > 0 else np.nan
                return {{
                    "case_id": int(case["case_id"]),
                    "relative_l2": rel_l2,
                    "area_weighted_relative_l2": aw_rel_l2,
                    "mae": np.mean(np.abs(err)),
                    "rmse": math.sqrt(np.mean(err**2)),
                    "correlation": corr,
                }}

            predictions = {{}}
            metric_rows = []
            for case in test_cases:
                case_id = int(case["case_id"])
                predictions[case_id] = predict_case(case)
                metric_rows.append(metrics(case, predictions[case_id]))

            metrics_df = pd.DataFrame(metric_rows)
            metrics_path = OUTPUT_DIR / "domino_surface_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            display(metrics_df)
            print("Saved:", metrics_path)
            """
        ),
        code(
            """
            if history:
                h = pd.DataFrame(history)
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.plot(h["elapsed_min"], h["loss"], color="#76B900", lw=2)
                ax.set_yscale("log")
                ax.set_xlabel("Cumulative training time [min]")
                ax.set_ylabel("Standardized pressure MSE")
                ax.grid(alpha=0.25)
                ax.set_title("Tiny surface-only DoMINo training history")
                fig.tight_layout()
                curve_path = OUTPUT_DIR / "domino_training_curve.png"
                fig.savefig(curve_path, dpi=180, bbox_inches="tight")
                plt.show()
                print("Saved:", curve_path)
            """
        ),
        code(
            """
            case = test_cases[0]
            case_id = int(case["case_id"])
            xyz = case["points"]
            true = case["pressure"]
            pred = predictions[case_id]
            error = np.abs(pred - true)
            lo, hi = np.percentile(np.concatenate([true, pred]), [2, 98])
            err_hi = max(np.percentile(error, 98), 1e-6)

            fig = plt.figure(figsize=(16, 5.2))
            panels = [
                (true, "Reference surface pressure", "coolwarm", lo, hi),
                (pred, "DoMINo prediction", "coolwarm", lo, hi),
                (error, "Absolute error", "magma", 0, err_hi),
            ]
            for index, (values, title, cmap, vmin, vmax) in enumerate(panels, 1):
                ax = fig.add_subplot(1, 3, index, projection="3d")
                scatter = ax.scatter(
                    xyz[:, 0], xyz[:, 1], xyz[:, 2], c=values,
                    s=1.2, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0,
                )
                ax.view_init(elev=22, azim=-120)
                ax.set_box_aspect(np.ptp(xyz, axis=0))
                ax.set_axis_off()
                ax.set_title(title)
                fig.colorbar(scatter, ax=ax, shrink=0.62, pad=0.02)
            fig.suptitle(f"Held-out DrivAerML-derived case {{case_id}} (surface-only study)", y=0.98)
            fig.tight_layout()
            field_path = OUTPUT_DIR / "domino_surface_pressure_comparison.png"
            fig.savefig(field_path, dpi=220, bbox_inches="tight")
            plt.show()
            print("Saved:", field_path)
            """
        ),
        code(
            """
            # local stencilを1点だけ可視化
            case = test_cases[0]
            anchor = int(np.argmax(np.abs(predictions[int(case["case_id"])] - case["pressure"])))
            neighbor_ids = case["neighbor_idx"][anchor]
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(projection="3d")
            preview = np.arange(0, len(case["points"]), max(1, len(case["points"]) // 2500))
            ax.scatter(*case["points"][preview].T, s=1, c="#BBBBBB", alpha=0.25)
            ax.scatter(*case["points"][neighbor_ids].T, s=55, c="#F28E2B", label="neighbors")
            ax.scatter(*case["points"][anchor], s=90, c="#76B900", label="query point")
            for nid in neighbor_ids:
                line = np.stack([case["points"][anchor], case["points"][nid]])
                ax.plot(*line.T, c="#333333", lw=1)
            ax.set_box_aspect(np.ptp(case["points"], axis=0))
            ax.view_init(elev=22, azim=-120)
            ax.set_axis_off()
            ax.legend()
            ax.set_title("Example local surface stencil")
            stencil_path = OUTPUT_DIR / "domino_local_stencil.png"
            fig.savefig(stencil_path, dpi=180, bbox_inches="tight")
            plt.show()
            print("Saved:", stencil_path)
            """
        ),
        code(
            """
            result_path = OUTPUT_DIR / "domino_surface_predictions.npz"
            result_arrays = {{
                "pressure_mean": np.asarray(PRESSURE_MEAN),
                "pressure_std": np.asarray(PRESSURE_STD),
                "data_source_mode": np.asarray(data_source_mode),
            }}
            for case in test_cases:
                cid = int(case["case_id"])
                result_arrays[f"case_{{cid}}_points"] = case["points"]
                result_arrays[f"case_{{cid}}_reference"] = case["pressure"]
                result_arrays[f"case_{{cid}}_prediction"] = predictions[cid]
            np.savez_compressed(result_path, **result_arrays)

            atomic_json(manifest_path, {{
                "status": "complete",
                "step": step,
                "elapsed_min": total_elapsed / 60,
                "data_source_mode": data_source_mode,
                "test_case_ids": [int(c["case_id"]) for c in test_cases],
                "metrics_csv": str(metrics_path),
                "predictions_npz": str(result_path),
                "updated_unix": time.time(),
            }})
            print("Complete. Outputs:")
            for path in sorted(OUTPUT_DIR.iterdir()):
                print(" -", path.name)
            """
        ),
        markdown(
            """
            ## 結果を読むときの境界

            - PhysicsNeMoの実 `DoMINO` クラスで学習・推論している
            - ただしsurface-only、少数形状、低解像度gridの教材用設定である
            - fallback実行は通信確認用であり、汎化性能の主張には使用しない
            - 2時間実行でもフルスケールDoMINoやNVIDIA公開結果の再現ではない
            - 誤差が大きい場合も、局所ステンシル、形状符号化、データ量の影響を考察する材料として残す

            発表に使う主図は `domino_surface_pressure_comparison.png`、補助図は `domino_training_curve.png` と `domino_local_stencil.png` です。
            """
        ),
        markdown(
            """
            ## 出典

            - NVIDIA PhysicsNeMo, DoMINo external aerodynamics recipe: https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/external_aerodynamics/domino/README.html
            - Ranade et al., “DoMINO: A Decomposable Multi-scale Iterative Neural Operator for Modeling Large Scale Engineering Simulations”: https://arxiv.org/abs/2501.13350
            - Ashton et al., DrivAerML dataset: https://huggingface.co/datasets/neashton/drivaerml
            - Colab用surface派生データ: https://huggingface.co/datasets/EmmiAI/DrivAerML_subsampled_10x
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
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
    fallback_b64 = build_fallback_payload()
    notebook = build_notebook(fallback_b64)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
