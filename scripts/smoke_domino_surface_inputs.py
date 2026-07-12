"""Colab用DoMINo入力を実データとCPU縮小設定でforward/backward検査する。"""

from __future__ import annotations

import ast
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from physicsnemo.models.domino.config import Config, DEFAULT_MODEL_PARAMS
from physicsnemo.models.domino.model import DoMINO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "domino_surface_only_colab.ipynb"


def load_case(case_id: int = 105) -> dict[str, np.ndarray]:
    notebook = json.loads(NOTEBOOK.read_text())
    fallback_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "FALLBACK_B64 =" in "".join(cell["source"])
    )
    assignment = next(
        node
        for node in ast.parse(fallback_cell).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "FALLBACK_B64" for t in node.targets)
    )
    payload = base64.b64decode(ast.literal_eval(assignment.value))
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {
            key: np.asarray(archive[f"case_{case_id}_{key}"], dtype=np.float32)
            for key in ("points", "normals", "areas", "pressure")
        }


def make_case() -> dict[str, np.ndarray]:
    raw = load_case()
    points = raw["points"][:512]
    normals = raw["normals"][:512]
    areas = raw["areas"][:512]
    pressure = raw["pressure"][:512]
    center = 0.5 * (points.min(0) + points.max(0))
    scale = 0.55 * float(np.max(points.max(0) - points.min(0)))
    points = ((points - center) / scale).astype(np.float32)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    areas = np.maximum(areas / scale**2, 1e-10).astype(np.float32)
    tree = cKDTree(points)
    _, neighbor_idx = tree.query(points, k=4)
    axis = np.linspace(-1, 1, 8, dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    _, nearest = tree.query(grid.reshape(-1, 3), k=1)
    delta = grid.reshape(-1, 3) - points[nearest]
    sdf = np.sum(delta * normals[nearest], axis=1).reshape(8, 8, 8).astype(np.float32)
    return {
        "points": points,
        "normals": normals,
        "areas": areas,
        "pressure": pressure,
        "neighbor_idx": neighbor_idx[:, 1:].astype(np.int32),
        "geometry": points[:256],
        "grid": grid,
        "sdf": sdf,
    }


def model_config() -> Config:
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


def main() -> None:
    case = make_case()
    query = np.arange(16)
    neighbors = case["neighbor_idx"][query]

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)

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
        "global_params_values": torch.ones((1, 1, 1)),
        "global_params_reference": torch.ones((1, 1, 1)),
    }
    target = tensor(case["pressure"][query])
    model = DoMINO(
        input_features=3,
        output_features_vol=None,
        output_features_surf=1,
        global_features=1,
        model_parameters=model_config(),
    )
    _, prediction = model(inputs)
    loss = torch.mean((prediction.squeeze(-1) - target) ** 2)
    loss.backward()
    print(
        "OK:",
        "prediction",
        tuple(prediction.shape),
        "loss",
        float(loss.detach()),
        "params",
        sum(p.numel() for p in model.parameters()),
    )


if __name__ == "__main__":
    main()
