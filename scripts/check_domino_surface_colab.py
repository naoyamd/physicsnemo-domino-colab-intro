"""domino_surface_only_colab.ipynbの構文とfallback実データを検査する。"""

from __future__ import annotations

import ast
import base64
import io
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "domino_surface_only_colab.ipynb"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    errors: list[tuple[int, str, str]] = []

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        original_source = "".join(cell["source"])
        if "{{" in original_source or "}}" in original_source:
            errors.append(
                (index, "DoubleBrace", "二重波括弧がコードセルに残っています")
            )
            continue
        source = "\n".join(
            "pass  # " + line if line.lstrip().startswith("%") else line
            for line in original_source.splitlines()
        )
        try:
            tree = ast.parse(source, f"cell-{index}", "exec")
            compile(tree, f"cell-{index}", "exec")
        except Exception as exc:  # pragma: no cover - diagnostic tool
            errors.append((index, type(exc).__name__, str(exc)))
            continue

        # Colabの標準Matplotlibフォントでは日本語が文字化けしやすい。
        # PNGを保存するセル内の文字列は英語（ASCII）に限定する。
        if ".savefig(" in original_source:
            non_ascii = sorted(
                {
                    node.value
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and not node.value.isascii()
                }
            )
            if non_ascii:
                errors.append(
                    (index, "NonAsciiPlotText", repr(non_ascii))
                )

    if errors:
        raise SystemExit(f"Notebook syntax/plot text errors: {errors}")

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
    fallback_b64 = ast.literal_eval(assignment.value)
    with np.load(
        io.BytesIO(base64.b64decode(fallback_b64)), allow_pickle=False
    ) as archive:
        case_ids = archive["case_ids"].astype(int).tolist()
        if case_ids != [105, 130, 202]:
            raise SystemExit(f"Unexpected fallback case IDs: {case_ids}")
        for case_id in case_ids:
            points = archive[f"case_{case_id}_points"]
            normals = archive[f"case_{case_id}_normals"]
            areas = archive[f"case_{case_id}_areas"]
            pressure = archive[f"case_{case_id}_pressure"]
            if not (
                points.shape == normals.shape == (4096, 3)
                and areas.shape == pressure.shape == (4096,)
            ):
                raise SystemExit(f"Unexpected fallback shapes for case {case_id}")
            if not all(np.isfinite(x).all() for x in (points, normals, areas, pressure)):
                raise SystemExit(f"Non-finite fallback values for case {case_id}")
            if not np.all(areas > 0):
                raise SystemExit(f"Non-positive areas for case {case_id}")

    print(
        f"OK: {NOTEBOOK.name}, {len(notebook['cells'])} cells, "
        f"fallback cases {case_ids}, plot text ASCII, "
        f"{NOTEBOOK.stat().st_size / 1e6:.2f} MB"
    )


if __name__ == "__main__":
    main()
