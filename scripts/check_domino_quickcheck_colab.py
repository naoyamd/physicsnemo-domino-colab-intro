"""LLM/Colab向けDoMINO quick check Notebookを標準ライブラリだけで検査する。"""

from __future__ import annotations

import ast
import base64
import io
import json
import struct
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "domino_quickcheck_colab.ipynb"
EXPECTED_ARRAYS = {
    "case_105_points.npy": (4096, 3),
    "case_105_normals.npy": (4096, 3),
    "case_105_areas.npy": (4096,),
    "case_105_pressure.npy": (4096,),
}


def _npy_header(payload: bytes) -> dict:
    stream = io.BytesIO(payload)
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("invalid NPY magic")
    major, _minor = stream.read(2)
    if major == 1:
        header_length = struct.unpack("<H", stream.read(2))[0]
    elif major in (2, 3):
        header_length = struct.unpack("<I", stream.read(4))[0]
    else:
        raise ValueError(f"unsupported NPY version: {major}")
    return ast.literal_eval(stream.read(header_length).decode("latin1").strip())


def _embedded_payload(code_cells: list[str]) -> bytes:
    source = next(source for source in code_cells if "FALLBACK_CASE_B64 =" in source)
    assignment = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "FALLBACK_CASE_B64"
            for target in node.targets
        )
    )
    return base64.b64decode(ast.literal_eval(assignment.value), validate=True)


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise SystemExit("Notebook must use nbformat 4")

    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    all_source = "\n".join(code_cells)
    errors: list[str] = []

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        original = "".join(cell["source"])
        source = "\n".join(
            "pass  # " + line if line.lstrip().startswith("%") else line
            for line in original.splitlines()
        )
        try:
            compile(ast.parse(source, f"cell-{index}"), f"cell-{index}", "exec")
        except Exception as exc:
            errors.append(f"cell {index}: {type(exc).__name__}: {exc}")
        if cell.get("execution_count") is not None or cell.get("outputs"):
            errors.append(f"cell {index}: committed execution state must be empty")

    required_markers = [
        '"nvidia-physicsnemo==2.1.1"',
        "from physicsnemo.models.domino.model import DoMINO",
        "loss_before.backward()",
        "optimizer.step()",
        '"quickcheck_result.json"',
        '"DOMINO_QUICKCHECK: PASS"',
        '"status": "pass"',
        "allow_nan=False",
    ]
    for marker in required_markers:
        if marker not in all_source:
            errors.append(f"required marker missing: {marker}")

    forbidden_markers = [
        "drive.mount(",
        "hf_hub_download(",
        "HfApi(",
        "HF_TOKEN",
        "input(",
        "getpass(",
    ]
    for marker in forbidden_markers:
        if marker in all_source:
            errors.append(f"quick check must be non-interactive: {marker}")

    first_markdown = "".join(notebook["cells"][0]["source"])
    expected_url = (
        "https://colab.research.google.com/github/naoyamd/"
        "physicsnemo-domino-colab-intro/blob/main/notebooks/"
        "domino_quickcheck_colab.ipynb"
    )
    if expected_url not in first_markdown:
        errors.append("quick-check Colab URL missing from first cell")

    try:
        payload = _embedded_payload(code_cells)
        if len(payload) < 100_000:
            errors.append(f"embedded payload unexpectedly small: {len(payload)} bytes")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if set(archive.namelist()) != set(EXPECTED_ARRAYS):
                errors.append(f"unexpected payload members: {archive.namelist()}")
            for name, shape in EXPECTED_ARRAYS.items():
                header = _npy_header(archive.read(name))
                if tuple(header.get("shape", ())) != shape:
                    errors.append(f"unexpected shape for {name}: {header.get('shape')}")
                if header.get("descr") not in ("<f4", "=f4"):
                    errors.append(f"unexpected dtype for {name}: {header.get('descr')}")
                if header.get("fortran_order") is not False:
                    errors.append(f"unexpected Fortran order for {name}")
    except Exception as exc:
        errors.append(f"embedded payload invalid: {type(exc).__name__}: {exc}")

    if errors:
        raise SystemExit("Quick-check notebook errors:\n- " + "\n- ".join(errors))

    print(
        f"OK: {NOTEBOOK.name}, {len(notebook['cells'])} cells, "
        f"embedded run 105 real data, non-interactive, LLM-readable PASS contract, "
        f"{NOTEBOOK.stat().st_size / 1e6:.2f} MB"
    )


if __name__ == "__main__":
    main()
