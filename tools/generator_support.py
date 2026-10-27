"""Platform launcher for the organiser's code-generation subprocess."""

import json
import sys
import tempfile
from pathlib import Path


def emmy_executable() -> str:
    if sys.platform != "win32":
        return str(Path(sys.executable).with_name("emmy"))
    root = Path(__file__).resolve().parents[1]
    launcher = root / ".cache" / "emmy_codegen.cmd"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    content = f'@echo off\n"{sys.executable}" -X utf8 "{root / "tools" / "emmy_codegen.py"}" %*\n'
    if not launcher.exists() or launcher.read_text() != content:
        with tempfile.NamedTemporaryFile(mode="w", dir=launcher.parent, suffix=".cmd", delete=False) as handle:
            handle.write(content)
        Path(handle.name).replace(launcher)
    return str(launcher)


def validate_package(path: Path) -> None:
    """Reject packages the runner cannot represent, before claiming generation passed."""
    meta = json.loads((path / "meta.json").read_text())
    for kernel in meta["kernels"]:
        for arg in kernel["args"]:
            if "data" in arg and any(x is None for x in arg["data"]):
                raise ValueError(f"Generator emitted null constant data for {arg['name']}; runner cannot materialize it")
            if arg["dtype"] not in {"f16", "f32", "f64", "i32", "i64"}:
                raise ValueError(f"Generator emitted runner-unsupported dtype {arg['dtype']} for {arg['name']}")
    import numpy as np

    shapes = {a["name"]: a["shape"] for k in meta["kernels"] for a in k["args"] if a.get("output")}
    with np.load(path / "expected.npz") as archive:
        for name in meta["graph_outputs"]:
            if name not in archive or list(archive[name].shape or (1,)) != shapes[name]:
                raise ValueError(f"Generator reference shape does not match graph output {name}")
