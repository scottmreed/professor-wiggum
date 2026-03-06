from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "render_rxn_map_pngs.py"
INPUT_PATH = PROJECT_ROOT / "training_data" / "rxn_map_expanded.json"


def test_render_rxn_map_pngs_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "pngs"
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--input",
        str(INPUT_PATH),
        "--output",
        str(out_dir),
        "--max-reactions",
        "2",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    # Graceful missing-RDKit/Pillow path.
    if proc.returncode == 2:
        combined = f"{proc.stdout}\n{proc.stderr}"
        assert "RDKit" in combined or "Pillow" in combined
        return

    assert proc.returncode == 0, proc.stderr
    index_path = out_dir / "index.json"
    assert index_path.exists()

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["rendered_count"] == len(payload["items"])
    assert payload["rendered_count"] <= 2

    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) == payload["rendered_count"]
