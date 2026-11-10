"""
paths.py — find the repo root from anywhere (script, notebook, pytest).

Same idea as R/_utils.R: walk up the folder tree until we find
environment.yml, which only exists at the root.
"""
from pathlib import Path


def project_root(marker: str = "environment.yml") -> Path:
    here = Path.cwd().resolve()
    for folder in [here, *here.parents]:
        if (folder / marker).exists():
            return folder
    # Fall back to this file's location (python/climate_discount/paths.py)
    for folder in Path(__file__).resolve().parents:
        if (folder / marker).exists():
            return folder
    raise FileNotFoundError(f"Could not find '{marker}' above {here}")


ROOT = project_root()
DIR_PROCESSED = ROOT / "data" / "processed"
DIR_TABLES = ROOT / "output" / "tables"
DIR_FIGURES = ROOT / "output" / "figures"
for _d in (DIR_TABLES, DIR_FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
