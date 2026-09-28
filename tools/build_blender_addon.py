"""Build the installable Blender add-on zip.

    python tools/build_blender_addon.py          -> dist/pointille_blender-<version>.zip

The zip holds one folder, ``pointille_blender/``, with the algorithm package
vendored inside, so it installs both as a Blender 4.2+ extension
("Install from Disk") and as a legacy add-on (Blender 3.6-4.1).
"""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADDON = ROOT / "blender_addon" / "pointille_blender"
PACKAGE = ROOT / "pointille"
VENDORED = ("__init__.py", "core", "io")


def _ignore(_dir, names):
    return [n for n in names if n == "__pycache__" or n.endswith(".pyc")]


def stage(out_dir: Path) -> Path:
    """Copy the add-on with the vendored package into ``out_dir/pointille_blender``."""
    target = out_dir / "pointille_blender"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(ADDON, target, ignore=_ignore)
    vendor = target / "pointille"
    if vendor.exists():
        shutil.rmtree(vendor)
    vendor.mkdir()
    for name in VENDORED:
        src = PACKAGE / name
        if src.is_dir():
            shutil.copytree(src, vendor / name, ignore=_ignore)
        else:
            shutil.copy2(src, vendor / name)
    return target


def version() -> str:
    text = (ADDON / "blender_manifest.toml").read_text(encoding="utf-8")
    return re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)


def build(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    staged = stage(out_dir / "stage")
    zip_path = out_dir / f"pointille_blender-{version()}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(staged.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(staged.parent).as_posix())
    shutil.rmtree(staged.parent)
    return zip_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--out", type=Path, default=ROOT / "dist")
    print(build(parser.parse_args().out))


if __name__ == "__main__":
    main()
