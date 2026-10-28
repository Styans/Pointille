"""Command line: ``pointille model.glb -n 800`` -> the model with its drones (GLB), CSV and PLY."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import COLOR_MODES, DistributeOptions, distribute
from .io.export import to_z_up, write_csv, write_ply
from .io.gltf_append import GlbError, append_drones, empty_glb, required_extensions

# glTF and OBJ are Y-up and look along -Z: the direction from the model to the audience (as in the web app)
VIEWS = {
    "front": (0.0, 0.0, 1.0),
    "back": (0.0, 0.0, -1.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "top": (0.0, 1.0, 0.0),
}
UP_AXIS = 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pointille",
        description="Ровная расстановка дронов по поверхности 3D-модели (GLB, glTF, OBJ, PLY, STL).",
    )
    p.add_argument("model", type=Path, help="файл модели")
    p.add_argument("-n", "--count", type=int, default=500, help="сколько дронов (ровно столько, по умолчанию 500)")
    p.add_argument("-o", "--out", type=Path, help="папка для результата (по умолчанию — папка модели)")
    p.add_argument("--facade", action="store_true", help="только фасад: дроны там, где модель видят зрители")
    p.add_argument("--view", choices=list(VIEWS), default="front", help="где зрители (по умолчанию front)")
    p.add_argument("--spread", type=float, default=0.0, help="ширина зрителей вокруг модели, градусы")
    p.add_argument("--hidden", type=float, default=0.0,
                   help="плотность там, где модель не видит никто, %% от фасада")
    p.add_argument("--color", choices=COLOR_MODES, default="area",
                   help="area — средний по участку, point — точно под дроном, none — белый")
    p.add_argument("--min-brightness", type=float, default=0.0,
                   help="самый тёмный дрон горит с этой яркостью, %% (0 — цвета как у модели)")
    p.add_argument("--outline", action="store_true", help="часть дронов на линиях модели")
    scale = p.add_mutually_exclusive_group()
    scale.add_argument("--min-dist", type=float, help="масштаб: мин. дистанция между дронами, м")
    scale.add_argument("--height", type=float, help="масштаб: высота фигуры, м")
    p.add_argument("--seed", type=int, default=0, help="другой вариант такой же ровной расстановки")
    p.add_argument("--iterations", type=int, default=20, help="итерации выравнивания (0 — без них)")
    p.add_argument("--oversample", type=float, default=8.0, help="случайных точек-кандидатов на дрона")
    return p


def run(args: argparse.Namespace) -> dict:
    """Place the drones and write the files; returns the result's summary and the files written."""
    src = args.model
    if not src.is_file():
        raise SystemExit(f"pointille: нет файла {src}")
    glb = src.read_bytes() if src.suffix.lower() == ".glb" else None
    if glb is not None and "KHR_draco_mesh_compression" in required_extensions(glb):
        raise SystemExit("pointille: модель сжата Draco — распакуйте её: "
                         "npx @gltf-transform/cli decompress in.glb out.glb")
    from .io.model import load_mesh  # trimesh only here: --help works without it

    scale_mode = "min_distance" if args.min_dist else "height" if args.height else "none"
    opt = DistributeOptions(
        count=args.count,
        seed=args.seed,
        oversample=args.oversample,
        relax_iterations=args.iterations,
        facade=args.facade,
        view_dir=VIEWS[args.view],
        audience_spread=args.spread,
        hidden_density=args.hidden / 100.0,
        color_mode=args.color,
        min_brightness=args.min_brightness / 100.0,
        outline=args.outline,
        scale_mode=scale_mode,
        target_min_distance=args.min_dist or 1.5,
        target_height=args.height or 20.0,
        up_axis=UP_AXIS,
    )
    result = distribute(load_mesh(src), opt)
    summary = result.summary()

    out = args.out or src.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    # the model keeps its own data; the drones come in as one more node, scaled with it about the pivot
    glb_path = out / (f"{stem}_pointille.glb" if glb is not None else f"{stem}_drones.glb")
    glb_path.write_bytes(append_drones(glb if glb is not None else empty_glb(), result.positions,
                                       result.colors_linear, scale=result.scale, pivot=result.pivot,
                                       extras=summary))
    points = to_z_up(result.scaled_positions(), "y")
    csv_path, ply_path = out / f"{stem}_drones.csv", out / f"{stem}_drones.ply"
    write_csv(csv_path, points, result.colors_srgb8)
    write_ply(ply_path, points, result.colors_srgb8)
    summary["files"] = [str(glb_path), str(csv_path), str(ply_path)]
    return summary


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        s = run(args)
    except (ValueError, GlbError) as e:
        print(f"pointille: {e}", file=sys.stderr)
        return 1
    width, height, depth = s["size"]
    print(f"Дронов: {s['count']}")
    if s["outline_count"]:
        print(f"На обводке: {s['outline_count']}")
    print(f"Мин. дистанция: {s['min_distance']:.3f}, средняя: {s['mean_distance']:.3f}")
    print(f"Размер: {width:.2f} x {depth:.2f} x {height:.2f} (высота)")
    if s["scale"] != 1.0:
        print(f"Масштаб: {s['scale']:.4g}")
    for f in s["files"]:
        print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
