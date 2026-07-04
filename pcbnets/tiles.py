"""Tile asset generation for pcbnets viewer builds."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

from PIL import Image


def make_tiles_for_dir(src_dir: pathlib.Path,
                       out_dir: pathlib.Path,
                       grid: int,
                       progress: Callable[[str], None] | None = None) -> list[pathlib.Path]:
    """Split every PNG in ``src_dir`` into a fixed ``grid`` x ``grid`` set."""
    written: list[pathlib.Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for png in sorted(p for p in src_dir.glob('*.png') if p.is_file()):
        base = png.stem
        if progress:
            progress(f'tiling level grid {grid}x{grid}: {png.name}')
        with Image.open(png) as im:
            for ty in range(grid):
                y0 = ty * im.height // grid
                y1 = (ty + 1) * im.height // grid
                for tx in range(grid):
                    x0 = tx * im.width // grid
                    x1 = (tx + 1) * im.width // grid
                    tile = im.crop((x0, y0, x1, y1))
                    out_path = out_dir / f'{base}_{tx}_{ty}.png'
                    tile.save(out_path, optimize=True)
                    written.append(out_path)
    return written


def make_tiles(build_dir: pathlib.Path,
               progress: Callable[[str], None] | None = None) -> list[pathlib.Path]:
    """Generate viewer tiles for the high-resolution mip levels."""
    tile_grids = {
        1: 16,
        2: 8,
        4: 4,
        8: 2,
    }
    written: list[pathlib.Path] = []
    for level, grid in tile_grids.items():
        src_dir = build_dir if level == 1 else build_dir / 'mips' / str(level)
        written.extend(
            make_tiles_for_dir(
                src_dir,
                build_dir / 'mips' / str(level) / 'tiles',
                grid,
                progress=progress,
            )
        )
    return written
