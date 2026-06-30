"""Tile asset generation for pcbnets viewer builds."""

from __future__ import annotations

import pathlib

from PIL import Image


def make_tiles_for_dir(src_dir: pathlib.Path,
                       out_dir: pathlib.Path,
                       grid: int) -> list[pathlib.Path]:
    """Split every PNG in ``src_dir`` into a fixed ``grid`` x ``grid`` set."""
    written: list[pathlib.Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for png in sorted(p for p in src_dir.glob('*.png') if p.is_file()):
        base = png.stem
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


def make_tiles(build_dir: pathlib.Path) -> list[pathlib.Path]:
    """Generate viewer tiles matching the historical ``make_tiles`` script."""
    written = make_tiles_for_dir(build_dir, build_dir / 'mips' / '1' / 'tiles', 4)
    written.extend(
        make_tiles_for_dir(
            build_dir / 'mips' / '2',
            build_dir / 'mips' / '2' / 'tiles',
            2,
        )
    )
    return written
