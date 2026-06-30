"""Mip-map asset generation for pcbnets viewer builds."""

from __future__ import annotations

import pathlib
from collections.abc import Iterable

from PIL import Image


def make_mips(build_dir: pathlib.Path,
              levels: Iterable[int] = (2, 4, 8, 16)) -> list[pathlib.Path]:
    """Generate downsampled mip-map PNGs for every root PNG in ``build_dir``."""
    written: list[pathlib.Path] = []
    pngs = sorted(p for p in build_dir.glob('*.png') if p.is_file())
    for level in levels:
        out_dir = build_dir / 'mips' / str(level)
        out_dir.mkdir(parents=True, exist_ok=True)
        for png in pngs:
            with Image.open(png) as im:
                width = max(1, im.width // level)
                height = max(1, im.height // level)
                resample = Image.Resampling.NEAREST if png.name == 'idmap.png' \
                    else Image.Resampling.LANCZOS
                resized = im.resize((width, height), resample=resample)
                out_path = out_dir / png.name
                resized.save(out_path, optimize=True)
                written.append(out_path)
    return written
