"""Mip-map asset generation for pcbnets viewer builds."""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Iterable

import numpy as np
from PIL import Image


def _decode_ids_rgb(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert('RGB')).astype(np.uint32)
    return (arr[..., 0]
            | (arr[..., 1] << 8)
            | (arr[..., 2] << 16)).astype(np.int32)


def _encode_ids_rgb(labels: np.ndarray) -> Image.Image:
    lbl = labels.astype(np.uint32)
    rgb = np.stack(
        [lbl & 0xFF, (lbl >> 8) & 0xFF, (lbl >> 16) & 0xFF],
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode='RGB')


def _downsample_idmap(img: Image.Image,
                      width: int,
                      height: int,
                      progress: Callable[[str], None] | None = None,
                      ) -> Image.Image:
    """Downsample an encoded idmap, preferring trace labels over blank space."""
    if progress:
        progress('  idmap mip: decoding RGB ids')
    labels = _decode_ids_rgb(img)
    src_h, src_w = labels.shape
    block_h = src_h // height
    block_w = src_w // width
    if progress:
        progress(f'  idmap mip: max-pooling {src_w}x{src_h} → {width}x{height}')
    cropped = labels[:height * block_h, :width * block_w]
    out = cropped.reshape(height, block_h, width, block_w).max(axis=(1, 3))
    if progress:
        progress('  idmap mip: encoding RGB ids')
    return _encode_ids_rgb(out)


def make_mips(build_dir: pathlib.Path,
              levels: Iterable[int] = (2, 4, 8, 16),
              progress: Callable[[str], None] | None = None) -> list[pathlib.Path]:
    """Generate downsampled mip-map PNGs for every root PNG in ``build_dir``."""
    written: list[pathlib.Path] = []
    pngs = sorted(p for p in build_dir.glob('*.png') if p.is_file())
    for level in levels:
        out_dir = build_dir / 'mips' / str(level)
        out_dir.mkdir(parents=True, exist_ok=True)
        for png in pngs:
            if progress:
                progress(f'generating mip level {level}: {png.name}')
            with Image.open(png) as im:
                width = max(1, im.width // level)
                height = max(1, im.height // level)
                if png.name == 'idmap.png':
                    resized = _downsample_idmap(
                        im,
                        width,
                        height,
                        progress=progress,
                    )
                else:
                    if progress:
                        progress(f'  image mip: resizing to {width}x{height}')
                    resized = im.resize((width, height), Image.Resampling.LANCZOS)
                out_path = out_dir / png.name
                if progress:
                    progress(f'  saving mip {out_path.relative_to(build_dir)}')
                # PNG optimisation is disproportionately expensive for large
                # encoded idmaps and can look like a hang.  Save those mips
                # directly; visual PNGs still use optimisation for size.
                resized.save(out_path, optimize=(png.name != 'idmap.png'))
                if progress:
                    progress(f'  wrote mip {out_path.relative_to(build_dir)}')
                written.append(out_path)
    return written
