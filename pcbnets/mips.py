"""Mip-map asset generation for pcbnets viewer builds."""

from __future__ import annotations

import pathlib
from collections.abc import Iterable

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


def _downsample_idmap(img: Image.Image, width: int, height: int) -> Image.Image:
    """Downsample an encoded idmap, preferring trace labels over blank space."""
    labels = _decode_ids_rgb(img)
    src_h, src_w = labels.shape
    block_h = src_h // height
    block_w = src_w // width
    cropped = labels[:height * block_h, :width * block_w]
    out = cropped.reshape(height, block_h, width, block_w).max(axis=(1, 3))
    return _encode_ids_rgb(out)


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
                if png.name == 'idmap.png':
                    resized = _downsample_idmap(im, width, height)
                else:
                    resized = im.resize((width, height), Image.Resampling.LANCZOS)
                out_path = out_dir / png.name
                resized.save(out_path, optimize=True)
                written.append(out_path)
    return written
