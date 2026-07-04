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
                      chunk_rows: int = 256,
                      ) -> Image.Image:
    """Downsample an encoded idmap, preferring trace labels over blank space."""
    src_w, src_h = img.size
    block_h = src_h // height
    block_w = src_w // width
    if block_h <= 0 or block_w <= 0:
        return img.resize((width, height), Image.Resampling.NEAREST)

    if progress:
        progress(
            f'  idmap mip: block max-pooling {src_w}x{src_h} '
            f'→ {width}x{height} in row chunks'
        )

    rgb_out = np.empty((height, width, 3), dtype=np.uint8)
    rows_per_chunk = max(1, min(chunk_rows, height))
    for y0 in range(0, height, rows_per_chunk):
        y1 = min(height, y0 + rows_per_chunk)
        src_y0 = y0 * block_h
        src_y1 = y1 * block_h
        if progress:
            progress(f'  idmap mip: rows {y0 + 1}-{y1}/{height}')

        arr = np.asarray(
            img.crop((0, src_y0, width * block_w, src_y1)).convert('RGB')
        )
        labels = (
            arr[..., 0].astype(np.uint32)
            | (arr[..., 1].astype(np.uint32) << 8)
            | (arr[..., 2].astype(np.uint32) << 16)
        )
        pooled = labels.reshape(y1 - y0, block_h, width, block_w).max(axis=(1, 3))
        rgb_out[y0:y1, :, 0] = pooled & 0xFF
        rgb_out[y0:y1, :, 1] = (pooled >> 8) & 0xFF
        rgb_out[y0:y1, :, 2] = (pooled >> 16) & 0xFF
    return Image.fromarray(rgb_out, mode='RGB')


def make_mips(build_dir: pathlib.Path,
              levels: Iterable[int] = (2, 4, 8, 16),
              progress: Callable[[str], None] | None = None) -> list[pathlib.Path]:
    """Generate downsampled mip-map PNGs for every root PNG in ``build_dir``."""
    written: list[pathlib.Path] = []
    pngs = sorted(p for p in build_dir.glob('*.png') if p.is_file())
    idmap_source = build_dir / 'idmap.png'
    idmap_source_level = 1
    for level in levels:
        out_dir = build_dir / 'mips' / str(level)
        out_dir.mkdir(parents=True, exist_ok=True)
        for png in pngs:
            if progress:
                progress(f'generating mip level {level}: {png.name}')
            source_png = png
            factor = level
            if png.name == 'idmap.png' and level % idmap_source_level == 0:
                source_png = idmap_source
                factor = level // idmap_source_level
                if progress and source_png != png:
                    progress(
                        f'  idmap mip: using {source_png.relative_to(build_dir)} '
                        f'as source (additional /{factor})'
                    )
            with Image.open(source_png) as im:
                width = max(1, im.width // factor)
                height = max(1, im.height // factor)
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
                if png.name == 'idmap.png':
                    idmap_source = out_path
                    idmap_source_level = level
    return written
