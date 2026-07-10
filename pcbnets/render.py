"""Grid composition and net-id encoding for the interactive viewer."""

from __future__ import annotations

import json
from typing import Callable, Mapping

import numpy as np
from PIL import Image


def _encode_ids_rgb(labels: np.ndarray) -> Image.Image:
    """Encode an integer label array as an RGB image (R = low byte, etc).

    Supports up to 2**24 - 1 distinct ids, which is far more than any board
    will ever have. The viewer JS reverses this with ``r | (g<<8) | (b<<16)``.
    """
    lbl = labels.astype(np.uint32)
    rgb = np.stack(
        [lbl & 0xFF, (lbl >> 8) & 0xFF, (lbl >> 16) & 0xFF],
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode='RGB')


def labels_to_rgb(labels: np.ndarray, seed: int = 0) -> Image.Image:
    """Render a label array with a stable random palette.

    Useful for debugging — every distinct net gets its own colour.
    Background (label 0) stays black.
    """
    rng = np.random.default_rng(seed)
    n = int(labels.max())
    palette = rng.integers(64, 256, size=(n + 1, 3), dtype=np.uint8)
    palette[0] = 0
    return Image.fromarray(palette[labels], mode='RGB')


def _downsample_image(img: Image.Image, scale: float) -> Image.Image:
    if scale == 1.0:
        return img
    w, h = img.size
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                      Image.Resampling.LANCZOS)


def _downsample_labels(labels: np.ndarray, scale: float) -> np.ndarray:
    """Downsample labels without letting background erase traces.

    Output pixels take the maximum label from their source area. Since blank
    pixels are label 0, any trace label in the area wins over blank space while
    still only emitting labels that existed in the source image.
    """
    if scale == 1.0:
        return labels
    h, w = labels.shape
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    block_h = h // new_h
    block_w = w // new_w
    if block_h > 0 and block_w > 0:
        cropped = labels[:new_h * block_h, :new_w * block_w]
        return cropped.reshape(new_h, block_h, new_w, block_w).max(axis=(1, 3))

    # Upscaling or heavily skewed dimensions should not happen for idmaps in
    # normal builds; keep exact ids by falling back to nearest-neighbour.
    img = Image.fromarray(labels.astype(np.int32), mode='I')
    img = img.resize((new_w, new_h), Image.Resampling.NEAREST)
    return np.asarray(img, dtype=np.int32)


def build_grid_and_idmap(
    layer_images: Mapping[str, Image.Image],
    net_labels: Mapping[str, np.ndarray],
    cols: int = 2,
    scale: float = 1.0,
    label_text: bool = True,
    progress: Callable[[str], None] | None = None,
) -> tuple[Image.Image, Image.Image, dict]:
    """Build the greyscale display grid and the encoded net-id map.

    Layers are placed left-to-right, top-to-bottom in dict-iteration order
    (Python dicts preserve insertion order). The id map has the same
    geometry as the grid, so click coordinates map directly between them.

    ``scale`` shrinks both outputs by that factor (1.0 = full size). The id
    map prefers real net ids over background when reducing pixels; the
    display image uses LANCZOS for smoother edges.

    Returns ``(grid_image, idmap_image, metadata)``. The metadata dict
    captures layer placements and per-layer dimensions so the viewer can
    label tiles and report which layer was clicked.
    """
    if progress:
        progress('checking grid inputs')
    layers = list(layer_images)
    if not layers:
        raise ValueError('no layers provided')

    # All layers must agree on shape.
    sizes = {n: layer_images[n].size for n in layers}
    if len(set(sizes.values())) > 1:
        details = '\n  '.join(f'{n}: {s}' for n, s in sizes.items())
        raise ValueError(f'layer image sizes differ:\n  {details}')

    w, h = sizes[layers[0]]
    rows = (len(layers) + cols - 1) // cols

    if progress:
        progress(f'allocating full grid/idmap canvases: {w * cols}x{h * rows}')
    grid_full = Image.new('L', (w * cols, h * rows), 0)
    # 'RGB' with black background = net id 0 = background. Perfect.
    idmap_full = Image.new('RGB', (w * cols, h * rows), (0, 0, 0))

    placements: dict[str, dict] = {}
    for i, name in enumerate(layers):
        if progress:
            progress(f'placing layer {i + 1}/{len(layers)}: {name}')
        col = i % cols
        row = i // cols
        x, y = col * w, row * h
        grid_full.paste(layer_images[name].convert('L'), (x, y))
        idmap_full.paste(_encode_ids_rgb(net_labels[name]), (x, y))
        placements[name] = {
            'x': x, 'y': y, 'w': w, 'h': h,
            'col': col, 'row': row,
        }

    if progress:
        progress(f'downsampling display grid at scale {scale:g}')
    grid_out = _downsample_image(grid_full, scale)
    # For the idmap, downsample the *labels* via nearest then re-encode, so
    # we never invent intermediate ids.
    if scale != 1.0:
        if progress:
            progress('decoding idmap labels for trace-preserving downsample')
        idmap_full_labels = _decode_ids_rgb(idmap_full)
        if progress:
            progress(f'downsampling idmap labels at scale {scale:g}')
        idmap_full_labels = _downsample_labels(idmap_full_labels, scale)
        if progress:
            progress('encoding downsampled idmap labels')
        idmap_out = _encode_ids_rgb(idmap_full_labels)
    else:
        if progress:
            progress('keeping full-size idmap without downsample')
        idmap_out = idmap_full

    # Scale placements to match the downsampled output.
    if scale != 1.0:
        if progress:
            progress('scaling layer placement metadata')
        scaled_placements = {}
        for name, p in placements.items():
            scaled_placements[name] = {
                'x': int(p['x'] * scale),
                'y': int(p['y'] * scale),
                'w': int(p['w'] * scale),
                'h': int(p['h'] * scale),
                'col': p['col'],
                'row': p['row'],
            }
        placements = scaled_placements

    if progress:
        progress('assembling grid metadata')
    meta = {
        'layers': layers,
        'cols': cols,
        'rows': rows,
        'tile_w': placements[layers[0]]['w'],
        'tile_h': placements[layers[0]]['h'],
        'grid_w': grid_out.size[0],
        'grid_h': grid_out.size[1],
        'placements': placements,
        'show_labels': bool(label_text),
    }

    return grid_out, idmap_out, meta


def _decode_ids_rgb(img: Image.Image) -> np.ndarray:
    """Inverse of ``_encode_ids_rgb`` — RGB image back to int32 labels."""
    arr = np.asarray(img.convert('RGB')).astype(np.uint32)
    return (arr[..., 0]
            | (arr[..., 1] << 8)
            | (arr[..., 2] << 16)).astype(np.int32)


def write_meta(meta: dict, path) -> None:
    """Write metadata as JSON. Convenience for CLI users."""
    with open(path, 'w') as fp:
        json.dump(meta, fp, indent=2)


def labels_to_netmap_svg(net_labels: Mapping[str, np.ndarray], placements: Mapping[str, Mapping[str, int]], width: int, height: int) -> str:
    """Serialize per-layer label arrays to an inline-pickable SVG net map."""
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {int(width)} {int(height)}">',
        '<style>.net-shape{fill:transparent;stroke:transparent;pointer-events:all}.net-shape.selected{fill:rgba(255,80,80,0.7);stroke:rgba(255,80,80,0.7);pointer-events:all}</style>',
    ]
    for layer, labels in net_labels.items():
        if layer not in placements:
            raise ValueError(f'cannot generate netmap.svg: missing placement for {layer}')
        p = placements[layer]
        ox, oy = int(p['x']), int(p['y'])
        arr = np.asarray(labels)
        ids = [int(i) for i in np.unique(arr) if int(i) > 0]
        parts.append(f'<g data-layer="{layer}">')
        for net_id in ids:
            runs: list[str] = []
            ys, xs = np.where(arr == net_id)
            if len(xs) == 0:
                continue
            for y in sorted(set(int(v) for v in ys)):
                row = arr[y] == net_id
                x = 0
                w = row.shape[0]
                while x < w:
                    if not row[x]:
                        x += 1
                        continue
                    x0 = x
                    while x < w and row[x]:
                        x += 1
                    x1 = x
                    runs.append(f'M{ox+x0} {oy+y}H{ox+x1}V{oy+y+1}H{ox+x0}Z')
            d = ''.join(runs)
            parts.append(f'<path class="net-shape" data-net-id="{net_id}" data-layer="{layer}" fill-rule="evenodd" d="{d}"/>')
        parts.append('</g>')
    parts.append('</svg>')
    return '\n'.join(parts)
