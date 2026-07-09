"""Loading PCB layer masks from PNG files on disk."""

from __future__ import annotations

import pathlib
from typing import Callable, Iterable

from PIL import Image

Image.MAX_IMAGE_PIXELS = 1_000_000_000  # 1 gigapixel

# Optional silkscreen layer filenames, looked up alongside the copper layers.
# Canonical KiCad-style names only.
SILK_LAYERS = ('F_Silkscreen', 'B_Silkscreen')

# Map any recognised silk filename → canonical side. The viewer uses these
# positions to decide which silk to show in front view vs back view.
SILK_POSITION = {
    'F_Silkscreen': 'front',
    'B_Silkscreen': 'back',
}

# Optional solder-mask artwork. In KiCad these Gerbers normally describe
# mask *openings* rather than a complete green board shape, but they are
# still useful in the viewer for showing exposed pads/windows.
MASK_LAYERS = ('F_Mask', 'B_Mask')
MASK_POSITION = {
    'F_Mask': 'front',
    'B_Mask': 'back',
}


def alpha_to_mask(img: Image.Image) -> Image.Image:
    """Convert an arbitrary image to a clean 1-bit mask.

    Avoids ``convert('1')``'s dithering — anything above ``threshold`` becomes
    white, everything else black. The result is mode ``'1'`` and safe to feed
    to ``ImageChops.logical_*`` and ``scipy.ndimage.label`` (after going via
    numpy).
    """
    img = img.getchannel('A').point(lambda p: p > 128, mode='1')
    return img


def load_masks(
    directory: pathlib.Path,
    layer_names: Iterable[str],
    drill_name: str = 'drill',
    silk: bool = True,
    extra_names: Iterable[str] = (),
    progress: Callable[[str], None] | None = None,
) -> dict[str, Image.Image]:
    """Load each named layer + drill from ``directory`` as a mode-'1' mask.

    Filenames are ``{name}.png`` inside ``directory``. All masks must have
    identical dimensions or a ``ValueError`` is raised — alignment mismatch
    is the single most common cause of bad net extraction.

    If ``silk`` is True (default), optional silkscreen and solder-mask PNGs
    are loaded when present. They are not part of the connectivity analysis
    but are useful in the viewer.
    """
    directory = pathlib.Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a directory")

    # Preserve order but avoid loading the same mask twice.  ``extra_names``
    # is useful when the build has both a physical drill-hole mask and a
    # separate electrical via/PTH connectivity mask.
    names = []
    for name in [*layer_names, drill_name, *extra_names]:
        if name and name not in names:
            names.append(name)

    masks: dict[str, Image.Image] = {}
    for i, name in enumerate(names, start=1):
        if progress:
            progress(f'loading required mask {i}/{len(names)}: {name}.png')
        path = directory / f'{name}.png'
        if not path.is_file():
            raise FileNotFoundError(f'expected mask file not found: {path}')
        masks[name] = alpha_to_mask(Image.open(path))

    if silk:
        optional = (*SILK_LAYERS, *MASK_LAYERS)
        for i, name in enumerate(optional, start=1):
            if progress:
                progress(f'checking optional visual mask {i}/{len(optional)}: {name}.png')
            path = directory / f'{name}.png'
            if path.is_file():
                if progress:
                    progress(f'loading optional visual mask: {name}.png')
                masks[name] = alpha_to_mask(Image.open(path))

    # Sanity-check: every mask must have the same dimensions.
    if progress:
        progress(f'checking dimensions across {len(masks)} loaded mask(s)')
    sizes = {n: m.size for n, m in masks.items()}
    unique_sizes = set(sizes.values())
    if len(unique_sizes) > 1:
        details = '\n  '.join(f'{n}: {s}' for n, s in sizes.items())
        raise ValueError(
            f'mask dimensions differ — re-render with a common bounding box.\n'
            f'  {details}'
        )

    return masks
