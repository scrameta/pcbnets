"""Apply polarity inversion and offset shifts to raw masks.

Sits between ``load_masks`` (which produces raw boolean arrays from disk)
and ``extract_nets`` (which assumes its inputs are already correct).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from PIL import Image

from .audit import (
    PolarityVerdict,
    AlignmentVerdict,
    audit_alignment,
    detect_polarity,
)


DEFAULT_OUTER_LAYERS = {'F_Cu', 'B_Cu'}


def default_outer_for(layer_names: list[str]) -> set[str]:
    """Sensible outer-layer set when none has been specified explicitly.

    First and last names in the layer list are "outer" by convention
    (copper layers are conventionally listed top-to-bottom). The static
    ``DEFAULT_OUTER_LAYERS`` set is included as a safety net for the
    common naming schemes.
    """
    if not layer_names:
        return set(DEFAULT_OUTER_LAYERS)
    outer = {layer_names[0], layer_names[-1]}
    outer |= DEFAULT_OUTER_LAYERS & set(layer_names)
    return outer


@dataclass
class LayerCorrection:
    """The actual corrections applied to one layer, after merging auto with overrides."""
    layer: str
    invert: bool
    offset: tuple[int, int]
    source: str    # 'auto', 'override', 'none' — how the decision was reached
    notes: str = ''


@dataclass
class PreparationReport:
    """What ``prepare_masks`` decided. Useful for printing and for the cache key."""
    polarity: dict[str, PolarityVerdict]
    alignment: AlignmentVerdict | None
    corrections: dict[str, LayerCorrection]
    warnings: list[str]

    def cache_signature(self) -> dict:
        """Stable, JSON-serialisable summary for invalidating the cache."""
        return {
            'corrections': {
                name: {'invert': c.invert, 'offset': list(c.offset)}
                for name, c in self.corrections.items()
            },
        }


def _to_bool(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert('L')) > 0


def _arrays_from_masks(masks: Mapping[str, Image.Image]) -> dict[str, np.ndarray]:
    return {name: _to_bool(img) for name, img in masks.items()}


def _shift_bool(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a boolean array by (dy, dx); fill False at edges."""
    if dy == 0 and dx == 0:
        return arr
    out = np.zeros_like(arr)
    h, w = arr.shape
    sy_src = slice(max(0, -dy), min(h, h - dy))
    sx_src = slice(max(0, -dx), min(w, w - dx))
    sy_dst = slice(max(0, dy), max(0, dy) + (sy_src.stop - sy_src.start))
    sx_dst = slice(max(0, dx), max(0, dx) + (sx_src.stop - sx_src.start))
    out[sy_dst, sx_dst] = arr[sy_src, sx_src]
    return out


def prepare_masks(
    masks: Mapping[str, Image.Image],
    layer_names: list[str],
    drill_name: str,
    outer_layers: set[str] | None = None,
    auto_invert: bool = True,
    auto_align: bool = True,
    invert_overrides: set[str] | None = None,
    no_invert_overrides: set[str] | None = None,
    offset_overrides: dict[str, tuple[int, int]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, PreparationReport]:
    """Apply detected and user-requested polarity/offset corrections.

    Returns ``(layer_arrays, drill_array, report)`` where the arrays are
    boolean numpy arrays ready to feed to ``extract_nets``.
    """
    outer_layers = (outer_layers if outer_layers is not None
                    else default_outer_for(list(layer_names)))
    invert_overrides = invert_overrides or set()
    no_invert_overrides = no_invert_overrides or set()
    offset_overrides = offset_overrides or {}

    if progress:
        progress(f'converting {len(layer_names)} copper mask(s) to arrays')
    arrs = _arrays_from_masks({n: masks[n] for n in layer_names})
    if progress:
        progress(f'converting {drill_name} mask to array')
    drill_arr = _to_bool(masks[drill_name])

    polarity: dict[str, PolarityVerdict] = {}
    corrections: dict[str, LayerCorrection] = {}
    warnings: list[str] = []

    for i, name in enumerate(layer_names, start=1):
        if progress:
            progress(f'auditing polarity {i}/{len(layer_names)}: {name}')
        verdict = detect_polarity(arrs[name], name, outer_layers)
        polarity[name] = verdict

        # Resolve the invert decision (explicit overrides beat auto-detection).
        if name in invert_overrides and name in no_invert_overrides:
            warnings.append(
                f'{name}: both --invert and --no-invert specified, '
                f'using --no-invert'
            )
            do_invert = False
            source = 'override'
        elif name in invert_overrides:
            do_invert = True
            source = 'override'
        elif name in no_invert_overrides:
            do_invert = False
            source = 'override'
        elif auto_invert and verdict.action == 'invert':
            do_invert = True
            source = 'auto'
        else:
            do_invert = False
            source = 'auto' if auto_invert else 'none'

        if do_invert:
            arrs[name] = ~arrs[name]

        if verdict.action == 'warn':
            warnings.append(f'{name}: {verdict.reason}')
        if verdict.action == 'ambiguous':
            warnings.append(
                f'{name}: {verdict.reason} (no action; use --invert {name} '
                f'or --no-invert {name} to set explicitly)'
            )

        corrections[name] = LayerCorrection(
            layer=name, invert=do_invert, offset=(0, 0),
            source=source,
            notes=verdict.reason,
        )

    # Apply drill offset (overrides win over auto).
    alignment: AlignmentVerdict | None = None
    if drill_name in offset_overrides:
        if progress:
            progress(f'applying manual {drill_name} offset')
        dy, dx = offset_overrides[drill_name]
        drill_arr = _shift_bool(drill_arr, dy, dx)
        corrections[drill_name] = LayerCorrection(
            layer=drill_name, invert=False, offset=(dy, dx),
            source='override',
            notes=f'manual offset ({dy}, {dx})',
        )
    else:
        if progress:
            progress(f'auditing {drill_name} alignment against copper')
        alignment = audit_alignment(
            drill_arr,
            arrs,
            auto_align=auto_align,
            progress=progress,
        )
        if alignment.action == 'shift' and alignment.detected_offset:
            if progress:
                progress(f'applying detected {drill_name} offset {alignment.detected_offset}')
            dy, dx = alignment.detected_offset
            drill_arr = _shift_bool(drill_arr, dy, dx)
            corrections[drill_name] = LayerCorrection(
                layer=drill_name, invert=False, offset=(dy, dx),
                source='auto', notes=alignment.reason,
            )
        elif alignment.action == 'fail':
            warnings.append(
                f'drill alignment failed: {alignment.reason}. '
                f'Try --offset {drill_name} DY,DX or re-export masks with '
                f'a common bounding box.'
            )
            corrections[drill_name] = LayerCorrection(
                layer=drill_name, invert=False, offset=(0, 0),
                source='none', notes=alignment.reason,
            )
        else:
            corrections[drill_name] = LayerCorrection(
                layer=drill_name, invert=False, offset=(0, 0),
                source='auto' if auto_align else 'none',
            )

    return arrs, drill_arr, PreparationReport(
        polarity=polarity,
        alignment=alignment,
        corrections=corrections,
        warnings=warnings,
    )
