"""Polarity and alignment auditing.

A common source of garbage output is masks that look fine but are:
  - inverted (a plane stored positive — copper = white, but the file has
    copper = black, or vice versa), or
  - offset (drill mask exported with a different canvas origin or DPI
    than the copper layers).

The functions here detect both, propose corrections, and surface the
information in a form the CLI can present to the user before doing any
heavy work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
import copy

import numpy as np
from scipy.signal import fftconvolve



# --- polarity ---

@dataclass
class PolarityVerdict:
    layer: str
    fill: float                     # fraction of pixels set (0..1)
    is_outer: bool
    action: str                     # 'none' | 'invert' | 'ambiguous' | 'warn'
    reason: str = ''


# Outer layers are essentially always exported positive (copper = white).
# A high fill on an outer is suspicious; we don't auto-invert it, we warn.
# Inner planes, by contrast, regularly come through as positive copper
# pours, where "everything except the antipads is set" produces ~85%+ fill.
INVERT_FILL_THRESHOLD = 0.6
AMBIGUOUS_LOW = 0.4


def detect_polarity(arr: np.ndarray, layer: str, outer: set[str]) -> PolarityVerdict:
    """Decide whether ``arr`` (a boolean copper mask) should be inverted."""
    fill = float(arr.mean())
    is_outer = layer in outer

    if is_outer:
        if fill > INVERT_FILL_THRESHOLD:
            return PolarityVerdict(
                layer=layer, fill=fill, is_outer=True, action='warn',
                reason=(f'outer layer with {fill:.0%} fill — likely wrong '
                        f'file mapped, or layer is actually an inner plane'),
            )
        return PolarityVerdict(layer=layer, fill=fill, is_outer=True, action='none')

    # Inner layer
    if fill > INVERT_FILL_THRESHOLD:
        return PolarityVerdict(
            layer=layer, fill=fill, is_outer=False, action='invert',
            reason=f'{fill:.0%} fill suggests a plane stored positive',
        )
    if fill < AMBIGUOUS_LOW:
        return PolarityVerdict(layer=layer, fill=fill, is_outer=False, action='none')
    return PolarityVerdict(
        layer=layer, fill=fill, is_outer=False, action='ambiguous',
        reason=f'{fill:.0%} fill is between signal-layer and plane ranges',
    )


# --- alignment ---

@dataclass
class AlignmentVerdict:
    total_drills: int
    drills_on_copper: int
    score: float                    # drills_on_copper / total_drills
    mask_fill: float = 0.0          # fraction of image pixels set (0..1)
    detected_offset: tuple[int, int] | None = None    # (dy, dx) or None
    action: str = 'none'            # 'none' | 'shift' | 'fail'
    reason: str = ''


def score_alignment(drill: np.ndarray,
                    layers: Mapping[str, np.ndarray]) -> tuple[int, int, float]:
    """How many drill components land on at least one copper pixel."""
    from scipy.ndimage import label
    lbl_drill, n = label(drill)
    if n == 0:
        return 0, 0, 1.0

    any_copper = np.zeros_like(drill, dtype=bool)
    for k,arr in layers.items():
        if k.startswith('In'):
            continue #Skip planes!
        any_copper |= arr

    touches = 0
    for d in range(1, n + 1):
        if ((lbl_drill == d) & any_copper).any():
            touches += 1
    return n, touches, touches / n

def detect_offset(
    drill: np.ndarray,
    copper_union: np.ndarray,
    max_shift: int = 500,
) -> tuple[int, int]:
    """
    Find the (dy, dx) shift to apply to `drill` so it best overlaps
    `copper_union`.

    Positive dy means move drill down.
    Positive dx means move drill right.
    """

    # Treat masks as binary. If you want weighted matching, remove the != 0.
    a = (copper_union != 0).astype(np.float32)
    b = (drill != 0).astype(np.float32)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Both inputs must be 2D masks")

    if not np.any(a) or not np.any(b):
        return 0, 0

    # Full cross-correlation:
    # corr[y, x] corresponds to shift:
    #   dy = y - (b.shape[0] - 1)
    #   dx = x - (b.shape[1] - 1)
    corr = fftconvolve(a, b[::-1, ::-1], mode="full")

    zero_y = b.shape[0] - 1
    zero_x = b.shape[1] - 1

    y0 = max(0, zero_y - max_shift)
    y1 = min(corr.shape[0], zero_y + max_shift + 1)

    x0 = max(0, zero_x - max_shift)
    x1 = min(corr.shape[1], zero_x + max_shift + 1)

    region = corr[y0:y1, x0:x1]

    py, px = np.unravel_index(np.argmax(region), region.shape)

    best_y = y0 + py
    best_x = x0 + px

    dy = best_y - zero_y
    dx = best_x - zero_x

    return int(dy), int(dx)


def audit_alignment(drill: np.ndarray,
                    layers: Mapping[str, np.ndarray],
                    auto_align: bool = True) -> AlignmentVerdict:
    """Audit whether the electrical connector mask is plausibly aligned.

    This is deliberately a *coarse* sanity check.  A fallback ``drill.png``
    often contains both PTH and NPTH/mechanical holes; requiring 90%+ of all
    drill blobs to touch copper then produces false alignment failures even
    when the file is perfectly aligned.  Net extraction later uses the
    annular-contact test to decide which holes are actually electrical.

    Therefore:
      * >= 90% touch score is excellent.
      * >= 60% is accepted as plausible mixed PTH/NPTH content.
      * lower scores are suspicious and may trigger auto-alignment/warnings.
    """
    mask_fill = float(drill.mean())
    n, touches, score = score_alignment(drill, layers)
    if n == 0:
        return AlignmentVerdict(0, 0, 1.0, mask_fill=mask_fill, action='none',
                                reason='no drills present')

    excellent_score = 0.90
    plausible_mixed_score = 0.60

    if score >= excellent_score:
        return AlignmentVerdict(
            n, touches, score, mask_fill=mask_fill, action='none',
            reason=(f'{touches}/{n} drill blobs touch outer copper '
                    f'({score:.0%}); alignment looks good'),
        )

    if not auto_align:
        if score >= plausible_mixed_score:
            return AlignmentVerdict(
                n, touches, score, mask_fill=mask_fill, action='none',
                reason=(f'{touches}/{n} drill blobs touch outer copper '
                        f'({score:.0%}); plausible mixed PTH/NPTH drill file, '
                        'kept unshifted'),
            )
        return AlignmentVerdict(
            n, touches, score, mask_fill=mask_fill, detected_offset=None, action='fail',
            reason=(f'only {touches}/{n} drill blobs touch outer copper '
                    f'({score:.0%} touch score); auto-align disabled'),
        )

    # Score is not excellent. Try to recover an offset, but keep in mind that
    # mixed PTH+NPTH drill masks may never reach 90% because the non-plated
    # holes should not touch copper pads.
    copper_union = np.zeros_like(drill, dtype=bool)
    for k, arr in layers.items():
        if k.startswith('In'):
            continue  # skip internal planes for alignment scoring
        copper_union |= arr

    offset = detect_offset(drill, copper_union)
    shifted = _shift_bool(drill, *offset)
    _, touches2, score2 = score_alignment(shifted, layers)

    if offset != (0, 0) and score2 >= excellent_score and score2 > score + 0.10:
        return AlignmentVerdict(
            n, touches, score, mask_fill=mask_fill, detected_offset=offset, action='shift',
            reason=(f'offset {offset} raises touch score from {score:.0%} '
                    f'to {score2:.0%}'),
        )

    if score >= plausible_mixed_score:
        return AlignmentVerdict(
            n, touches, score, mask_fill=mask_fill, detected_offset=offset, action='none',
            reason=(f'{touches}/{n} drill blobs touch outer copper '
                    f'({score:.0%}); best offset {offset} reaches {score2:.0%}; '
                    'kept unshifted as plausible mixed PTH/NPTH content'),
        )

    return AlignmentVerdict(
        n, touches, score, mask_fill=mask_fill, detected_offset=offset, action='fail',
        reason=(f'best offset {offset} only reaches {score2:.0%} touch score; '
                f'current score is {score:.0%}. Masks may have different extents '
                'or the wrong drill/via file may be selected'),
    )

def _shift_bool(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a boolean array by (dy, dx), filling with False at the edges."""
    out = np.zeros_like(arr)
    h, w = arr.shape
    sy_src = slice(max(0, -dy), min(h, h - dy))
    sx_src = slice(max(0, -dx), min(w, w - dx))
    sy_dst = slice(max(0, dy), max(0, dy) + (sy_src.stop - sy_src.start))
    sx_dst = slice(max(0, dx), max(0, dx) + (sx_src.stop - sx_src.start))
    out[sy_dst, sx_dst] = arr[sy_src, sx_src]
    return out


# --- post-merge sanity ---

@dataclass
class PostMergeCheck:
    n_nets: int
    largest_net_share: float        # share of copper occupied by the single biggest net
    warnings: list[str] = field(default_factory=list)


def check_merged_nets(net_labels: Mapping[str, np.ndarray]) -> PostMergeCheck:
    """Sanity-check the output of ``merge_nets``.

    These checks catch the obvious failure modes that earlier auditing
    might let through — for instance, an inner-plane polarity mistake
    that survives auto-detection because the user disabled it.
    """
    warnings: list[str] = []

    n_nets = 0
    total_copper = 0
    biggest = 0
    for arr in net_labels.values():
        if arr.max() > 0:
            n_nets = max(n_nets, int(arr.max()))
            counts = np.bincount(arr.ravel())
            if len(counts) > 1:
                copper = int(counts[1:].sum())
                total_copper += copper
                biggest = max(biggest, int(counts[1:].max()))

    share = biggest / total_copper if total_copper else 0.0

    if n_nets == 0:
        warnings.append('no nets found — check that copper masks are not empty')
    elif n_nets == 1:
        warnings.append('only one net found — drill alignment or polarity '
                        'is probably wrong')
    if share > 0.7:
        warnings.append(
            f'one net occupies {share:.0%} of all copper — likely an '
            'inverted plane swallowed the board'
        )

    return PostMergeCheck(n_nets=n_nets, largest_net_share=share, warnings=warnings)


# --- audit overlay image ---

def make_audit_overlay(drill: np.ndarray,
                       copper_union: np.ndarray):
    """Build a debug image: copper in grey, drills tinted red on top.

    Returns a ``PIL.Image.Image`` (RGB). Looking at this for two seconds
    tells you whether drills sit on pads or not.
    """
    from PIL import Image
    h, w = drill.shape
    if copper_union.shape != drill.shape:
        # Pad/crop to match — only used for display.
        copper_union = copper_union[:h, :w] if copper_union.shape[0] >= h else copper_union
    base = (copper_union.astype(np.uint8) * 128)
    rgb = np.stack([base, base, base], axis=-1)
    # Mark drills as red (additive on top of the grey).
    rgb[drill, 0] = 255
    rgb[drill, 1] = (rgb[drill, 1].astype(int) * 0).astype(np.uint8)
    rgb[drill, 2] = (rgb[drill, 2].astype(int) * 0).astype(np.uint8)
    return Image.fromarray(rgb, mode='RGB')
