"""pcbnets command-line interface."""

from __future__ import annotations

import argparse
import copy
import filecmp
import json
import logging
import pathlib
import pickle
import re
import shutil
import sys
import tempfile
import time
import zipfile
from typing import Iterable

import numpy as np
from PIL import Image

from . import __version__
from .audit import check_merged_nets, detect_offset, make_audit_overlay
from .gerber import (
    GerbvMissingError,
    detect_layers,
    rasterise,
    write_layers_json,
)
from .masks import MASK_LAYERS, SILK_LAYERS, load_masks, threshold_mask
from .mips import make_mips
from .tiles import make_tiles
from .nets import extract_nets, merge_nets_debug, explain_merge_path
from .prepare import DEFAULT_OUTER_LAYERS, _shift_bool, prepare_masks
from .render import build_grid_and_idmap


def _format_elapsed(seconds: float) -> str:
    """Format an elapsed duration for human-readable progress messages."""
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours}h{minutes:02d}m{secs:02d}s'
    return f'{minutes}m{secs:02d}s'


class _Progress:
    """Timestamped progress reporter for long render operations."""

    def __init__(self, total: int, label: str = 'render') -> None:
        self.total = total
        self.current = 0
        self.started = time.monotonic()
        self.step_started = self.started
        self.label = label
        self.log = logging.getLogger('pcbnets.render')

    def step(self, message: str) -> None:
        now = time.monotonic()
        if self.current:
            self.log.info('  previous step took %s',
                          _format_elapsed(now - self.step_started))
        self.current += 1
        self.step_started = now
        pct = min(99, int(self.current * 100 / self.total))
        elapsed = _format_elapsed(now - self.started)
        self.log.info('  [%s/%s %2s%% %s] %s',
                      self.current, self.total, pct, elapsed, message)

    def detail(self, message: str) -> None:
        now = time.monotonic()
        total_elapsed = _format_elapsed(now - self.started)
        step_elapsed = _format_elapsed(now - self.step_started)
        self.log.info('    ... %s (%s step, %s total)',
                      message, step_elapsed, total_elapsed)

    def finish(self) -> None:
        now = time.monotonic()
        self.log.info('  previous step took %s',
                      _format_elapsed(now - self.step_started))
        elapsed = _format_elapsed(now - self.started)
        self.log.info('  [%s/%s 100%% %s] %s complete',
                      self.total, self.total, elapsed, self.label)


def _configure_cli_logging() -> None:
    """Emit CLI status logs to stdout with readable timestamps."""
    logger = logging.getLogger('pcbnets')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    logger.addHandler(handler)

# Canonical copper naming only.  The normal auto-detected form is
# F_Cu, optional InN_Cu layers, then B_Cu.
DEFAULT_LAYER_SETS = [
    ['F_Cu', 'B_Cu'],
]


# ---------- helpers ----------

def _sorted_kicad_copper_layers(available: set[str]) -> list[str]:
    """Return a full KiCad copper stack if F/B copper are present.

    The old fixed presets accidentally matched the first four layers of a
    six-layer board because ``F_Cu, In1_Cu, In2_Cu, B_Cu`` is a subset of
    ``F_Cu, In1_Cu, In2_Cu, In3_Cu, In4_Cu, B_Cu``.  Build the stack
    dynamically instead: front, all numbered inner layers, then back.
    """
    if not {'F_Cu', 'B_Cu'} <= available:
        return []
    inners: list[tuple[int, str]] = []
    for name in available:
        m = re.fullmatch(r'In(\d+)_Cu', name)
        if m:
            inners.append((int(m.group(1)), name))
    return ['F_Cu', *(name for _, name in sorted(inners)), 'B_Cu']


def _resolve_layers(directory: pathlib.Path,
                    requested: Iterable[str] | None,
                    drill_name: str = 'drill',
                    via_name: str = 'via') -> list[str]:
    """Pick the copper layer list to use.

    1. If ``requested`` was given, use it.
    2. Otherwise dynamically recognise a full KiCad-style stack, including
       6+ layer boards.
    3. Otherwise try the remaining canonical fixed presets.
    4. Failing that, return no auto-selected layers.

    Legacy aliases such as ``top``, ``inner1`` and ``bottom`` are deliberately
    not auto-recognised.  Use canonical ``F_Cu``/``InN_Cu``/``B_Cu`` names,
    or pass an explicit ``--layers`` list for unusual input.
    """
    if requested:
        return list(requested)
    available = {p.stem for p in directory.glob('*.png')}

    # Non-connectivity visual/assembly layers should not be auto-selected as copper.
    # Always exclude PTH/NPTH too: they are drill masks, not copper layers.
    excluded = {
        drill_name, via_name, 'drill', 'via', 'PTH', 'NPTH',
        *SILK_LAYERS, *MASK_LAYERS,
        'F_Paste', 'B_Paste', 'Edge_Cuts',
    }
    available -= excluded

    dynamic = _sorted_kicad_copper_layers(available)
    if dynamic:
        return dynamic

    for candidate in DEFAULT_LAYER_SETS:
        if set(candidate) <= available:
            return list(candidate)

    return []


def _first_existing_png(directory: pathlib.Path, names: Iterable[str]) -> str | None:
    for name in names:
        if name and (directory / f'{name}.png').is_file():
            return name
    return None


def _resolve_drill_name(directory: pathlib.Path, requested: str = 'auto') -> str:
    """Resolve the physical drill-hole mask.

    ``drill`` means physical holes.  It is useful for diagnostics/metadata,
    but electrical merging should normally use :func:`_resolve_via_name`.
    """
    if requested in ('auto', 'drill'):
        found = _first_existing_png(directory, ['drill', 'PTH', 'via'])
        return found or 'drill'
    if (directory / f'{requested}.png').is_file():
        return requested
    return requested


def _resolve_via_name(directory: pathlib.Path, requested: str = 'auto') -> str:
    """Resolve the electrical vertical-connector mask.

    Preference order for automatic mode:
      1. ``via.png``  -- explicit/generated electrical via mask
      2. ``PTH.png``  -- plated through-hole drill mask
      3. ``drill.png`` -- compatibility fallback when no separate via/PTH exists

    ``NPTH.png`` is deliberately never selected automatically.
    """
    if requested in ('auto', 'via'):
        found = _first_existing_png(directory, ['via', 'PTH', 'drill'])
        return found or 'via'
    if (directory / f'{requested}.png').is_file():
        return requested
    return requested


def _connector_mode(directory: pathlib.Path, via_name: str) -> str:
    """Return how the selected vertical-connector mask should be interpreted."""
    if via_name == 'NPTH':
        return 'never'
    if via_name == 'via':
        via_path = directory / 'via.png'
        drill_path = directory / 'drill.png'
        pth_path = directory / 'PTH.png'
        if (not pth_path.exists() and via_path.exists() and drill_path.exists()
                and filecmp.cmp(via_path, drill_path, shallow=False)):
            return 'infer'
        return 'explicit'
    if via_name in {'via', 'PTH'}:
        return 'explicit'
    return 'infer'


def _parse_offset(spec: str) -> tuple[int, int]:
    """Parse 'DY,DX' into (dy, dx)."""
    try:
        dy_str, dx_str = spec.split(',', 1)
        return int(dy_str), int(dx_str)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(
            f'offset must be DY,DX (integers); got {spec!r}'
        )


def _collect_overrides(args: argparse.Namespace) -> dict:
    """Build the override dicts from CLI args."""
    invert = set(args.invert or [])
    no_invert = set(args.no_invert or [])
    offsets: dict[str, tuple[int, int]] = {}
    for spec in (args.offset or []):
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            layer, off_str = spec
        else:
            raise argparse.ArgumentTypeError(f'bad --offset value: {spec!r}')
        offsets[layer] = _parse_offset(off_str)

    outer = set(args.outer) if args.outer else DEFAULT_OUTER_LAYERS
    return {
        'invert_overrides': invert,
        'no_invert_overrides': no_invert,
        'offset_overrides': offsets,
        'outer_layers': outer,
    }



def _mask_side_pairs(layer_names: list[str]) -> list[tuple[str, str]]:
    """Return optional KiCad solder-mask layer → copper reference pairs.

    The mask Gerbers are visual-only here, so this does not affect net
    extraction.  Only canonical KiCad names are recognised.
    """
    names = set(layer_names)
    pairs: list[tuple[str, str]] = []
    if 'F_Cu' in names:
        pairs.append(('F_Mask', 'F_Cu'))
    if 'B_Cu' in names:
        pairs.append(('B_Mask', 'B_Cu'))
    return pairs

def _bool_to_mask_image(arr: np.ndarray) -> Image.Image:
    """Convert a boolean array back to a clean 1-bit PIL mask."""
    return Image.fromarray(arr.astype(np.uint8) * 255, mode='L').convert('1')


def _mask_to_bool(img: Image.Image) -> np.ndarray:
    """Convert a loaded mask image to a boolean array without changing polarity."""
    return np.asarray(img.convert('L')) > 0


def _visual_drill_array(masks: dict[str, Image.Image],
                        drill_name: str,
                        via_name: str,
                        via_arr: np.ndarray,
                        offset_overrides: dict[str, tuple[int, int]]) -> np.ndarray:
    """Return the physical hole mask to punch out of the viewer artwork.

    Net extraction uses ``via_arr`` as the electrical vertical-connector mask.
    The viewer, however, should show physical holes as empty space.  Those are
    represented by ``drill_name`` and may be different from ``via_name`` when a
    board has separate PTH/via and NPTH/mechanical drill files.
    """
    if drill_name == via_name:
        # ``prepare_masks`` has already applied any manual/auto alignment to
        # the electrical drill mask, so reuse the corrected array.
        return via_arr.astype(bool, copy=False)

    drill_arr = _mask_to_bool(masks[drill_name])
    if drill_name in offset_overrides:
        dy, dx = offset_overrides[drill_name]
        drill_arr = _shift_bool(drill_arr, dy, dx)
    return drill_arr


def _punch_drill_holes_for_display(arrs: dict[str, np.ndarray],
                                   net_labels: dict[str, np.ndarray],
                                   drill_arr: np.ndarray,
                                   layer_names: list[str]
                                   ) -> tuple[dict[str, Image.Image],
                                              dict[str, np.ndarray]]:
    """Remove physical holes from the visual copper and click idmap.

    This is intentionally display-only.  Connectivity has already been
    extracted from the unpunched copper masks, where the annulus around a drill
    is needed to decide which pads/vias are connected.
    """
    display_images: dict[str, Image.Image] = {}
    display_labels: dict[str, np.ndarray] = {}

    for name in layer_names:
        visible = arrs[name] & ~drill_arr
        display_images[name] = Image.fromarray(
            visible.astype(np.uint8) * 255, mode='L'
        )

        labels = net_labels[name].copy()
        labels[drill_arr] = 0
        display_labels[name] = labels

    return display_images, display_labels


def _align_visual_masks(masks: dict[str, Image.Image],
                        arrs: dict[str, np.ndarray],
                        layer_names: list[str],
                        offset_overrides: dict[str, tuple[int, int]],
                        auto_align: bool = True,
                        max_shift: int = 500,
                        min_overlap_gain: float = 0.10) -> dict[str, dict]:
    """Apply visual-only alignment to solder-mask opening layers.

    Drill alignment can use a component-touch score.  Solder-mask openings are
    different: a misaligned opening can still touch the right copper component,
    so we use raw overlap area and require a useful improvement before shifting.

    Existing ``--offset F_Mask DY,DX`` / ``--offset B_Mask DY,DX`` overrides
    are honoured and beat auto-detection.
    """
    corrections: dict[str, dict] = {}

    for mask_name, copper_name in _mask_side_pairs(layer_names):
        if mask_name not in masks or copper_name not in arrs:
            continue

        mask_arr = np.asarray(masks[mask_name].convert('L')) > 0
        copper_arr = arrs[copper_name]

        if mask_name in offset_overrides:
            dy, dx = offset_overrides[mask_name]
            masks[mask_name] = _bool_to_mask_image(_shift_bool(mask_arr, dy, dx))
            corrections[mask_name] = {
                'offset': [dy, dx],
                'source': 'override',
                'reference': copper_name,
                'notes': f'manual visual offset against {copper_name}',
            }
            logging.getLogger('pcbnets.render').info('  %-14s visual  shift (forced)     offset (%s, %s) against %s', mask_name, dy, dx, copper_name)
            continue

        if not auto_align:
            corrections[mask_name] = {
                'offset': [0, 0],
                'source': 'none',
                'reference': copper_name,
                'notes': 'visual auto-align disabled',
            }
            continue

        overlap0 = int((mask_arr & copper_arr).sum())
        if overlap0 == 0:
            # Still try, but avoid divide-by-zero in the reporting below.
            overlap0_for_gain = 1
        else:
            overlap0_for_gain = overlap0

        dy, dx = detect_offset(mask_arr, copper_arr, max_shift=max_shift)
        shifted = _shift_bool(mask_arr, dy, dx)
        overlap1 = int((shifted & copper_arr).sum())
        gain = (overlap1 - overlap0) / overlap0_for_gain

        if (dy, dx) != (0, 0) and gain >= min_overlap_gain:
            masks[mask_name] = _bool_to_mask_image(shifted)
            corrections[mask_name] = {
                'offset': [dy, dx],
                'source': 'auto',
                'reference': copper_name,
                'notes': f'overlap with {copper_name} improved by {gain:.0%}',
            }
            logging.getLogger('pcbnets.render').info('  %-14s visual  shift (auto)       offset (%s, %s) against %s; overlap +%.0f%%', mask_name, dy, dx, copper_name, gain * 100)
        else:
            corrections[mask_name] = {
                'offset': [0, 0],
                'source': 'auto',
                'reference': copper_name,
                'notes': f'best offset ({dy}, {dx}) improved overlap by {gain:.0%}; kept unshifted',
            }
            if (dy, dx) != (0, 0):
                logging.getLogger('pcbnets.render').info('  %-14s visual  keep               best offset (%s, %s) only overlap +%.0f%%', mask_name, dy, dx, gain * 100)

    return corrections


def _load_net_labels_for_render(
    nets_dir: pathlib.Path,
    layer_names: list[str],
    arrs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Load ``pcbnets nets`` output and validate it can drive rendering."""
    labels_path = nets_dir / 'net-labels.npz'
    if not labels_path.exists():
        raise FileNotFoundError(
            f'missing {labels_path}; run `pcbnets nets ... -o {nets_dir}` first'
        )

    with np.load(labels_path) as data:
        net_labels = {
            name: data[name].astype(np.int32, copy=False)
            for name in data.files
        }

    missing = [name for name in layer_names if name not in net_labels]
    extra = [name for name in net_labels if name not in layer_names]
    if missing or extra:
        details = []
        if missing:
            details.append(f'missing layers: {", ".join(missing)}')
        if extra:
            details.append(f'unexpected layers: {", ".join(extra)}')
        raise ValueError(
            f'precomputed net labels in {nets_dir} do not match render layers '
            f'({"; ".join(details)})'
        )

    for name in layer_names:
        if net_labels[name].shape != arrs[name].shape:
            raise ValueError(
                f'precomputed net labels for {name} have shape '
                f'{net_labels[name].shape}, expected {arrs[name].shape}'
            )

    # Preserve the render layer order rather than the archive's file order.
    return {name: net_labels[name] for name in layer_names}


def _print_report(report, drill_name: str) -> None:
    """Pretty-print the preparation report to the console.

    Copper rows report pixel fill.  The via/drill row reports an alignment
    touch score, not pixel fill, so print it explicitly to avoid confusing
    a sparse drill mask with an 80-100% alignment score.
    """
    log = logging.getLogger('pcbnets.render')
    log.info('  --- audit ---')
    log.info('  %-14s %7s  %-7s %-18s Note', 'Layer', 'Fill%', 'Kind', 'Action')
    for name, p in report.polarity.items():
        c = report.corrections.get(name)
        if c and c.invert and c.source == 'override':
            action = 'invert (forced)'
        elif c and c.invert and c.source == 'auto':
            action = 'invert (auto)'
        elif c and not c.invert and c.source == 'override':
            action = 'keep (forced)'
        else:
            action = 'keep'
        kind = 'outer' if p.is_outer else 'inner'
        log.info('  %-14s %6.1f%%  %-7s %-18s %s', name, p.fill * 100, kind, action, p.reason)

    if report.alignment is not None:
        a = report.alignment
        drill_action = 'keep' if a.action == 'none' else a.action
        log.info('  %-14s %6.1f%%  via     %-18s touch %s/%s = %.1f%%', drill_name, a.mask_fill * 100, drill_action, a.drills_on_copper, a.total_drills, a.score * 100)
        if a.detected_offset is not None and a.detected_offset != (0, 0):
            log.info('  %-14s %7s          %-18s detected offset: %s', '', '', '', a.detected_offset)

    if report.warnings:
        log.info('  --- warnings ---')
        for w in report.warnings:
            log.warning('  ! %s', w)


def _run_pipeline(directory: pathlib.Path,
                  layer_names: list[str],
                  drill_name: str,
                  via_name: str,
                  drill_grow_px: int,
                  threshold: int,
                  scale: float,
                  cols: int,
                  cache: bool,
                  nets_dir: pathlib.Path | None,
                  prep_kwargs: dict,
                  auto_invert: bool,
                  auto_align: bool,
                  progress: _Progress | None = None) -> tuple:
    """Returns (grid, idmap, meta, masks, report).

    ``drill_name`` is the physical hole mask.  ``via_name`` is the electrical
    vertical-connector mask used for net merging.  They can be the same file
    for old/simple builds.
    """
    extra = [] if drill_name == via_name else [drill_name]
    if progress:
        progress.step('loading masks')
    masks = load_masks(directory, layer_names, via_name, threshold,
                       extra_names=extra,
                       progress=progress.detail if progress else None)

    if progress:
        progress.step('preparing and auditing masks')
    arrs, via_arr, report = prepare_masks(
        masks=masks,
        layer_names=layer_names,
        drill_name=via_name,
        auto_invert=auto_invert,
        auto_align=auto_align,
        progress=progress.detail if progress else None,
        **prep_kwargs,
    )
    _print_report(report, via_name)

    if progress:
        progress.step('aligning visual masks')
    visual_corrections = _align_visual_masks(
        masks=masks,
        arrs=arrs,
        layer_names=layer_names,
        offset_overrides=prep_kwargs.get('offset_overrides', {}),
        auto_align=auto_align,
    )

    cache_path = directory / '.pcbnets-cache.pkl'
    cache_key = {
        'version': __version__,
        'layers': layer_names,
        'drill': drill_name,
        'via': via_name,
        'connector_mode': _connector_mode(directory, via_name),
        'grow': drill_grow_px,
        'threshold': threshold,
        'sizes': {n: masks[n].size for n in masks
                  if n not in (*SILK_LAYERS, *MASK_LAYERS)},
        'prep': report.cache_signature(),
    }

    if progress:
        progress.step('loading precomputed/cached nets or extracting/merging nets')
    net_labels = None
    net_debug = None
    if nets_dir is not None:
        net_labels = _load_net_labels_for_render(
            nets_dir=nets_dir,
            layer_names=layer_names,
            arrs=arrs,
        )
        logging.getLogger('pcbnets.render').info(
            '  using precomputed net labels from %s', nets_dir
        )

    if net_labels is None and cache and cache_path.exists():
        try:
            with open(cache_path, 'rb') as fp:
                cached = pickle.load(fp)
            if cached.get('key') == cache_key:
                net_labels = cached['net_labels']
                net_debug = cached.get('net_debug')
                logging.getLogger('pcbnets.render').info('  using cached net labels from %s', cache_path.name)
        except Exception:
            net_labels = None

    if net_labels is None:
        logging.getLogger('pcbnets.render').info('  extracting components on %s layers...', len(layer_names))
        result = extract_nets(
            copper_layers=arrs,
            drill=via_arr,
            drill_grow_px=drill_grow_px,
            connector_mode=_connector_mode(directory, via_name),
            progress=progress.detail if progress else None,
        )
        logging.getLogger('pcbnets.render').info('  merging via %s connector components...', len(result['drill_touches']))
        net_labels, net_debug = merge_nets_debug(
            result['drill_touches'],
            result['layer_labels'],
            result['drill_labels'],
            result.get('drill_classifications'),
        )
        if cache:
            try:
                with open(cache_path, 'wb') as fp:
                    pickle.dump({'key': cache_key, 'net_labels': net_labels, 'net_debug': net_debug}, fp)
            except Exception as e:
                logging.getLogger('pcbnets.render').warning('  (warning: cache write failed: %s)', e)

    if progress:
        progress.step('validating merged nets')
    check = check_merged_nets(net_labels)
    for w in check.warnings:
        logging.getLogger('pcbnets.render').warning('  ! %s', w)

    logging.getLogger('pcbnets.render').info('  %s distinct nets across the board', check.n_nets)
    if progress:
        progress.step('building display grid + id map')
    logging.getLogger('pcbnets.render').info('  building display grid + id map...')

    # ``arrs`` and ``net_labels`` are deliberately kept unpunched for the
    # electrical extraction above.  For the viewer, physical drill holes should
    # render as holes rather than solid gold copper, and clicking inside a hole
    # should not select the surrounding net.
    visual_drill = _visual_drill_array(
        masks=masks,
        drill_name=drill_name,
        via_name=via_name,
        via_arr=via_arr,
        offset_overrides=prep_kwargs.get('offset_overrides', {}),
    )
    if progress:
        progress.detail('punching physical drill holes out of display layers')
    display_images, display_net_labels = _punch_drill_holes_for_display(
        arrs=arrs,
        net_labels=net_labels,
        drill_arr=visual_drill,
        layer_names=layer_names,
    )

    grid, idmap, meta = build_grid_and_idmap(
        layer_images=display_images,
        net_labels=display_net_labels,
        cols=cols,
        scale=scale,
        progress=progress.detail if progress else None,
    )
    meta['n_nets'] = check.n_nets
    meta['source_dir'] = str(directory)
    meta['render_scale'] = scale
    meta['drill_name'] = drill_name
    meta['via_name'] = via_name
    meta['connector_mode'] = _connector_mode(directory, via_name)
    meta['drill_grow_px'] = drill_grow_px
    meta['outer_layers'] = sorted(prep_kwargs.get('outer_layers', DEFAULT_OUTER_LAYERS))
    meta['corrections'] = {
        name: {'invert': c.invert, 'offset': list(c.offset), 'source': c.source}
        for name, c in report.corrections.items()
    }
    for name, corr in visual_corrections.items():
        meta['corrections'][name] = {
            'invert': False,
            'offset': corr['offset'],
            'source': corr['source'],
            'reference': corr.get('reference'),
            'notes': corr.get('notes', ''),
        }

    silk_present: list[str] = []
    for silk_name in SILK_LAYERS:
        if silk_name in masks:
            silk_present.append(silk_name)
    if silk_present:
        meta['silk_layers'] = silk_present

    mask_present: list[str] = []
    for mask_name in MASK_LAYERS:
        if mask_name in masks:
            mask_present.append(mask_name)
    if mask_present:
        meta['mask_layers'] = mask_present

    return grid, idmap, meta, masks, report


def _write_build(build_dir: pathlib.Path,
                 grid: Image.Image,
                 idmap: Image.Image,
                 meta: dict,
                 masks: dict[str, Image.Image]) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    grid.save(build_dir / 'grid.png', optimize=True)
    idmap.save(build_dir / 'idmap.png', optimize=True)
    with open(build_dir / 'meta.json', 'w') as fp:
        json.dump(meta, fp, indent=2)

    for visual_name in (*SILK_LAYERS, *MASK_LAYERS):
        if visual_name in masks:
            masks[visual_name].convert('L').save(build_dir / f'{visual_name}.png',
                                                 optimize=True)
            logging.getLogger('pcbnets.render').info('  wrote %s.png', visual_name)

    logging.getLogger('pcbnets.render').info('  wrote %s/grid.png, idmap.png, meta.json', build_dir)


def _write_mips_and_tiles(build_dir: pathlib.Path,
                           progress: _Progress | None = None) -> None:
    """Create all progressive-loading assets expected by the web viewer."""
    if progress:
        progress.step('generating mip PNGs')
    mip_paths = make_mips(build_dir, progress=progress.detail if progress else None)
    logging.getLogger('pcbnets.render').info('  wrote %s mip PNG(s)', len(mip_paths))

    if progress:
        progress.step('generating tile PNGs')
    tile_paths = make_tiles(build_dir, progress=progress.detail if progress else None)
    logging.getLogger('pcbnets.render').info('  wrote %s tile PNG(s)', len(tile_paths))


# ---------- subcommands ----------

def cmd_audit(args: argparse.Namespace) -> int:
    directory = pathlib.Path(args.directory).resolve()
    drill_name = _resolve_drill_name(directory, args.drill)
    via_name = _resolve_via_name(directory, args.via)
    layers = _resolve_layers(directory, args.layers, drill_name, via_name)
    overrides = _collect_overrides(args)

    print(f'auditing {directory}')
    print(f'  layers: {", ".join(layers)}')
    print(f'  drill : {drill_name}')
    print(f'  via   : {via_name}')
    print(f'  outer : {", ".join(sorted(overrides["outer_layers"]))}')
    extra = [] if drill_name == via_name else [drill_name]
    masks = load_masks(directory, layers, via_name, args.threshold, extra_names=extra)
    arrs, via_arr, report = prepare_masks(
        masks=masks,
        layer_names=layers,
        drill_name=via_name,
        auto_invert=not args.no_auto_invert,
        auto_align=args.auto_align,
        **overrides,
    )
    _print_report(report, via_name)

    if args.output:
        out_path = pathlib.Path(args.output)
        copper_union = np.zeros_like(via_arr, dtype=bool)
        for arr in arrs.values():
            copper_union |= arr
        overlay = make_audit_overlay(via_arr, copper_union)
        overlay.save(out_path)
        print(f'  wrote audit overlay: {out_path}')

    return 0 if not report.warnings else 1


def cmd_render(args: argparse.Namespace) -> int:
    _configure_cli_logging()
    directory = pathlib.Path(args.directory).resolve()
    build_dir = pathlib.Path(args.output).resolve()
    drill_name = _resolve_drill_name(directory, args.drill)
    via_name = _resolve_via_name(directory, args.via)
    layers = _resolve_layers(directory, args.layers, drill_name, via_name)
    overrides = _collect_overrides(args)

    logging.getLogger('pcbnets.render').info('rendering nets from %s', directory)
    logging.getLogger('pcbnets.render').info('  layers: %s', ', '.join(layers))
    logging.getLogger('pcbnets.render').info('  drill : %s', drill_name)
    logging.getLogger('pcbnets.render').info('  via   : %s', via_name)
    logging.getLogger('pcbnets.render').info('  mode  : %s', _connector_mode(directory, via_name))

    progress = _Progress(total=9)
    grid, idmap, meta, masks, _ = _run_pipeline(
        directory, layers, drill_name, via_name,
        args.drill_grow, args.threshold, args.scale, args.cols,
        cache=not args.no_cache,
        nets_dir=pathlib.Path(args.nets).resolve() if args.nets else None,
        prep_kwargs=overrides,
        auto_invert=not args.no_auto_invert,
        auto_align=args.auto_align,
        progress=progress,
    )
    progress.step('writing build artifacts')
    _write_build(build_dir, grid, idmap, meta, masks)
    _write_mips_and_tiles(build_dir, progress=progress)
    progress.finish()
    return 0


def _load_prepared_net_inputs(args: argparse.Namespace) -> tuple[pathlib.Path, list[str], str, str, dict, dict[str, np.ndarray], np.ndarray, object]:
    """Load and prepare masks for net-only/debug commands."""
    directory = pathlib.Path(args.directory).resolve()
    drill_name = _resolve_drill_name(directory, args.drill)
    via_name = _resolve_via_name(directory, args.via)
    layers = _resolve_layers(directory, args.layers, drill_name, via_name)
    overrides = _collect_overrides(args)
    extra = [] if drill_name == via_name else [drill_name]
    masks = load_masks(directory, layers, via_name, args.threshold, extra_names=extra)
    arrs, via_arr, report = prepare_masks(
        masks=masks,
        layer_names=layers,
        drill_name=via_name,
        auto_invert=not args.no_auto_invert,
        auto_align=args.auto_align,
        **overrides,
    )
    return directory, layers, drill_name, via_name, overrides, arrs, via_arr, report


def _write_net_debug(out_dir: pathlib.Path,
                     source_dir: pathlib.Path,
                     layers: list[str],
                     drill_name: str,
                     via_name: str,
                     connector_mode: str,
                     threshold: int,
                     layer_labels: dict[str, np.ndarray],
                     net_labels: dict[str, np.ndarray],
                     debug: dict) -> None:
    """Write the reusable outputs of the net-finding stage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / 'layer-labels.npz', **layer_labels)
    np.savez_compressed(out_dir / 'net-labels.npz', **net_labels)
    manifest = {
        'source_dir': str(source_dir),
        'layers': layers,
        'drill_name': drill_name,
        'via_name': via_name,
        'connector_mode': connector_mode,
        'threshold': threshold,
        'shapes': {
            name: list(net_labels[name].shape)
            for name in layers
        },
        **debug,
    }
    with open(out_dir / 'nets.json', 'w') as fp:
        json.dump(manifest, fp, indent=2)
        fp.write('\n')


def cmd_nets(args: argparse.Namespace) -> int:
    """Run only the net-finding stage and write debug artefacts."""
    _configure_cli_logging()
    out_dir = pathlib.Path(args.output).resolve()
    directory, layers, drill_name, via_name, _, arrs, via_arr, report = _load_prepared_net_inputs(args)
    logging.getLogger('pcbnets.render').info('finding nets from %s', directory)
    logging.getLogger('pcbnets.render').info('  layers: %s', ', '.join(layers))
    logging.getLogger('pcbnets.render').info('  drill : %s', drill_name)
    logging.getLogger('pcbnets.render').info('  via   : %s', via_name)
    logging.getLogger('pcbnets.render').info('  mode  : %s', _connector_mode(directory, via_name))
    _print_report(report, via_name)

    progress = _Progress(total=3, label='net extraction')
    progress.step('extracting per-layer components and drill contacts')
    result = extract_nets(
        arrs,
        via_arr,
        drill_grow_px=args.drill_grow,
        connector_mode=_connector_mode(directory, via_name),
        progress=progress.detail,
    )
    progress.step('merging nets and building debug metadata')
    net_labels, debug = merge_nets_debug(
        result['drill_touches'],
        result['layer_labels'],
        result['drill_labels'],
        result.get('drill_classifications'),
    )
    progress.step('writing net debug artefacts')
    _write_net_debug(
        out_dir=out_dir,
        source_dir=directory,
        layers=layers,
        drill_name=drill_name,
        via_name=via_name,
        connector_mode=_connector_mode(directory, via_name),
        threshold=args.threshold,
        layer_labels=result['layer_labels'],
        net_labels=net_labels,
        debug=debug,
    )
    print(f'wrote net debug data to {out_dir}')
    print('  layer-labels.npz  local per-layer connected components')
    print('  net-labels.npz    board-wide merged net labels')
    print('  nets.json         components, drill merge edges, merged groups')
    progress.finish()
    return 0


def _decision_map_from_choices(path: pathlib.Path) -> dict[int, bool]:
    data = json.loads(path.read_text())
    decisions = data.get('drill_classifications', data.get('decisions', []))
    return {
        int(d['drill']): bool(d['plated'])
        for d in decisions
        if 'drill' in d and 'plated' in d
    }


def _split_drill_masks(drill_labels: np.ndarray,
                       classifications: list[dict],
                       overrides: dict[int, bool] | None = None
                       ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    overrides = overrides or {}
    plated_ids: set[int] = set()
    decisions: list[dict] = []
    for decision in classifications:
        drill_id = int(decision['drill'])
        out = dict(decision)
        if drill_id in overrides:
            out['plated'] = bool(overrides[drill_id])
            out['override'] = True
            out['reason'] = (
                f'user override from choices JSON: '
                f'{"plated/PTH" if out["plated"] else "non-plated/NPTH"}'
            )
        else:
            out['override'] = False
        if out.get('plated'):
            plated_ids.add(drill_id)
        decisions.append(out)

    pth = np.isin(drill_labels, list(plated_ids)) if plated_ids else np.zeros_like(drill_labels, dtype=bool)
    npth = (drill_labels != 0) & ~pth
    return pth, npth, decisions


def _copy_pngs_for_drill_identify(src_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.glob('*.png')):
        shutil.copy2(src, out_dir / src.name)


def _write_drill_identify_outputs(out_dir: pathlib.Path,
                                  source_dir: pathlib.Path,
                                  drill_name: str,
                                  layer_names: list[str],
                                  drill_labels: np.ndarray,
                                  classifications: list[dict],
                                  choices_path: pathlib.Path | None = None) -> None:
    overrides = _decision_map_from_choices(choices_path) if choices_path else {}
    pth, npth, decisions = _split_drill_masks(
        drill_labels,
        classifications,
        overrides=overrides,
    )

    _bool_to_mask_image(pth).save(out_dir / 'PTH.png', optimize=True)
    _bool_to_mask_image(pth).save(out_dir / 'via.png', optimize=True)
    _bool_to_mask_image(npth).save(out_dir / 'NPTH.png', optimize=True)
    _bool_to_mask_image(drill_labels != 0).save(out_dir / 'drill.png', optimize=True)

    manifest = {
        'source_dir': str(source_dir),
        'drill_name': drill_name,
        'layers': layer_names,
        'outputs': {
            'PTH': 'PTH.png',
            'via': 'via.png',
            'NPTH': 'NPTH.png',
            'drill': 'drill.png',
        },
        'plated_count': int(sum(1 for d in decisions if d.get('plated'))),
        'npth_count': int(sum(1 for d in decisions if not d.get('plated'))),
        'choices_source': str(choices_path) if choices_path else None,
        'drill_classifications': decisions,
    }
    with open(out_dir / 'drill-identify.json', 'w') as fp:
        json.dump(manifest, fp, indent=2)
        fp.write('\n')


class GerbonaraMissingError(RuntimeError):
    pass


def _load_gerbonara_excellon(path: pathlib.Path):
    try:
        try:
            from gerbonara import ExcellonFile
        except ImportError:
            from gerbonara.excellon import ExcellonFile
    except ImportError as e:
        raise GerbonaraMissingError(
            'Excellon drill_identify mode requires gerbonara. '
            'Install it with `pip install gerbonara`.'
        ) from e

    if hasattr(ExcellonFile, 'open'):
        return ExcellonFile.open(path)
    if hasattr(ExcellonFile, 'from_file'):
        return ExcellonFile.from_file(path)
    raise RuntimeError('unsupported gerbonara ExcellonFile API')


def _save_gerbonara_file(obj, path: pathlib.Path) -> None:
    if hasattr(obj, 'save'):
        obj.save(path)
        return
    if hasattr(obj, 'write'):
        obj.write(path)
        return
    if hasattr(obj, 'to_file'):
        obj.to_file(path)
        return
    raise RuntimeError('unsupported gerbonara file save API')


def _copy_excellon_with_objects(excellon, objects: list):
    out = copy.copy(excellon)
    out.objects = list(objects)
    return out


def _split_excellon_objects(excellon_path: pathlib.Path,
                            out_dir: pathlib.Path,
                            object_decisions: dict[int, bool]) -> tuple[pathlib.Path, pathlib.Path]:
    excellon = _load_gerbonara_excellon(excellon_path)
    objects = list(getattr(excellon, 'objects'))
    pth_objects = [
        obj for idx, obj in enumerate(objects)
        if object_decisions.get(idx, False)
    ]
    npth_objects = [
        obj for idx, obj in enumerate(objects)
        if idx in object_decisions and not object_decisions[idx]
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    pth_path = out_dir / 'PTH.drl'
    npth_path = out_dir / 'NPTH.drl'
    _save_gerbonara_file(_copy_excellon_with_objects(excellon, pth_objects), pth_path)
    _save_gerbonara_file(_copy_excellon_with_objects(excellon, npth_objects), npth_path)
    return pth_path, npth_path


def _split_excellon_with_gerbonara(excellon_path: pathlib.Path,
                                   out_dir: pathlib.Path,
                                   choices_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    data = json.loads(choices_path.read_text())
    decisions = data.get('drill_classifications', data.get('decisions', []))
    object_decisions = {
        int(d.get('source_object_index', d.get('object_index'))): bool(d['plated'])
        for d in decisions
        if (d.get('source_object_index', d.get('object_index')) is not None
            and 'plated' in d)
    }
    if not object_decisions:
        raise ValueError(
            'choices JSON does not contain source_object_index/object_index fields; '
            'cannot map decisions back to Excellon objects'
        )
    return _split_excellon_objects(excellon_path, out_dir, object_decisions)


def _object_xy(obj) -> tuple[float, float] | None:
    x = getattr(obj, 'x', None)
    y = getattr(obj, 'y', None)
    if x is not None and y is not None:
        try:
            return float(x), float(y)
        except (TypeError, ValueError):
            pass
    bbox = None
    for name in ('bounding_box', 'bounding_box_no_aperture'):
        candidate = getattr(obj, name, None)
        if callable(candidate):
            try:
                bbox = candidate()
                break
            except TypeError:
                continue
    if bbox and len(bbox) == 2:
        (min_x, min_y), (max_x, max_y) = bbox
        try:
            return (float(min_x) + float(max_x)) / 2, (float(min_y) + float(max_y)) / 2
        except (TypeError, ValueError):
            return None
    return None


def _infer_excellon_object_decisions(
    excellon_path: pathlib.Path,
    classifications: list[dict],
) -> tuple[dict[int, bool], dict[int, int], dict[int, tuple[float, float]]]:
    excellon = _load_gerbonara_excellon(excellon_path)
    objects = list(getattr(excellon, 'objects'))
    if len(objects) != len(classifications):
        raise RuntimeError(
            f'cannot map {len(classifications)} raster drill blobs to '
            f'{len(objects)} Excellon objects'
        )

    sortable = []
    for idx, obj in enumerate(objects):
        xy = _object_xy(obj)
        if xy is None:
            sortable.append((1, idx, idx))
        else:
            x, y = xy
            # gerbv renders +Y upward into image coordinates, while scipy labels
            # raster blobs top-to-bottom.  Sort Excellon objects high-Y first,
            # then low-X first, to match raster connected-component order.
            sortable.append((0, -y, x, idx))
    sorted_object_indexes = [item[-1] for item in sorted(sortable)]

    sorted_classifications = sorted(classifications, key=lambda d: int(d['drill']))
    object_decisions: dict[int, bool] = {}
    drill_to_object: dict[int, int] = {}
    drill_to_xy: dict[int, tuple[float, float]] = {}
    for decision, object_index in zip(sorted_classifications, sorted_object_indexes):
        drill_id = int(decision['drill'])
        object_decisions[int(object_index)] = bool(decision['plated'])
        drill_to_object[drill_id] = int(object_index)
        xy = _object_xy(objects[object_index])
        if xy is not None:
            drill_to_xy[drill_id] = xy
    return object_decisions, drill_to_object, drill_to_xy


def cmd_drill_identify(args: argparse.Namespace) -> int:
    """Classify a generic drill mask into PTH/via and NPTH PNG masks."""
    _configure_cli_logging()
    directory = pathlib.Path(args.directory).resolve()
    out_dir = pathlib.Path(args.output).resolve()
    if not directory.is_dir():
        print(f'{directory} is not a directory', file=sys.stderr)
        return 1

    temp_pngs = None
    png_dir = directory
    if not any(directory.glob('*.png')):
        if not args.excellon:
            print(
                'drill_identify needs either a PNG mask directory or '
                '`--excellon` with a Gerber/Excellon directory to rasterise. '
                'For PNG mode, run `pcbnets gerber ... -o pngs` first, then '
                '`pcbnets drill_identify pngs -o identified`.',
                file=sys.stderr,
            )
            return 2

        temp_pngs = tempfile.TemporaryDirectory(prefix='pcbnets-drill-identify-')
        png_dir = pathlib.Path(temp_pngs.name)
        mapping = detect_layers(directory)
        excellon_path = pathlib.Path(args.excellon).resolve()
        mapping['drill'] = str(
            excellon_path if excellon_path.is_absolute()
            else pathlib.Path(args.excellon)
        )
        layer_targets = list(args.layers) if args.layers else [
            name for name in mapping
            if re.match(r'^(F_Cu|In\d+_Cu|B_Cu)$', name)
        ]
        if 'drill' not in layer_targets:
            layer_targets.append('drill')
        try:
            rasterise(directory, mapping, png_dir, dpi=args.dpi, layers=layer_targets)
        except (GerbvMissingError, RuntimeError) as e:
            if temp_pngs is not None:
                temp_pngs.cleanup()
            print(str(e), file=sys.stderr)
            return 2

    drill_name = _resolve_drill_name(png_dir, args.drill)
    layers = _resolve_layers(png_dir, args.layers, drill_name, drill_name)
    overrides = _collect_overrides(args)

    logging.getLogger('pcbnets.render').info('identifying drill holes from %s', directory)
    logging.getLogger('pcbnets.render').info('  layers: %s', ', '.join(layers))
    logging.getLogger('pcbnets.render').info('  drill : %s', drill_name)

    masks = load_masks(png_dir, layers, drill_name, args.threshold)
    arrs, drill_arr, report = prepare_masks(
        masks=masks,
        layer_names=layers,
        drill_name=drill_name,
        auto_invert=not args.no_auto_invert,
        auto_align=args.auto_align,
        **overrides,
    )
    _print_report(report, drill_name)

    progress = _Progress(total=3, label='drill identification')
    progress.step('classifying drill components')
    result = extract_nets(
        arrs,
        drill_arr,
        connector_mode='infer',
        progress=progress.detail,
    )
    classifications = list(result['drill_classifications'])
    excellon_object_decisions = None
    if args.excellon and not args.choices:
        try:
            excellon_object_decisions, drill_to_object, drill_to_xy = _infer_excellon_object_decisions(
                pathlib.Path(args.excellon).resolve(),
                classifications,
            )
        except (GerbonaraMissingError, RuntimeError) as e:
            if temp_pngs is not None:
                temp_pngs.cleanup()
            print(str(e), file=sys.stderr)
            return 2
        classifications = [
            {
                **decision,
                'source_object_index': drill_to_object.get(int(decision['drill'])),
                'source_x': (
                    drill_to_xy[int(decision['drill'])][0]
                    if int(decision['drill']) in drill_to_xy else None
                ),
                'source_y': (
                    drill_to_xy[int(decision['drill'])][1]
                    if int(decision['drill']) in drill_to_xy else None
                ),
            }
            for decision in classifications
        ]

    progress.step('copying PNG inputs')
    _copy_pngs_for_drill_identify(png_dir, out_dir)
    progress.step('writing PTH/NPTH/via masks and choices JSON')
    _write_drill_identify_outputs(
        out_dir=out_dir,
        source_dir=directory,
        drill_name=drill_name,
        layer_names=layers,
        drill_labels=result['drill_labels'],
        classifications=classifications,
        choices_path=pathlib.Path(args.choices).resolve() if args.choices else None,
    )
    if args.excellon and args.choices:
        try:
            pth_path, npth_path = _split_excellon_with_gerbonara(
                pathlib.Path(args.excellon).resolve(),
                out_dir,
                pathlib.Path(args.choices).resolve(),
            )
            print(f'wrote {pth_path}')
            print(f'wrote {npth_path}')
        except (GerbonaraMissingError, RuntimeError, ValueError) as e:
            if temp_pngs is not None:
                temp_pngs.cleanup()
            print(str(e), file=sys.stderr)
            return 2
    elif args.excellon:
        try:
            pth_path, npth_path = _split_excellon_objects(
                pathlib.Path(args.excellon).resolve(),
                out_dir,
                excellon_object_decisions or {},
            )
            print(f'wrote {pth_path}')
            print(f'wrote {npth_path}')
        except (GerbonaraMissingError, RuntimeError) as e:
            if temp_pngs is not None:
                temp_pngs.cleanup()
            print(str(e), file=sys.stderr)
            return 2

    if temp_pngs is not None:
        temp_pngs.cleanup()

    progress.finish()
    print(f'wrote drill identification outputs to {out_dir}')
    print('  PTH.png, via.png, NPTH.png, drill.png')
    print('  drill-identify.json')
    return 0


def _parse_component_spec(spec: str, labels: dict[str, np.ndarray]) -> tuple[str, int]:
    """Parse LAYER:COMPONENT or LAYER:X,Y into a local component node."""
    try:
        layer, rest = spec.split(':', 1)
    except ValueError:
        raise argparse.ArgumentTypeError(f'expected LAYER:COMPONENT or LAYER:X,Y, got {spec!r}')
    if layer not in labels:
        raise argparse.ArgumentTypeError(f'unknown layer {layer!r}; available: {", ".join(labels)}')
    if ',' in rest:
        x_s, y_s = rest.split(',', 1)
        x, y = int(x_s), int(y_s)
        arr = labels[layer]
        if y < 0 or x < 0 or y >= arr.shape[0] or x >= arr.shape[1]:
            raise argparse.ArgumentTypeError(f'point {spec!r} is outside layer bounds')
        component = int(arr[y, x])
        if component == 0:
            raise argparse.ArgumentTypeError(f'point {spec!r} is background, not copper')
        return layer, component
    component = int(rest)
    if component <= 0:
        raise argparse.ArgumentTypeError('component ids must be positive')
    return layer, component


def _debug_component_index(debug: dict) -> dict[tuple[str, int], dict]:
    return {
        (c['layer'], int(c['component'])): c
        for c in debug.get('components', [])
    }


def _debug_drill_index(debug: dict) -> dict[int, dict]:
    return {
        int(d['drill']): d
        for d in debug.get('drills', [])
    }


def _format_bbox(bbox: list | tuple | None) -> str:
    if not bbox or len(bbox) < 4:
        return 'bbox unavailable'
    x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
    return f'bbox {x0},{y0}..{x1 - 1},{y1 - 1}'


def _format_component_location(node: dict, components: dict[tuple[str, int], dict]) -> str:
    comp = components.get((node['layer'], int(node['component'])))
    if not comp:
        return f'{node["layer"]}:{node["component"]} (location unavailable)'
    bbox = comp.get('bbox')
    if bbox and len(bbox) >= 4:
        cx = (float(bbox[0]) + float(bbox[2]) - 1) / 2
        cy = (float(bbox[1]) + float(bbox[3]) - 1) / 2
        return (
            f'{node["layer"]}:{node["component"]} '
            f'@ approx {cx:.0f},{cy:.0f} ({_format_bbox(bbox)})'
        )
    return f'{node["layer"]}:{node["component"]} ({_format_bbox(None)})'


def _format_drill_location(drill_id: int, drills: dict[int, dict]) -> str:
    drill = drills.get(int(drill_id))
    if not drill:
        return f'drill {drill_id} (location unavailable; regenerate nets data with a newer pcbnets)'
    centroid = drill.get('centroid')
    if centroid and len(centroid) >= 2:
        return (
            f'drill {drill_id} @ {float(centroid[0]):.0f},{float(centroid[1]):.0f} '
            f'({_format_bbox(drill.get("bbox"))})'
        )
    return f'drill {drill_id} ({_format_bbox(drill.get("bbox"))})'


def cmd_explain(args: argparse.Namespace) -> int:
    """Explain which drill contacts connect two local components."""
    debug_dir = pathlib.Path(args.debug_dir).resolve()
    labels_path = debug_dir / 'layer-labels.npz'
    json_path = debug_dir / 'nets.json'
    if not labels_path.exists() or not json_path.exists():
        print(f'missing net debug files in {debug_dir}; run `pcbnets nets ... -o {debug_dir}` first', file=sys.stderr)
        return 1

    with np.load(labels_path) as data:
        labels = {name: data[name] for name in data.files}
    debug = json.loads(json_path.read_text())
    try:
        start = _parse_component_spec(args.start, labels)
        end = _parse_component_spec(args.end, labels)
    except argparse.ArgumentTypeError as e:
        print(str(e), file=sys.stderr)
        return 2
    path = explain_merge_path(debug, start, end)
    if not path:
        print(f'{args.start} and {args.end} are not connected by recorded drill merge edges')
        return 1

    components = _debug_component_index(debug)
    drills = _debug_drill_index(debug)
    print(f'{start[0]}:{start[1]} connects to {end[0]}:{end[1]} through {len(path)} drill edge(s):')
    for step in path:
        a = step['from']
        b = step['to']
        print(f'  {_format_component_location(a, components)}')
        print(f'    -- {_format_drill_location(step["drill"], drills)} --')
        print(f'  {_format_component_location(b, components)}')
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve
    build_dir = pathlib.Path(args.build_dir).resolve()
    if not build_dir.exists() or not (build_dir / 'grid.png').exists():
        candidate = build_dir / 'pcbnets-build'
        if (build_dir / 'drill.png').exists() or any(build_dir.glob('*.png')):
            print(f'no build artefacts in {build_dir}, rendering first...')
            render_args = argparse.Namespace(
                directory=str(build_dir),
                output=str(candidate),
                layers=None,
                drill='auto',
                via='auto',
                drill_grow=0,
                threshold=0,
                scale=1.0,
                cols=2,
                no_cache=False,
                nets=None,
                no_auto_invert=False,
                auto_align=False,
                invert=[],
                no_invert=[],
                offset=[],
                outer=None,
            )
            cmd_render(render_args)
            build_dir = candidate
        else:
            print(f'no PNGs in {build_dir}', file=sys.stderr)
            return 1
    serve(build_dir, host=args.host, port=args.port)
    return 0


def _validate_build_dir(build_dir: pathlib.Path) -> list[str]:
    """Return required build artefacts, or raise FileNotFoundError."""
    required = ['grid.png', 'idmap.png', 'meta.json']
    missing = [name for name in required if not (build_dir / name).exists()]
    if missing:
        miss = ', '.join(missing)
        raise FileNotFoundError(
            f'missing {miss} in {build_dir}; run `pcbnets render` first'
        )
    return required


def _bundle_file_names(build_dir: pathlib.Path) -> list[str]:
    """List build artefacts needed by the static HTML viewer."""
    names = _validate_build_dir(build_dir)

    # The viewer always needs the generated grid/id map/metadata.  Visual
    # overlay PNGs are optional and named in meta.json, but include the known
    # layer names as a fallback for older build dirs.
    visual_names: set[str] = set()
    try:
        meta = json.loads((build_dir / 'meta.json').read_text())
        visual_names.update(meta.get('silk_layers', []))
        visual_names.update(meta.get('mask_layers', []))
    except Exception:
        # Keep packaging useful even if metadata has a non-fatal issue.
        pass
    visual_names.update((*SILK_LAYERS, *MASK_LAYERS))

    for visual_name in sorted(visual_names):
        png = f'{visual_name}.png'
        if (build_dir / png).exists():
            names.append(png)

    return names


def _bundle_paths(build_dir: pathlib.Path) -> list[tuple[pathlib.Path, str]]:
    """Return build bundle files as ``(source_path, relative_name)`` pairs."""
    paths = [(build_dir / name, name) for name in _bundle_file_names(build_dir)]
    mips_dir = build_dir / 'mips'
    if mips_dir.is_dir():
        for path in sorted(p for p in mips_dir.rglob('*') if p.is_file()):
            paths.append((path, path.relative_to(build_dir).as_posix()))
    return paths


def _copy_static_bundle(build_dir: pathlib.Path,
                        out_dir: pathlib.Path,
                        title: str | None = None,
                        description: str | None = None) -> list[str]:
    """Copy the standalone viewer into ``out_dir``.

    Returns the file names written relative to ``out_dir``.
    """
    static_dir = pathlib.Path(__file__).parent / 'static'
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for src, name in _bundle_paths(build_dir):
        dst = out_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        written.append(name)

    shutil.copy2(static_dir / 'index.html', out_dir / 'index.html')
    written.append('index.html')

    if description:
        desc_path = pathlib.Path(description)
        if desc_path.exists():
            shutil.copy2(desc_path, out_dir / 'README.md')
            written.append('README.md')

    if title:
        html_path = out_dir / 'index.html'
        html = html_path.read_text()
        html = html.replace('<title>pcbnets viewer</title>',
                            f'<title>{title} — pcbnets</title>')
        html = html.replace('<h1>pcbnets viewer</h1>',
                            f'<h1>{title}</h1>')
        html_path.write_text(html)

    return written


def _zip_static_bundle(build_dir: pathlib.Path,
                       zip_path: pathlib.Path,
                       title: str | None = None,
                       description: str | None = None,
                       prefix: str | None = None) -> list[str]:
    """Write a deployable zip containing the static viewer."""
    static_dir = pathlib.Path(__file__).parent / 'static'
    file_paths = _bundle_paths(build_dir)

    prefix = (prefix or '').strip('/\\')
    if prefix:
        prefix = prefix + '/'

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for src, name in file_paths:
            arcname = prefix + name
            zf.write(src, arcname)
            written.append(arcname)

        # Patch the title in-memory so deploy does not need a temporary dir.
        html = (static_dir / 'index.html').read_text()
        if title:
            html = html.replace('<title>pcbnets viewer</title>',
                                f'<title>{title} — pcbnets</title>')
            html = html.replace('<h1>pcbnets viewer</h1>',
                                f'<h1>{title}</h1>')
        zf.writestr(prefix + 'index.html', html)
        written.append(prefix + 'index.html')

        if description:
            desc_path = pathlib.Path(description)
            if desc_path.exists():
                zf.write(desc_path, prefix + 'README.md')
                written.append(prefix + 'README.md')

    return written



def _copy_png_directory(src_dir: pathlib.Path, out_dir: pathlib.Path) -> list[str]:
    """Copy all PNG masks from ``src_dir`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for src in sorted(src_dir.glob('*.png')):
        dst = out_dir / src.name
        shutil.copy2(src, dst)
        written.append(src.stem)
    return written


def _apply_png_shift(src_dir: pathlib.Path,
                     out_dir: pathlib.Path,
                     name: str,
                     offset: tuple[int, int],
                     threshold: int = 0) -> None:
    """Shift one mask PNG from ``src_dir`` into ``out_dir`` in-place size."""
    dy, dx = offset
    src = src_dir / f'{name}.png'
    if not src.is_file():
        raise FileNotFoundError(f'cannot align {name}: missing {src}')
    arr = np.asarray(threshold_mask(Image.open(src), threshold).convert('L')) > 0
    shifted = _shift_bool(arr, dy, dx)
    _bool_to_mask_image(shifted).save(out_dir / f'{name}.png', optimize=True)


def _detect_via_alignment(directory: pathlib.Path,
                          layer_names: list[str],
                          drill_name: str,
                          via_name: str,
                          threshold: int,
                          prep_kwargs: dict,
                          auto_invert: bool) -> tuple[int, int] | None:
    """Return an automatic via/PTH offset, or None if no useful shift found."""
    if not layer_names:
        return None
    try:
        extra = [] if drill_name == via_name else [drill_name]
        masks = load_masks(directory, layer_names, via_name, threshold,
                           silk=False, extra_names=extra)
        _, _, report = prepare_masks(
            masks=masks,
            layer_names=layer_names,
            drill_name=via_name,
            auto_invert=auto_invert,
            auto_align=True,
            invert_overrides=prep_kwargs.get('invert_overrides', set()),
            no_invert_overrides=prep_kwargs.get('no_invert_overrides', set()),
            offset_overrides={},
            outer_layers=prep_kwargs.get('outer_layers'),
        )
    except Exception as e:
        print(f'  ! via auto-align skipped: {e}', file=sys.stderr)
        return None

    corr = report.corrections.get(via_name)
    if corr and corr.source == 'auto' and corr.offset != (0, 0):
        return corr.offset
    return None


def _detect_mask_alignments(directory: pathlib.Path,
                            layer_names: list[str],
                            drill_name: str,
                            via_name: str,
                            threshold: int,
                            prep_kwargs: dict,
                            auto_invert: bool) -> dict[str, tuple[int, int]]:
    """Return visual solder-mask offsets detected against F_Cu/B_Cu."""
    offsets: dict[str, tuple[int, int]] = {}
    if not layer_names:
        return offsets
    try:
        extra = [] if drill_name == via_name else [drill_name]
        masks = load_masks(directory, layer_names, via_name, threshold,
                           silk=True, extra_names=extra)
        arrs, _, _ = prepare_masks(
            masks=masks,
            layer_names=layer_names,
            drill_name=via_name,
            auto_invert=auto_invert,
            auto_align=False,
            invert_overrides=prep_kwargs.get('invert_overrides', set()),
            no_invert_overrides=prep_kwargs.get('no_invert_overrides', set()),
            offset_overrides={},
            outer_layers=prep_kwargs.get('outer_layers'),
        )
        corrections = _align_visual_masks(
            masks=masks,
            arrs=arrs,
            layer_names=layer_names,
            offset_overrides={},
            auto_align=True,
        )
    except Exception as e:
        print(f'  ! mask auto-align skipped: {e}', file=sys.stderr)
        return offsets

    for name, corr in corrections.items():
        off = tuple(corr.get('offset', [0, 0]))
        if corr.get('source') == 'auto' and off != (0, 0):
            offsets[name] = (int(off[0]), int(off[1]))
    return offsets


def cmd_align(args: argparse.Namespace) -> int:
    """Copy PNG masks to a new directory and apply optional alignment shifts."""
    directory = pathlib.Path(args.directory).resolve()
    out_dir = pathlib.Path(args.output).resolve()
    if not directory.is_dir():
        print(f'{directory} is not a directory', file=sys.stderr)
        return 1
    if directory == out_dir:
        print('align output must be a different directory from the input', file=sys.stderr)
        return 1

    drill_name = _resolve_drill_name(directory, args.drill)
    via_name = _resolve_via_name(directory, args.via)
    layers = _resolve_layers(directory, args.layers, drill_name, via_name)
    overrides = _collect_overrides(args)
    manual_offsets: dict[str, tuple[int, int]] = dict(overrides['offset_overrides'])

    print(f'aligning PNG masks from {directory}')
    logging.getLogger('pcbnets.render').info('  layers: %s', ', '.join(layers))
    logging.getLogger('pcbnets.render').info('  drill : %s', drill_name)
    logging.getLogger('pcbnets.render').info('  via   : %s', via_name)

    shift_info: dict[str, dict] = {}
    for name, off in manual_offsets.items():
        shift_info[name] = {
            'offset': [off[0], off[1]],
            'source': 'manual',
        }

    if args.auto_via and via_name not in manual_offsets:
        off = _detect_via_alignment(
            directory, layers, drill_name, via_name, args.threshold,
            overrides, auto_invert=not args.no_auto_invert,
        )
        if off:
            shift_info[via_name] = {
                'offset': [off[0], off[1]],
                'source': 'auto-via',
            }
            print(f'  {via_name:<14} align   shift (auto-via)    offset ({off[0]}, {off[1]})')
        else:
            print(f'  {via_name:<14} align   keep                no useful auto offset')

    if args.auto_masks:
        mask_offsets = _detect_mask_alignments(
            directory, layers, drill_name, via_name, args.threshold,
            overrides, auto_invert=not args.no_auto_invert,
        )
        for name, off in mask_offsets.items():
            if name in manual_offsets:
                continue
            shift_info[name] = {
                'offset': [off[0], off[1]],
                'source': 'auto-mask',
            }

    written = _copy_png_directory(directory, out_dir)
    for name, info in shift_info.items():
        dy, dx = info['offset']
        _apply_png_shift(directory, out_dir, name, (int(dy), int(dx)), args.threshold)
        print(f'  {name:<14} wrote shifted PNG offset ({dy}, {dx}) [{info["source"]}]')

    manifest = {
        'source_dir': str(directory),
        'layers': layers,
        'drill_name': drill_name,
        'via_name': via_name,
        'threshold': args.threshold,
        'shifts': shift_info,
    }
    with open(out_dir / 'alignment.json', 'w') as fp:
        json.dump(manifest, fp, indent=2)
        fp.write('\n')

    print(f'wrote {len(written)} PNG(s) to {out_dir}')
    print(f'wrote {out_dir / "alignment.json"}')
    print('render the aligned directory next, for example:')
    print(f'  pcbnets render {out_dir} -o ./pcbnets-build')
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    build_dir = pathlib.Path(args.build_dir).resolve()
    out_dir = pathlib.Path(args.output).resolve()

    try:
        written = _copy_static_bundle(
            build_dir, out_dir, title=args.title, description=args.description,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    print(f'static bundle written to {out_dir}')
    print(f'  {len(written)} file(s) copied')
    print(f'  serve {out_dir} over HTTP/HTTPS, or rsync it to a web host')
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    build_dir = pathlib.Path(args.build_dir).resolve()
    if args.output:
        zip_path = pathlib.Path(args.output).resolve()
    else:
        zip_path = pathlib.Path.cwd() / f'{build_dir.name}.zip'

    try:
        written = _zip_static_bundle(
            build_dir, zip_path, title=args.title,
            description=args.description, prefix=args.prefix,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    print(f'deploy bundle written to {zip_path}')
    print(f'  {len(written)} file(s) packaged')
    if args.prefix:
        prefix_stripped = args.prefix.strip("/\\")
        print(f'  zip root folder: {prefix_stripped}')
    print('  unzip on a web server and open index.html via HTTP/HTTPS')
    return 0


def cmd_gerber(args: argparse.Namespace) -> int:
    """Rasterise Gerber/Excellon files in a directory to PNG masks."""
    _configure_cli_logging()
    log = logging.getLogger('pcbnets.gerber')
    directory = pathlib.Path(args.directory).resolve()
    out_dir = pathlib.Path(args.output).resolve() if args.output \
              else directory / 'pngs'

    if not directory.is_dir():
        log.error('%s is not a directory', directory)
        return 1

    # Layer mapping is explicit or auto-detected.  Deliberately do NOT read
    # layers.json automatically: a stale mapping is too easy to carry forward
    # after changing boards, layer count, or drill split.  Use --mapping
    # manually when you really want a hand-edited mapping.
    if args.mapping:
        from .gerber import read_layers_json
        mapping = read_layers_json(pathlib.Path(args.mapping))
        log.info('using mapping from %s', args.mapping)
        created = False
    else:
        mapping = detect_layers(directory)
        write_layers_json(directory, mapping)
        if args.force_detect:
            log.info('wrote (re-detected) %s', directory / "layers.json")
        else:
            log.info('wrote detected %s', directory / "layers.json")
        created = True

    if not mapping:
        log.error('no Gerber/Excellon files recognised in directory')
        log.error('  files seen:')
        for p in sorted(directory.iterdir()):
            if p.is_file():
                log.error('    %s', p.name)
        return 1

    log.info('rasterising %s layers at %s DPI into %s',
             len(mapping), args.dpi, out_dir)
    from .gerber import _sort_key
    for name in sorted(mapping, key=_sort_key):
        log.info('  %-14s ← %s', name, mapping[name])

    try:
        written = rasterise(
            directory, mapping, out_dir, dpi=args.dpi
        )
    except GerbvMissingError as e:
        log.error('%s', e)
        return 2

    log.info('wrote %s PNG(s) to %s', len(written), out_dir)
    if created:
        log.info('Edit %s if needed, then pass it explicitly with --mapping.',
                 directory / "layers.json")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """One-shot: gerber + render in sequence, writing into a single output dir."""
    out_dir = pathlib.Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pngs_dir = out_dir / 'pngs'
    build_dir = out_dir / 'build'

    gerber_args = argparse.Namespace(
        directory=args.directory,
        output=str(pngs_dir),
        dpi=args.dpi,
        mapping=args.mapping,
        force_detect=args.force_detect
    )
    rc = cmd_gerber(gerber_args)
    if rc != 0:
        return rc

    render_args = argparse.Namespace(
        directory=str(pngs_dir),
        output=str(build_dir),
        layers=args.layers,
        drill=getattr(args, 'drill', 'auto'),
        via=getattr(args, 'via', 'auto'),
        drill_grow=args.drill_grow,
        threshold=0,
        scale=args.scale,
        cols=args.cols,
        no_cache=False,
        nets=args.nets,
        no_auto_invert=args.no_auto_invert,
        auto_align=args.auto_align,
        invert=args.invert or [],
        no_invert=args.no_invert or [],
        offset=args.offset or [],
        outer=args.outer,
    )
    return cmd_render(render_args)


def cmd_index(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.directory).resolve()
    if not root.is_dir():
        print(f'{root} is not a directory', file=sys.stderr)
        return 1

    boards = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        meta_path = sub / 'meta.json'
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        boards.append({
            'name': sub.name,
            'layers': meta.get('layers', []),
            'n_nets': meta.get('n_nets', 0),
        })

    title = args.title or 'PCB boards'
    html = [
        '<!DOCTYPE html><html><head><meta charset="utf-8">',
        f'<title>{title}</title>',
        '<style>',
        'body { background:#1a1a1a; color:#e0e0e0;',
        '       font-family:-apple-system,BlinkMacSystemFont,sans-serif;',
        '       padding:24px; max-width:720px; margin:0 auto; }',
        'h1 { font-size:22px; }',
        'a { color:#4ac; text-decoration:none; }',
        'a:hover { text-decoration:underline; }',
        '.board { padding:12px 0; border-bottom:1px solid #333; }',
        '.meta { color:#888; font-size:12px; margin-top:4px; }',
        '</style></head><body>',
        f'<h1>{title}</h1>',
    ]
    for b in boards:
        layers = ', '.join(b['layers'])
        html.append('<div class="board">')
        html.append(f'<a href="./{b["name"]}/">{b["name"]}</a>')
        html.append(
            f'<div class="meta">{len(b["layers"])} layers '
            f'({layers}) · {b["n_nets"]} nets</div>'
        )
        html.append('</div>')
    html.append('</body></html>')

    out_path = root / 'index.html'
    out_path.write_text('\n'.join(html))
    print(f'wrote gallery index: {out_path}')
    print(f'  {len(boards)} boards listed')
    return 0


# ---------- parser ----------

def _add_audit_flags(p: argparse.ArgumentParser, *, include_auto_align: bool = True) -> None:
    p.add_argument('--no-auto-invert', action='store_true',
                   help='Disable automatic polarity inversion for inner planes')
    if include_auto_align:
        p.add_argument('--auto-align', action='store_true', default=False,
                       help='Apply automatic via/mask alignment inline. Normally use `pcbnets align` instead.')
        # Backward-compatible no-op-ish spelling.  Older builds auto-aligned by
        # default and used --no-auto-align to disable it; alignment is now opt-in.
        p.add_argument('--no-auto-align', action='store_false', dest='auto_align',
                       help=argparse.SUPPRESS)
    p.add_argument('--invert', action='append', default=[], metavar='LAYER',
                   help='Force-invert this layer (repeatable)')
    p.add_argument('--no-invert', action='append', default=[], metavar='LAYER',
                   help='Force-do-not-invert this layer (overrides auto)')
    p.add_argument('--offset', action='append', default=[], nargs=2,
                   metavar=('LAYER', 'DY,DX'),
                   help='Manually shift this layer (prefer `pcbnets align`; still useful for quick tests)')
    p.add_argument('--outer', nargs='+', default=None, metavar='LAYER',
                   help='Override the set of "outer" layers (default: F_Cu B_Cu)')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='pcbnets',
        description='Interactive PCB net explorer from rasterised Gerber layers.',
    )
    p.add_argument('--version', action='version', version=f'pcbnets {__version__}')
    sub = p.add_subparsers(dest='command', required=True)

    pr = sub.add_parser('render', help='Process a directory of PNGs into a build dir')
    pr.add_argument('directory')
    pr.add_argument('-o', '--output', default='./pcbnets-build')
    pr.add_argument('--layers', nargs='+')
    pr.add_argument('--drill', default='auto',
                    help='Physical drill-hole mask name (default: auto)')
    pr.add_argument('--via', default='auto',
                    help='Electrical vertical-connector mask name (default: auto: via, PTH, drill)')
    pr.add_argument('--drill-grow', type=int, default=0, dest='drill_grow')
    pr.add_argument('--threshold', type=int, default=0)
    pr.add_argument('--scale', type=float, default=1.0)
    pr.add_argument('--cols', type=int, default=2)
    pr.add_argument('--no-cache', action='store_true')
    pr.add_argument('--nets',
                    help='Use net-labels.npz from `pcbnets nets -o DIR` instead of extracting/merging again')
    _add_audit_flags(pr)
    pr.set_defaults(func=cmd_render)

    pn = sub.add_parser('nets', help='Run net finding only and write debug data')
    pn.add_argument('directory')
    pn.add_argument('-o', '--output', default='./pcbnets-nets')
    pn.add_argument('--layers', nargs='+')
    pn.add_argument('--drill', default='auto',
                    help='Physical drill-hole mask name (default: auto)')
    pn.add_argument('--via', default='auto',
                    help='Electrical vertical-connector mask name (default: auto: via, PTH, drill)')
    pn.add_argument('--drill-grow', type=int, default=0, dest='drill_grow')
    pn.add_argument('--threshold', type=int, default=0)
    _add_audit_flags(pn)
    pn.set_defaults(func=cmd_nets)

    px = sub.add_parser('explain', help='Explain why two local components are merged')
    px.add_argument('debug_dir',
                    help='Directory written by `pcbnets nets`')
    px.add_argument('--from', required=True, dest='start',
                    help='Start as LAYER:COMPONENT or LAYER:X,Y')
    px.add_argument('--to', required=True, dest='end',
                    help='End as LAYER:COMPONENT or LAYER:X,Y')
    px.set_defaults(func=cmd_explain)

    pa = sub.add_parser('audit', help='Detect polarity/alignment issues without rendering')
    pa.add_argument('directory')
    pa.add_argument('-o', '--output', help='Write debug overlay PNG to this path')
    pa.add_argument('--layers', nargs='+')
    pa.add_argument('--drill', default='auto',
                    help='Physical drill-hole mask name (default: auto)')
    pa.add_argument('--via', default='auto',
                    help='Electrical vertical-connector mask name (default: auto: via, PTH, drill)')
    pa.add_argument('--threshold', type=int, default=0)
    _add_audit_flags(pa)
    pa.set_defaults(func=cmd_audit)

    palign = sub.add_parser('align', help='Copy PNG masks and apply optional alignment shifts')
    palign.add_argument('directory')
    palign.add_argument('-o', '--output', required=True,
                        help='Output PNG directory; must differ from input')
    palign.add_argument('--layers', nargs='+')
    palign.add_argument('--drill', default='auto',
                        help='Physical drill-hole mask name (default: auto)')
    palign.add_argument('--via', default='auto',
                        help='Electrical connector mask name (default: auto: via, PTH, drill)')
    palign.add_argument('--threshold', type=int, default=0)
    palign.add_argument('--auto-via', action='store_true',
                        help='Auto-detect and apply the via/PTH offset against copper')
    palign.add_argument('--auto-masks', action='store_true',
                        help='Auto-detect and apply F_Mask/B_Mask visual offsets')
    _add_audit_flags(palign, include_auto_align=False)
    palign.set_defaults(func=cmd_align)

    pdrill = sub.add_parser(
        'drill_identify',
        aliases=['drill-identify'],
        help='Classify generic drill holes into PTH/via and NPTH PNG masks',
    )
    pdrill.add_argument('directory')
    pdrill.add_argument('-o', '--output', required=True,
                        help='Output PNG directory with PTH.png, via.png, NPTH.png, drill.png, and drill-identify.json')
    pdrill.add_argument('--choices',
                        help='Existing drill-identify.json with edited plated decisions to apply')
    pdrill.add_argument('--excellon',
                        help='Split this Excellon drill file with gerbonara using object_index decisions from --choices')
    pdrill.add_argument('--layers', nargs='+')
    pdrill.add_argument('--drill', default='auto',
                        help='Generic/physical drill-hole mask name (default: auto)')
    pdrill.add_argument('--threshold', type=int, default=0)
    pdrill.add_argument('--dpi', type=int, default=1000,
                        help='DPI to use when --excellon rasterises Gerbers for classification (default: 1000)')
    _add_audit_flags(pdrill)
    pdrill.set_defaults(func=cmd_drill_identify)

    ps = sub.add_parser('serve', help='Run the interactive viewer')
    ps.add_argument('build_dir')
    ps.add_argument('--host', default='127.0.0.1')
    ps.add_argument('--port', type=int, default=8000)
    ps.set_defaults(func=cmd_serve)

    pe = sub.add_parser('export', help='Export a static HTML bundle directory')
    pe.add_argument('build_dir')
    pe.add_argument('-o', '--output', required=True)
    pe.add_argument('--title')
    pe.add_argument('--description')
    pe.set_defaults(func=cmd_export)

    pd = sub.add_parser('deploy', help='Zip a static viewer bundle for web upload')
    pd.add_argument('build_dir')
    pd.add_argument('-o', '--output',
                    help='Output zip file (default: ./<build-dir-name>.zip)')
    pd.add_argument('--title')
    pd.add_argument('--description')
    pd.add_argument('--prefix',
                    help='Optional folder prefix inside the zip, e.g. board-v1')
    pd.set_defaults(func=cmd_deploy)

    pi = sub.add_parser('index', help='Generate a gallery for a directory of boards')
    pi.add_argument('directory')
    pi.add_argument('--title')
    pi.set_defaults(func=cmd_index)

    pg = sub.add_parser('gerber',
                        help='Rasterise Gerber/Excellon files to PNG masks (needs gerbv)')
    pg.add_argument('directory', help='Directory containing Gerber/Excellon files')
    pg.add_argument('-o', '--output',
                    help='Output PNG directory (default: <directory>/pngs)')
    pg.add_argument('--dpi', type=int, default=1000,
                    help='Rasterisation DPI (default: 1000)')
    pg.add_argument('--mapping',
                    help='Path to a layers.json (skips auto-detect)')
    pg.add_argument('--force-detect', action='store_true', dest='force_detect',
                    help='Re-detect layers and overwrite layers.json '
                         '(useful after upgrading pcbnets)')
    pg.set_defaults(func=cmd_gerber)

    pall = sub.add_parser('all',
                          help='One-shot: gerber + render (needs gerbv)')
    pall.add_argument('directory', help='Directory containing Gerber/Excellon files')
    pall.add_argument('-o', '--output', required=True,
                      help='Output directory; will contain pngs/ and build/')
    pall.add_argument('--dpi', type=int, default=1000)
    pall.add_argument('--mapping', help='Path to a layers.json')
    pall.add_argument('--force-detect', action='store_true', dest='force_detect',
                      help='Re-detect layers and overwrite layers.json')
    pall.add_argument('--layers', nargs='+')
    pall.add_argument('--drill', default='auto',
                       help='Physical drill-hole mask name (default: auto)')
    pall.add_argument('--via', default='auto',
                       help='Electrical vertical-connector mask name (default: auto: via, PTH, drill)')
    pall.add_argument('--drill-grow', type=int, default=0, dest='drill_grow')
    pall.add_argument('--scale', type=float, default=1.0)
    pall.add_argument('--cols', type=int, default=2)
    pall.add_argument('--nets',
                      help='Use net-labels.npz from `pcbnets nets -o DIR` during render')
    _add_audit_flags(pall)
    pall.set_defaults(func=cmd_all)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
