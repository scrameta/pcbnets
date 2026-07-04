"""Gerber/Excellon detection and rasterisation.

Detects which files in a directory map to which KiCad-style layers,
writes a ``layers.json`` so the mapping is inspectable and editable,
and rasterises each file with ``gerbv``.  The normal path estimates a
common Gerber coordinate crop and renders every layer into that same
origin/window, so modern KiCad/Altium layers stay aligned without producing
enormous origin-to-board PNGs.

gerbv is a runtime dependency only for this module; the rest of pcbnets
works on any PNG masks regardless of how they were produced.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import shutil
import subprocess
import time
from typing import Iterable

from PIL import Image


log = logging.getLogger('pcbnets.gerber')


# --- detection ---

# Canonical KiCad layer names. The pattern matches against the file *stem*
# (extension stripped) with leading "project-" prefix removed.
# Modern KiCad uses underscores (F_Cu); older versions used dots (F.Cu).
# We accept both.

LAYER_PATTERNS: list[tuple[str, re.Pattern]] = [
    ('F_Cu',         re.compile(r'F[._]Cu$',                       re.IGNORECASE)),
    ('B_Cu',         re.compile(r'B[._]Cu$',                       re.IGNORECASE)),
    ('F_Silkscreen', re.compile(r'F[._](?:SilkS|Silkscreen|Silk)$', re.IGNORECASE)),
    ('B_Silkscreen', re.compile(r'B[._](?:SilkS|Silkscreen|Silk)$', re.IGNORECASE)),
    ('F_Mask',       re.compile(r'F[._]Mask$',                     re.IGNORECASE)),
    ('B_Mask',       re.compile(r'B[._]Mask$',                     re.IGNORECASE)),
    ('Edge_Cuts',    re.compile(r'Edge[._]Cuts$',                  re.IGNORECASE)),
]
INNER_RE = re.compile(r'In(\d+)[._]Cu$', re.IGNORECASE)
PTH_RE   = re.compile(r'(?:^|[-_])PTH$', re.IGNORECASE)
NPTH_RE  = re.compile(r'(?:^|[-_])NPTH$', re.IGNORECASE)
VIA_RE   = re.compile(r'(?:^|[-_])VIAS?$', re.IGNORECASE)

GERBER_EXTS = {'.gbr', '.gtl', '.gbl', '.gts', '.gbs', '.gto', '.gbo', '.gko', '.gm1',
               '.gtp', '.gbp', '.gpb', '.gpt', '.gpl',
               '.g1', '.g2', '.g3', '.g4', '.g5', '.g6', '.g7', '.g8',
               '.gp1', '.gp2', '.gp3', '.gp4'}
DRILL_EXTS  = {'.drl', '.txt', '.xln', '.nc'}


# Altium-style extension → canonical KiCad-style layer name.
# Used as a fallback when filename-pattern detection (KiCad style) fails.
# A few EDA tools (Altium, OrCAD, older EAGLE exports) encode the layer
# function in the extension rather than the stem.
EXTENSION_LAYER_MAP = {
    '.gtl': 'F_Cu',
    '.gbl': 'B_Cu',
    '.gto': 'F_Silkscreen',
    '.gbo': 'B_Silkscreen',
    '.gts': 'F_Mask',
    '.gbs': 'B_Mask',
    '.gtp': 'F_Paste',
    '.gbp': 'B_Paste',
    '.gko': 'Edge_Cuts',
    '.gm1': 'Edge_Cuts',
    # Inner layers — numbered: .g1 → In1_Cu, .g2 → In2_Cu, etc.
    # Plane layers (.gp1, .gp2) also map to inner copper numerically.
    # Handled in detect_layers because the number needs to be parsed.
}


class GerbvMissingError(RuntimeError):
    """Raised when ``gerbv`` is not on PATH."""


def check_gerbv() -> str:
    """Return the path to gerbv or raise ``GerbvMissingError``."""
    path = shutil.which('gerbv')
    if not path:
        raise GerbvMissingError(
            'gerbv is required for the `gerber` subcommand but was not '
            'found on PATH.\n'
            '  macOS:    brew install gerbv\n'
            '  Debian:   apt install gerbv\n'
            '  Arch:     pacman -S gerbv\n'
            '  Windows:  https://gerbv.github.io/'
        )
    return path


def detect_layers(directory: pathlib.Path) -> dict[str, str]:
    """Scan ``directory`` for Gerber/Excellon files; map canonical → filename.

    Canonical names follow KiCad: ``F_Cu``, ``In1_Cu``, ``B_Cu``,
    ``F_Silkscreen``, ``B_Silkscreen``, ``via``, ``PTH``, ``NPTH``,
    ``drill``, ``Edge_Cuts``.
    Detection happens in three passes, first-match-wins:

    1. KiCad-style filename patterns (``F_Cu``, ``In1_Cu``, etc.) on the
       stem after stripping a leading ``project-`` prefix.
    2. Altium-style file extensions (``.gtl`` → ``F_Cu``, ``.g1`` →
       ``In1_Cu``, etc.) as a fallback.
    3. Via-named drill files map to ``via`` for electrical connectivity.
    4. Plain drill files (``.drl`` with no PTH/NPTH/Via suffix) map to
       ``drill`` as a physical-hole mask.  The render step may still fall
       back to it as the connectivity mask if no ``via``/``PTH`` exists.
    """
    directory = pathlib.Path(directory)
    mapping: dict[str, str] = {}

    altium_inner_re = re.compile(r'^\.gp?(\d+)$', re.IGNORECASE)

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        is_gerber = ext in GERBER_EXTS or altium_inner_re.match(ext)
        is_drill  = ext in DRILL_EXTS
        if not is_gerber and not is_drill:
            continue

        stem = path.stem
        layer_part = stem.rsplit('-', 1)[-1] if '-' in stem else stem

        # Drill files first.
        if is_drill:
            if VIA_RE.search(stem):
                mapping.setdefault('via', path.name)
            elif PTH_RE.search(stem):
                mapping.setdefault('PTH', path.name)
            elif NPTH_RE.search(stem):
                mapping.setdefault('NPTH', path.name)
            else:
                mapping.setdefault('drill', path.name)
            continue

        # Pass 1: inner copper by KiCad-style filename (numbered).
        m = INNER_RE.match(layer_part)
        if m:
            mapping.setdefault(f'In{int(m.group(1))}_Cu', path.name)
            continue

        # Pass 1 (cont): other KiCad-style patterns.
        matched = False
        for canonical, pattern in LAYER_PATTERNS:
            if pattern.match(layer_part):
                mapping.setdefault(canonical, path.name)
                matched = True
                break
        if matched:
            continue

        # Pass 2: Altium-style extensions.
        m = altium_inner_re.match(ext)
        if m:
            mapping.setdefault(f'In{int(m.group(1))}_Cu', path.name)
            continue
        if ext in EXTENSION_LAYER_MAP:
            mapping.setdefault(EXTENSION_LAYER_MAP[ext], path.name)
            continue

        # Unrecognised gerber file — no canonical mapping. Skip silently;
        # the user can edit layers.json to include it manually.

    return mapping


def _sort_key(name: str) -> tuple:
    """Stable order: F_Cu, In1..InN, B_Cu, PTH, NPTH, silks, masks, then alpha."""
    if name == 'F_Cu':         return (0, 0)
    if name == 'B_Cu':         return (0, 9999)
    m = re.match(r'In(\d+)_Cu', name)
    if m:                      return (0, int(m.group(1)))
    if name == 'via':          return (1, 0)
    if name == 'PTH':          return (1, 1)
    if name == 'drill':        return (1, 2)
    if name == 'NPTH':         return (1, 3)
    if name == 'F_Silkscreen': return (2, 0)
    if name == 'B_Silkscreen': return (2, 1)
    if name == 'F_Mask':       return (3, 0)
    if name == 'B_Mask':       return (3, 1)
    if name == 'Edge_Cuts':    return (4, 0)
    return (5, hash(name) & 0xffff)


def write_layers_json(directory: pathlib.Path, mapping: dict[str, str]) -> pathlib.Path:
    """Write ``layers.json`` in ``directory`` in canonical key order."""
    path = pathlib.Path(directory) / 'layers.json'
    ordered = dict(sorted(mapping.items(), key=lambda kv: _sort_key(kv[0])))
    payload = {
        '_comment': ('Generated by `pcbnets gerber`. Edit the entries below '
                     'to override auto-detection, then re-run.'),
        'layers': ordered,
    }
    with open(path, 'w') as fp:
        json.dump(payload, fp, indent=2)
        fp.write('\n')
    return path


def read_layers_json(path: pathlib.Path) -> dict[str, str]:
    """Read a layers.json, returning ``{canonical: filename}``.

    Tolerant of both the rich format (``{"layers": {...}}``) and the bare
    flat format (``{"F_Cu": "...", ...}``) for hand-edited files.
    """
    with open(path) as fp:
        data = json.load(fp)
    if isinstance(data, dict) and 'layers' in data and isinstance(data['layers'], dict):
        return dict(data['layers'])
    # Bare flat form: strip the underscore comment key if present.
    return {k: v for k, v in data.items() if not k.startswith('_')}


def load_or_create_layers_json(
    directory: pathlib.Path,
) -> tuple[dict[str, str], bool]:
    """Return ``(mapping, created)``.

    Reads ``layers.json`` if present. Otherwise auto-detects, writes the
    file for the user to inspect/edit, and returns the detected mapping.
    """
    directory = pathlib.Path(directory)
    path = directory / 'layers.json'
    if path.exists():
        return read_layers_json(path), False
    mapping = detect_layers(directory)
    write_layers_json(directory, mapping)
    return mapping, True



# --- coordinate-space crop estimation ---

_DRILL_CANONICAL = {'drill', 'via', 'PTH', 'NPTH'}

# --- rasterisation ---

def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours}h{minutes:02d}m{secs:02d}s'
    return f'{minutes}m{secs:02d}s'


def _rasterise_one(
                   all_sources: list[pathlib.Path],
                   selected_src: int,
                   output: pathlib.Path,
                   dpi: int,
                   progress_interval: float = 30.0) -> None:
    """Run gerbv on all these files, just outputting one
       Done this way to get alignment
    """
    sources = [str(x) for x in all_sources]
    foregrounds = []
    for i, src in enumerate(all_sources):
        if i == selected_src:
            foregrounds.append("--foreground=#FFFFFFFF")
        else:
            foregrounds.append("--foreground=#FFFFFF00")
    cmd = [
        'gerbv',
        '--export=png',
        f'--output={output}',
        f'--dpi={dpi}x{dpi}',
        '--background=#000000',
        *foregrounds,
        '--border=0',
        *sources,
    ]
    started = time.monotonic()
    next_progress = started + progress_interval
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    while proc.poll() is None:
        now = time.monotonic()
        if now >= next_progress:
            log.info(
                '    still rasterising %s in gerbv (%s elapsed)',
                output.name,
                _format_elapsed(now - started),
            )
            next_progress = now + progress_interval
        time.sleep(min(1.0, max(0.0, next_progress - now)))
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f'gerbv failed on {all_sources[selected_src].name}:\n{stderr.strip()}'
        )
    if not output.exists():
        raise RuntimeError(
            f'gerbv claimed success but {output} was not written'
        )
    if stdout.strip():
        log.debug('gerbv stdout for %s:\n%s', output.name, stdout.strip())

def rasterise(
    source_dir: pathlib.Path,
    mapping: dict[str, str],
    output_dir: pathlib.Path,
    dpi: int = 1000,
    layers: Iterable[str] | None = None
) -> list[pathlib.Path]:
    """Create an aligned PNG for each layer using gerbv

    Output PNGs are named ``{canonical_name}.png`` in ``output_dir``. The
    electrical drill mask is also aliased to ``via.png`` where possible so
    the render step can distinguish physical holes from electrical vertical
    connectors.

    Returns the list of paths written.
    """
    check_gerbv()
    source_dir = pathlib.Path(source_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = list(layers) if layers else list(mapping.keys())
    targets = [t for t in targets if t in mapping]
    if not targets:
        return []

    # Pass 1: render each file.  In the normal modern-Gerber path every
    # layer is rendered into the same origin/window, so CAD coordinates stay
    # aligned while the output is cropped close to the board instead of to
    # absolute origin 0,0.
    written: list[pathlib.Path] = []
    rendered_sizes: dict[str, tuple[int, int]] = {}
    sources = []
    for name in targets:
        fname = mapping[name]
        src = source_dir / fname
        sources.append(src)
    for tgt_i,name in enumerate(targets):
        fname = mapping[name]
        src = source_dir / fname
        if not src.is_file():
            log.warning('  warning: %s not found (skipping %s)', src, name)
            continue
        out = output_dir / f'{name}.png'
        started = time.monotonic()
        log.info('  rasterising %s.png from %s', name, fname)
        try:
            _rasterise_one(sources, tgt_i, out, dpi)
        except RuntimeError as e:
            log.error('  ! %s', e)
            continue
        size = Image.open(out).size
        rendered_sizes[name] = size
        log.info('    → %s×%s px (%s)',
                 size[0], size[1], _format_elapsed(time.monotonic() - started))
        written.append(out)

    # Convenience aliases for downstream use:
    #   via.png   = preferred electrical connectivity mask
    #   drill.png = physical hole mask, falling back to the electrical mask
    #               only when no explicit drill mask was rendered
    pth_path = output_dir / 'PTH.png'
    via_path = output_dir / 'via.png'
    drill_alias = output_dir / 'drill.png'

    if not via_path.exists() and pth_path.exists():
        shutil.copy2(pth_path, via_path)
        written.append(via_path)
    if not drill_alias.exists():
        source = None
        if pth_path.exists():
            source = pth_path
        elif via_path.exists():
            source = via_path
        if source is not None:
            shutil.copy2(source, drill_alias)
            written.append(drill_alias)

    return written
