"""Gerber/Excellon detection and rasterisation.

Detects which files in a directory map to which KiCad-style layers,
writes a ``layers.json`` so the mapping is inspectable and editable,
and exports each file to SVG with ``gerbv``. PNG masks are then rasterised
from those SVGs, so vector and raster outputs share the exact same geometry
and avoid backend-specific offsets.

gerbv is a runtime dependency only for this module; the rest of pcbnets
works on any PNG masks regardless of how they were produced.
"""

from __future__ import annotations

import importlib
import json
import logging
import pathlib
import re
import shutil
import subprocess
import time
from typing import Iterable
import xml.etree.ElementTree as ET

from PIL import Image

Image.MAX_IMAGE_PIXELS = 1_000_000_000  # 1 gigapixel


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


# gerbv's Cairo *SVG* backend does NOT honour per-file foreground alpha: it
# draws every loaded layer opaque and flattens them into one identical file
# (every layer's SVG came out byte-for-byte the same, an OR-merge with no way
# to tell paths apart). So we render each layer's SVG from its *single* source
# file. A lone file
# with no window makes gerbv emit an empty SVG (degenerate extent), so we pin an
# explicit --origin and --window_inch derived once from the all-loaded render.
# With identical origin+window, gerbv emits an identical viewBox and transform
# matrix for every layer (verified: same `matrix(72,0,0,-72,e,f)` and viewBox
# across F_Cu/inner/B_Cu), so the per-layer SVGs register against each other and
# against the PNG window with no post-normalisation. The canvas stays
# transparent (gerbv never writes --background as a rect in SVG), which is what
# we want for compositing.

# gerbv's SVG canvas is points at 72 pt/inch, y-flipped:
#   svg_x = 72*X_in + e ;  svg_y = -72*Y_in + f
# so window_inch = canvas_pt / 72, and the gerber origin (lower-left) inverts
# from the transform. We recover both from one all-loaded SVG probe.
_SVG_MATRIX_RE = re.compile(r'matrix\(([^)]*)\)')

def _parse_svg_length(s: str) -> float:
    """
    Return numeric part of an SVG length.

    For our gerbv use case, we deliberately ignore the written unit and work
    in the SVG coordinate/user units used by the transform matrix.
    """
    m = re.match(r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)', s)
    if not m:
        raise ValueError(f"bad SVG length: {s!r}")
    return float(m.group(1))


def _parse_matrix_values(s: str) -> tuple[float, float, float, float, float, float]:
    # SVG matrices may be comma-separated or whitespace-separated.
    vals = [v for v in re.split(r'[\s,]+', s.strip()) if v]
    if len(vals) != 6:
        raise ValueError(f"bad SVG matrix: {s!r}")
    return tuple(float(v) for v in vals)


def _derive_window(probe_svg: pathlib.Path) -> tuple[float, float, float, float]:
    """From an all-loaded SVG, return (origin_x, origin_y, width_in, height_in).

    Robust to gerbv SVGs that use either:
      width="123pt" height="456pt"
    or:
      width="123" height="456" viewBox="0 0 123 456"

    The transform matrix gives the real SVG-units-per-inch scale.
    """
    text = probe_svg.read_text()

    root = ET.fromstring(text)
    width_attr = root.attrib.get("width")
    height_attr = root.attrib.get("height")

    if width_attr is not None and height_attr is not None:
        w_svg = _parse_svg_length(width_attr)
        h_svg = _parse_svg_length(height_attr)
    else:
        viewbox = root.attrib.get("viewBox")
        if not viewbox:
            raise RuntimeError(f"could not parse SVG size from {probe_svg.name}")
        vb = [float(x) for x in re.split(r'[\s,]+', viewbox.strip())]
        if len(vb) != 4:
            raise RuntimeError(f"bad SVG viewBox in {probe_svg.name}: {viewbox!r}")
        _, _, w_svg, h_svg = vb

    mat = _SVG_MATRIX_RE.search(text)
    if not mat:
        raise RuntimeError(f"could not parse transform matrix from {probe_svg.name}")

    a, b, c, d, e, f = _parse_matrix_values(mat.group(1))

    if abs(b) > 1e-9 or abs(c) > 1e-9:
        raise RuntimeError(f"unexpected rotated/skewed SVG matrix in {probe_svg.name}")

    # a/d are SVG units per Gerber inch. Usually a ~= +72 or +96,
    # and d is negative because SVG y goes downward.
    width_in = w_svg / abs(a)
    height_in = h_svg / abs(d)

    # Invert:
    #   svg_x = a * X_in + e
    #   svg_y = d * Y_in + f
    #
    # Lower-left of exported window is SVG x=0, y=h_svg.
    origin_x = (0.0 - e) / a
    origin_y = (h_svg - f) / d

    return origin_x, origin_y, width_in, height_in


def _rasterise_all_svg(all_sources: list[pathlib.Path],
                       output: pathlib.Path) -> None:
    """Export one all-loaded SVG, used only to probe the shared window.

    All layers loaded, no explicit window: gerbv sizes the canvas to the full
    board extent, which is exactly the common frame we want to pin the
    per-layer renders to. Content is irrelevant here — we only read the header
    and transform — so this is discarded after ``_derive_window``.
    """
    cmd = [
        'gerbv',
        '--export=svg',
        '--background=#010101','--foreground=#FEFEFE', #Black/white does not work, hence almost...
        f'--output={output}',
        '--border=0',
        *[str(x) for x in all_sources],
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True)
    if proc.returncode != 0 or not output.exists():
        raise RuntimeError(
            f'gerbv window-probe SVG failed:\n{proc.stderr.strip()}'
        )

"""Make black transparent in black/white SVGs (e.g. gerbv exports).

Wraps the original drawing in a luminance <mask> applied to a single
solid rect. Painting order is preserved inside the mask, so negative
plane layers work correctly: clearances painted in black over the
copper pour become transparent holes instead of vanishing (no white
square!). White areas become the rect's fill colour, so layers are
trivially recolourable.

Renders correctly in browsers, resvg, librsvg, Inkscape. (cairosvg
treats masks as alpha masks and will get this wrong.)
"""
def _svg_to_transparent(path_in, path_out, fill="#FFFFFF"):
    text = pathlib.Path(path_in).read_text()
    m = re.search(r'<svg\b[^>]*>', text)
    header = text[:m.end()]
    header = re.sub(r'<svg\b', '<svg shape-rendering="crispEdges"', header, count=1)    
    body = text[m.end():text.rindex('</svg>')]
    # Snap gerbv's off-black/off-white so luminance hits exactly 0/255.
    vb = re.search(r'viewBox="([\d.eE+-]+)[ ,]+([\d.eE+-]+)[ ,]+'
                   r'([\d.eE+-]+)[ ,]+([\d.eE+-]+)"', header)
    if not vb:
        raise SystemExit(f"{path_in}: no viewBox found")
    x, y, w, h = vb.groups()
    out = (f'{header}\n'
           f'<mask id="lum" maskUnits="userSpaceOnUse" '
           f'x="{x}" y="{y}" width="{w}" height="{h}">{body}</mask>\n'
           f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
           f'fill="{fill}" mask="url(#lum)"/>\n</svg>\n')
    pathlib.Path(path_out).write_text(out)

def _optimise_svg_with_tools(svg_path: pathlib.Path) -> tuple[int, int]:
    """Optimise an SVG in-place using svgo followed by scour."""
    before = svg_path.stat().st_size
    tmp = svg_path.with_name(f'{svg_path.stem}.svgo.tmp{svg_path.suffix}')
    try:
        svgo = subprocess.run(
            ['svgo', '-i', str(svg_path), '-o', str(tmp)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if svgo.returncode != 0:
            raise RuntimeError(
                'svgo failed while optimising '
                f'{svg_path.name}: {svgo.stderr.strip() or svgo.stdout.strip()}'
            )
        scour = subprocess.run(
            [
                'scour',
                '-i', str(tmp),
                '-o', str(svg_path),
                '--enable-viewboxing',
                '--enable-id-stripping',
                '--enable-comment-stripping',
                '--shorten-ids',
                '--indent=none',
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if scour.returncode != 0:
            raise RuntimeError(
                'scour failed while optimising '
                f'{svg_path.name}: {scour.stderr.strip() or scour.stdout.strip()}'
            )
        return before, svg_path.stat().st_size
    except FileNotFoundError as e:
        raise RuntimeError(
            f'{e.filename} is required for SVG optimisation; install svgo and scour'
        ) from e
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass



def _export_one_svg(
                   all_sources: list[pathlib.Path],
                   selected_src: int,
                   output: pathlib.Path,
                   window: tuple[float, float, float, float],
                   progress_interval: float = 30.0) -> None:
    """Export one layer SVG with gerbv in the shared coordinate frame."""
    ox, oy, w_in, h_in = window
    cmd = [
        'gerbv',
        '--export=svg',
        '--background=#010101','--foreground=#FEFEFE', #Black/white does not work, hence almost...
        f'--output={output}',
        # Pin the frame so every single-file layer shares one viewBox/transform.
        f'--origin={ox:.6f}x{oy:.6f}',
        f'--window_inch={w_in:.6f}x{h_in:.6f}',
        '--border=0',
        str(all_sources[selected_src]),
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
                '    still exporting %s in gerbv (%s elapsed)',
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

def _rasterise_svg_to_png(svg_path: pathlib.Path, png_path: pathlib.Path,
                          window: tuple[float, float, float, float],
                          dpi: int) -> tuple[int, int]:
    """Rasterise an SVG layer to a 1-bit black-background PNG at ``dpi``."""
    _ox, _oy, w_in, h_in = window
    width = max(1, round(w_in * dpi))
    height = max(1, round(h_in * dpi))

    rsvg = subprocess.Popen(
        [
            'rsvg-convert', str(svg_path),
            '-w', str(width),
            '-h', str(height),
            '-b', 'black',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert rsvg.stdout is not None

    convert = subprocess.Popen(
        [
            'convert', 'png:-',
            '-background', 'black',
            '-alpha', 'remove',
            '-alpha', 'off',
            '-threshold', '50%',
            '-type', 'bilevel',
            '-depth', '1',
            str(png_path),
        ],
        stdin=rsvg.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Allow rsvg-convert to receive SIGPIPE if convert exits early.
    rsvg.stdout.close()

    _convert_stdout, convert_stderr = convert.communicate()
    _rsvg_stdout, rsvg_stderr = rsvg.communicate()

    if rsvg.returncode != 0:
        raise subprocess.CalledProcessError(
            rsvg.returncode,
            rsvg.args,
            stderr=rsvg_stderr,
        )

    if convert.returncode != 0:
        raise subprocess.CalledProcessError(
            convert.returncode,
            convert.args,
            stderr=convert_stderr,
        )

    if not png_path.exists():
        raise RuntimeError(
            f'SVG rasterizer claimed success but {png_path} was not written'
        )

    return width, height


def _write_drill_aliases(output_dir: pathlib.Path,
                         written: list[pathlib.Path]) -> None:
    """Create downstream-friendly drill/via aliases from rendered drill masks."""
    pth_path = output_dir / 'PTH.png'
    via_path = output_dir / 'via.png'
    drill_alias = output_dir / 'drill.png'

    if not via_path.exists():
        source = None
        if pth_path.exists():
            source = pth_path
        if source is not None:
            shutil.copy2(source, via_path)
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


def rasterise(
    source_dir: pathlib.Path,
    mapping: dict[str, str],
    output_dir: pathlib.Path,
    dpi: int = 1000,
    layers: Iterable[str] | None = None,
    svg: bool = True,
) -> list[pathlib.Path]:
    """Create aligned SVGs and PNGs for each layer using gerbv plus CairoSVG.

    Output PNGs are named ``{canonical_name}.png`` in ``output_dir``. The
    electrical drill mask is also aliased to ``via.png`` where possible so
    the render step can distinguish physical holes from electrical vertical
    connectors.

    gerbv is only used to emit SVG. Each PNG is rasterised from that SVG, so
    the raster and vector outputs share one geometry path. When ``svg`` is
    false, the intermediate SVGs are removed after PNG creation. The
    white=copper convention matches the SVGs; PNGs get a black background.

    Returns the list of paths written (PNGs and SVGs).
    """
    check_gerbv()
    source_dir = pathlib.Path(source_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = list(layers) if layers else list(mapping.keys())
    targets = [t for t in targets if t in mapping]
    if not targets:
        return []

    # Pass 1: export each file as SVG in one shared origin/window, then
    # rasterise those SVGs to PNG. This avoids small offsets between gerbv
    # backend implementations because gerbv never writes PNG directly.
    written: list[pathlib.Path] = []
    rendered_sizes: dict[str, tuple[int, int]] = {}
    sources = []
    for name in targets:
        fname = mapping[name]
        src = source_dir / fname
        sources.append(src)

    # For SVG we render each layer from its single source file (gerbv's SVG
    # backend flattens all loaded layers into one identical file, so the PNG
    # alpha-select trick does not work). A lone file needs an explicit window
    # or gerbv emits an empty SVG, so probe the shared frame once by exporting
    # one all-loaded SVG and reading its canvas size + transform. Every
    # per-layer render is then pinned to this same origin/window and comes out
    # with an identical viewBox, so the layers register for compositing.
    svg_window: tuple[float, float, float, float] | None = None
    if sources:
        probe = output_dir / '_window_probe.svg'
        try:
            _rasterise_all_svg(sources, probe)
            svg_window = _derive_window(probe)
            log.info('  svg window: origin %.4f,%.4f  %.4f×%.4f in',
                     *svg_window)
        except RuntimeError as e:
            log.error('  ! could not establish SVG window: %s', e)
            return written
        finally:
            if probe.exists():
                probe.unlink()

    for tgt_i,name in enumerate(targets):
        fname = mapping[name]
        src = source_dir / fname
        if not src.is_file():
            log.warning('  warning: %s not found (skipping %s)', src, name)
            continue
        out = output_dir / f'{name}.png'
        svg_out = output_dir / f'{name}.svg'
        started = time.monotonic()
        log.info('  exporting %s.svg from %s', name, fname)
        try:
            assert svg_window is not None
            _export_one_svg(sources, tgt_i, svg_out, svg_window)
        except RuntimeError as e:
            log.error('  ! %s', e)
            continue

        _svg_to_transparent(svg_out,svg_out)

        try:
            before_bytes, after_bytes = _optimise_svg_with_tools(svg_out)
        except RuntimeError as e:
            log.error('  ! SVG optimisation failed: %s', e)
            if not svg:
                svg_out.unlink(missing_ok=True)
            continue
        log.info('    → svg: %.1f→%.1f MB via svgo',
                 before_bytes / 1e6, after_bytes / 1e6)

        log.info('  rasterising %s.png from %s.svg', name, name)
        try:
            _rasterise_svg_to_png(svg_out, out, svg_window, dpi)
        except (ImportError, RuntimeError) as e:
            log.error('  ! SVG→PNG rasterisation failed: %s', e)
            if not svg:
                svg_out.unlink(missing_ok=True)
            continue
        size = Image.open(out).size
        rendered_sizes[name] = size
        log.info('    → %s×%s px (%s)',
                 size[0], size[1], _format_elapsed(time.monotonic() - started))
        written.append(out)
        if svg:
            written.append(svg_out)
        else:
            svg_out.unlink(missing_ok=True)

    # Convenience aliases for downstream use:
    #   via.png   = preferred electrical connectivity mask
    #   drill.png = physical hole mask, falling back to the electrical mask
    #               only when no explicit drill mask was rendered
    _write_drill_aliases(output_dir, written)

    return written
