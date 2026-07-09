"""Tests for the gerber module — detection and layers.json handling.

Rasterisation tests are intentionally skipped: they'd require gerbv on
PATH and real Gerber files, which is awkward for CI. The detection /
mapping / JSON logic is what matters here; rasterisation is a thin
subprocess wrapper.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from pcbnets.gerber import (
    _write_drill_aliases,
    detect_layers,
    load_or_create_layers_json,
    read_layers_json,
    write_layers_json,
)


def touch(p: pathlib.Path) -> None:
    p.write_text('')


def test_svg_tools_run_svgo_then_scour(monkeypatch, tmp_path):
    from pcbnets.gerber import _optimise_svg_with_tools

    svg = tmp_path / 'layer.svg'
    svg.write_text('<svg><!-- verbose --><path d="M0 0"/></svg>')
    calls = []

    def fake_run(cmd, stdout, stderr, text):
        calls.append(cmd)
        if cmd[0] == 'svgo':
            pathlib.Path(cmd[4]).write_text('<svg><path d="M0 0"/></svg>')
        elif cmd[0] == 'scour':
            pathlib.Path(cmd[4]).write_text('<svg><path d="M0 0"/></svg>')
        return type('Proc', (), {'returncode': 0, 'stdout': '', 'stderr': ''})()

    monkeypatch.setattr('pcbnets.gerber.subprocess.run', fake_run)

    before, after = _optimise_svg_with_tools(svg)

    assert before > after
    assert calls == [
        ['svgo', '-i', str(svg), '-o', str(tmp_path / 'layer.svgo.tmp.svg')],
        [
            'scour',
            '-i', str(tmp_path / 'layer.svgo.tmp.svg'),
            '-o', str(svg),
            '--enable-viewboxing',
            '--enable-id-stripping',
            '--enable-comment-stripping',
            '--shorten-ids',
            '--indent=none',
        ],
    ]
    assert not (tmp_path / 'layer.svgo.tmp.svg').exists()


def test_detect_kicad_modern_4_layer(tmp_path):
    """KiCad-style filenames with underscores."""
    for f in ['myboard-F_Cu.gbr', 'myboard-In1_Cu.gbr',
              'myboard-In2_Cu.gbr', 'myboard-B_Cu.gbr',
              'myboard-F_Silkscreen.gbr', 'myboard-B_Silkscreen.gbr',
              'myboard-Edge_Cuts.gbr', 'myboard-PTH.drl']:
        touch(tmp_path / f)
    m = detect_layers(tmp_path)
    assert m['F_Cu'] == 'myboard-F_Cu.gbr'
    assert m['In1_Cu'] == 'myboard-In1_Cu.gbr'
    assert m['In2_Cu'] == 'myboard-In2_Cu.gbr'
    assert m['B_Cu'] == 'myboard-B_Cu.gbr'
    assert m['F_Silkscreen'] == 'myboard-F_Silkscreen.gbr'
    assert m['B_Silkscreen'] == 'myboard-B_Silkscreen.gbr'
    assert m['Edge_Cuts'] == 'myboard-Edge_Cuts.gbr'
    assert m['PTH'] == 'myboard-PTH.drl'


def test_detect_kicad_legacy_dot_naming(tmp_path):
    """Older KiCad used dots: F.Cu, B.Cu."""
    for f in ['proj-F.Cu.gbr', 'proj-B.Cu.gbr']:
        touch(tmp_path / f)
    m = detect_layers(tmp_path)
    assert m['F_Cu'] == 'proj-F.Cu.gbr'
    assert m['B_Cu'] == 'proj-B.Cu.gbr'


def test_detect_altium_style_extensions(tmp_path):
    """Altium-style: layer function in the extension, not the stem."""
    for f in ['board.gtl', 'board.g1', 'board.g2', 'board.gbl',
              'board.gto', 'board.gbo', 'board.gts', 'board.gbs',
              'board.gtp', 'board.gbp', 'board.gko', 'board.drl']:
        touch(tmp_path / f)
    m = detect_layers(tmp_path)
    # Copper layers
    assert m['F_Cu']   == 'board.gtl'
    assert m['In1_Cu'] == 'board.g1'
    assert m['In2_Cu'] == 'board.g2'
    assert m['B_Cu']   == 'board.gbl'
    # Silkscreens
    assert m['F_Silkscreen'] == 'board.gto'
    assert m['B_Silkscreen'] == 'board.gbo'
    # Masks
    assert m['F_Mask'] == 'board.gts'
    assert m['B_Mask'] == 'board.gbs'
    # Paste
    assert m['F_Paste'] == 'board.gtp'
    assert m['B_Paste'] == 'board.gbp'
    # Edge cuts
    assert m['Edge_Cuts'] == 'board.gko'
    # Plain drill is physical; render can fall back to it as via if needed.
    assert m['drill'] == 'board.drl'


def test_detect_altium_ctpci_case(tmp_path):
    """Real-world reproduction: CTPCI board with mixed Altium extensions."""
    for f in ['CTPCI.drl', 'CTPCI.g1', 'CTPCI.g2', 'CTPCI.gbl',
              'CTPCI.gbo', 'CTPCI.gbp', 'CTPCI.gbs', 'CTPCI.gtl',
              'CTPCI.gto', 'CTPCI.gtp', 'CTPCI.gts']:
        touch(tmp_path / f)
    m = detect_layers(tmp_path)
    # Four-layer copper stack must all be picked up
    assert m['F_Cu']   == 'CTPCI.gtl'
    assert m['In1_Cu'] == 'CTPCI.g1'
    assert m['In2_Cu'] == 'CTPCI.g2'
    assert m['B_Cu']   == 'CTPCI.gbl'
    assert m['drill']  == 'CTPCI.drl'
    # Silkscreens
    assert m['F_Silkscreen'] == 'CTPCI.gto'
    assert m['B_Silkscreen'] == 'CTPCI.gbo'


def test_detect_altium_power_plane_extensions(tmp_path):
    """``.gp1``/``.gp2`` power-plane extensions also become inner copper."""
    touch(tmp_path / 'board.gtl')
    touch(tmp_path / 'board.gp1')
    touch(tmp_path / 'board.gp2')
    touch(tmp_path / 'board.gbl')
    m = detect_layers(tmp_path)
    assert m['F_Cu']   == 'board.gtl'
    assert m['In1_Cu'] == 'board.gp1'
    assert m['In2_Cu'] == 'board.gp2'
    assert m['B_Cu']   == 'board.gbl'


def test_kicad_naming_wins_over_extension(tmp_path):
    """If a stem matches a KiCad pattern, that wins over the extension."""
    # ``proj-F_Cu.gbl`` is weird (KiCad name with bottom-copper ext) but
    # the filename pattern is more specific than the extension fallback.
    touch(tmp_path / 'proj-F_Cu.gbl')
    m = detect_layers(tmp_path)
    assert m == {'F_Cu': 'proj-F_Cu.gbl'}


def test_detect_plain_drl_treated_as_physical_drill(tmp_path):
    """A bare ``project.drl`` maps to the physical drill mask."""
    touch(tmp_path / 'myboard.drl')
    m = detect_layers(tmp_path)
    assert m['drill'] == 'myboard.drl'


def test_pth_preferred_for_via_alias(tmp_path):
    """PTH remains the preferred source for via.png when present."""
    pth = tmp_path / 'PTH.png'
    drill = tmp_path / 'drill.png'
    pth.write_bytes(b'pth')
    drill.write_bytes(b'drill')
    written = [pth, drill]

    _write_drill_aliases(tmp_path, written)

    assert (tmp_path / 'via.png').read_bytes() == b'pth'


def test_detect_via_and_pth_drills(tmp_path):
    touch(tmp_path / 'myboard-Via.drl')
    touch(tmp_path / 'myboard-PTH.drl')
    touch(tmp_path / 'myboard-NPTH.drl')
    m = detect_layers(tmp_path)
    assert m['via'] == 'myboard-Via.drl'
    assert m['PTH'] == 'myboard-PTH.drl'
    assert m['NPTH'] == 'myboard-NPTH.drl'


def test_detect_pth_and_npth_separate(tmp_path):
    touch(tmp_path / 'board-PTH.drl')
    touch(tmp_path / 'board-NPTH.drl')
    m = detect_layers(tmp_path)
    assert m['PTH'] == 'board-PTH.drl'
    assert m['NPTH'] == 'board-NPTH.drl'


def test_detect_skips_non_gerber_files(tmp_path):
    touch(tmp_path / 'README.md')
    touch(tmp_path / 'myboard.pdf')
    touch(tmp_path / 'myboard-F_Cu.gbr')
    m = detect_layers(tmp_path)
    assert list(m) == ['F_Cu']


def test_layers_json_roundtrip(tmp_path):
    mapping = {
        'F_Cu':     'a.gbr',
        'In1_Cu':   'b.gbr',
        'B_Cu':     'c.gbr',
        'PTH':      'd.drl',
    }
    write_layers_json(tmp_path, mapping)
    path = tmp_path / 'layers.json'
    assert path.exists()
    loaded = read_layers_json(path)
    assert loaded == mapping


def test_layers_json_ordering(tmp_path):
    """Output JSON should be in canonical layer order: F → inner → B → drill."""
    mapping = {
        'B_Cu':         'b.gbr',
        'PTH':          'd.drl',
        'F_Cu':         'a.gbr',
        'In2_Cu':       'in2.gbr',
        'In1_Cu':       'in1.gbr',
        'F_Silkscreen': 's.gbr',
    }
    write_layers_json(tmp_path, mapping)
    data = json.loads((tmp_path / 'layers.json').read_text())
    keys = list(data['layers'])
    # F_Cu first
    assert keys[0] == 'F_Cu'
    # Inner layers in numeric order
    assert keys.index('In1_Cu') < keys.index('In2_Cu')
    # B_Cu after inner
    assert keys.index('B_Cu') > keys.index('In2_Cu')
    # Drill after copper
    assert keys.index('PTH') > keys.index('B_Cu')


def test_layers_json_accepts_bare_flat_form(tmp_path):
    """Hand-edited flat-form JSON without the "layers" wrapper should load."""
    bare = {
        'F_Cu':   'a.gbr',
        'B_Cu':   'b.gbr',
        '_comment': 'hi',
    }
    (tmp_path / 'layers.json').write_text(json.dumps(bare))
    loaded = read_layers_json(tmp_path / 'layers.json')
    assert loaded == {'F_Cu': 'a.gbr', 'B_Cu': 'b.gbr'}
    assert '_comment' not in loaded


def test_load_or_create_creates_file(tmp_path):
    touch(tmp_path / 'proj-F_Cu.gbr')
    touch(tmp_path / 'proj-B_Cu.gbr')
    mapping, created = load_or_create_layers_json(tmp_path)
    assert created is True
    assert (tmp_path / 'layers.json').exists()
    assert mapping == {
        'F_Cu': 'proj-F_Cu.gbr',
        'B_Cu': 'proj-B_Cu.gbr',
    }


def test_load_or_create_uses_existing(tmp_path):
    """Existing layers.json wins over fresh auto-detect."""
    touch(tmp_path / 'proj-F_Cu.gbr')
    touch(tmp_path / 'proj-B_Cu.gbr')
    # Pre-write a different mapping (user has edited).
    custom = {'F_Cu': 'override.gbr', 'B_Cu': 'whatever.gbr'}
    write_layers_json(tmp_path, custom)
    mapping, created = load_or_create_layers_json(tmp_path)
    assert created is False
    assert mapping == custom


def test_load_or_create_empty_dir(tmp_path):
    """Empty directory: returns empty mapping but still writes the file."""
    mapping, created = load_or_create_layers_json(tmp_path)
    assert created is True
    assert mapping == {}
    assert (tmp_path / 'layers.json').exists()


def test_inner_layer_numbers_preserved(tmp_path):
    for n in [1, 3, 7]:  # non-contiguous
        touch(tmp_path / f'proj-In{n}_Cu.gbr')
    m = detect_layers(tmp_path)
    assert m['In1_Cu'].endswith('In1_Cu.gbr')
    assert m['In3_Cu'].endswith('In3_Cu.gbr')
    assert m['In7_Cu'].endswith('In7_Cu.gbr')
    assert 'In2_Cu' not in m


def test_unknown_files_dont_break_detection(tmp_path):
    """Files that don't match any pattern just get ignored."""
    touch(tmp_path / 'random_file.gbr')      # gerber ext but no pattern match
    touch(tmp_path / 'proj-F_Cu.gbr')
    m = detect_layers(tmp_path)
    assert m == {'F_Cu': 'proj-F_Cu.gbr'}
