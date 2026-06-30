from __future__ import annotations

from PIL import Image

from pcbnets.cli import _resolve_drill_name, _resolve_layers, _resolve_via_name


def png(path):
    Image.new('1', (2, 2), 0).save(path)


def test_resolve_layers_keeps_full_kicad_6_layer_stack(tmp_path):
    for name in ['F_Cu', 'In1_Cu', 'In2_Cu', 'In3_Cu', 'In4_Cu', 'B_Cu',
                 'PTH', 'NPTH', 'F_Mask', 'B_Mask', 'F_Silkscreen']:
        png(tmp_path / f'{name}.png')

    assert _resolve_layers(tmp_path, None, 'PTH', 'PTH') == [
        'F_Cu', 'In1_Cu', 'In2_Cu', 'In3_Cu', 'In4_Cu', 'B_Cu'
    ]


def test_resolve_layers_does_not_auto_select_legacy_names(tmp_path):
    for name in ['top', 'inner1', 'inner2', 'inner3', 'inner4', 'bottom', 'drill']:
        png(tmp_path / f'{name}.png')

    assert _resolve_layers(tmp_path, None, 'drill', 'drill') == []


def test_resolve_drill_prefers_physical_drill_alias(tmp_path):
    png(tmp_path / 'drill.png')
    png(tmp_path / 'PTH.png')
    png(tmp_path / 'NPTH.png')
    assert _resolve_drill_name(tmp_path, 'auto') == 'drill'
    assert _resolve_drill_name(tmp_path, 'PTH') == 'PTH'


def test_resolve_drill_falls_back_to_pth_when_no_drill_alias(tmp_path):
    png(tmp_path / 'PTH.png')
    png(tmp_path / 'NPTH.png')
    assert _resolve_drill_name(tmp_path, 'auto') == 'PTH'


def test_resolve_via_prefers_via_then_pth_then_drill(tmp_path):
    png(tmp_path / 'drill.png')
    assert _resolve_via_name(tmp_path, 'auto') == 'drill'
    png(tmp_path / 'PTH.png')
    assert _resolve_via_name(tmp_path, 'auto') == 'PTH'
    png(tmp_path / 'via.png')
    assert _resolve_via_name(tmp_path, 'auto') == 'via'



def test_align_manual_offset_writes_shifted_png_and_manifest(tmp_path):
    import argparse
    import json
    import numpy as np
    from pcbnets.cli import cmd_align

    src = tmp_path / 'pngs'
    dst = tmp_path / 'aligned'
    src.mkdir()
    # Minimal canonical stack + separate physical/electrical drill names.
    for name in ['F_Cu', 'B_Cu', 'drill']:
        Image.new('1', (12, 12), 0).save(src / f'{name}.png')
    pth = Image.new('1', (12, 12), 0)
    pix = pth.load()
    pix[3, 4] = 255
    pth.save(src / 'PTH.png')

    args = argparse.Namespace(
        directory=str(src), output=str(dst), layers=None,
        drill='auto', via='auto', threshold=0,
        auto_via=False, auto_masks=False,
        no_auto_invert=False, auto_align=False,
        invert=[], no_invert=[], offset=[['PTH', '2,3']], outer=None,
    )
    assert cmd_align(args) == 0

    arr = np.asarray(Image.open(dst / 'PTH.png').convert('L')) > 0
    assert arr[6, 6]  # original x=3,y=4 shifted by dx=3,dy=2
    assert not arr[4, 3]

    manifest = json.loads((dst / 'alignment.json').read_text())
    assert manifest['via_name'] == 'PTH'
    assert manifest['shifts']['PTH']['offset'] == [2, 3]
    assert manifest['shifts']['PTH']['source'] == 'manual'


def test_render_auto_align_flag_defaults_off():
    from pcbnets.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(['render', 'pngs'])
    assert args.auto_align is False
    args = parser.parse_args(['render', 'pngs', '--auto-align'])
    assert args.auto_align is True
