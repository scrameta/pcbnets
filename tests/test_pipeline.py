"""Smoke tests for the core pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pcbnets import (
    UnionFind,
    build_grid_and_idmap,
    explain_merge_path,
    extract_nets,
    load_masks,
    merge_nets,
    merge_nets_debug,
    threshold_mask,
)


def make_mask(w=200, h=100, shapes=()):
    img = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(img)
    for shape in shapes:
        kind, args = shape[0], shape[1:]
        if kind == 'rect':
            d.rectangle(args, fill=255)
        elif kind == 'ellipse':
            d.ellipse(args, fill=255)
    return img


def test_threshold_mask_no_dither():
    img = Image.new('L', (10, 10), 100)  # mid-grey
    m = threshold_mask(img, threshold=50)
    assert m.mode == '1'
    assert all(p == 255 for p in m.convert('L').getdata())


def test_threshold_mask_below():
    img = Image.new('L', (10, 10), 30)
    m = threshold_mask(img, threshold=50)
    assert all(p == 0 for p in m.convert('L').getdata())


def test_union_find_basic():
    uf = UnionFind()
    uf.union('a', 'b')
    uf.union('b', 'c')
    uf.union('d', 'e')
    assert uf.find('a') == uf.find('c')
    assert uf.find('a') != uf.find('d')
    assert uf.find('e') == uf.find('d')


def test_union_find_path_compression():
    uf = UnionFind()
    for i in range(100):
        uf.union(i, i + 1)
    # After find, depths should be small.
    root = uf.find(0)
    for i in range(101):
        assert uf.find(i) == root


def test_extract_nets_two_pads_one_drill():
    """Two isolated pads on different layers, one drill bridging them.
    Should produce one electrical net."""
    layers = {
        'top': make_mask(shapes=[('rect', 10, 10, 40, 40)]),
        'bot': make_mask(shapes=[('rect', 100, 50, 130, 80)]),
    }
    # Two drill regions, one over each pad
    drill = make_mask(shapes=[
        ('ellipse', 20, 20, 30, 30),     # over top pad only
        ('ellipse', 110, 60, 120, 70),   # over bot pad only
    ])
    result = extract_nets(layers, drill, drill_grow_px=0)
    # Each drill independently touches only one layer's component
    touches = result['drill_touches']
    assert len(touches) == 2
    # Build the union-find merge
    net_labels = merge_nets(touches, result['layer_labels'])
    # Two separate nets, not connected
    top_id = int(net_labels['top'].max())
    bot_id = int(net_labels['bot'].max())
    assert top_id != bot_id


def test_extract_nets_drill_bridges_layers():
    """One drill straddling two overlapping pads on different layers → one net."""
    layers = {
        'top': make_mask(shapes=[('rect', 50, 30, 90, 70)]),
        'bot': make_mask(shapes=[('rect', 60, 40, 100, 80)]),
    }
    # Drill in the overlap region
    drill = make_mask(shapes=[('ellipse', 65, 45, 85, 65)])
    result = extract_nets(layers, drill, drill_grow_px=0)
    assert len(result['drill_touches']) == 1
    members = next(iter(result['drill_touches'].values()))
    assert ('top', 1) in members
    assert ('bot', 1) in members

    net_labels = merge_nets(result['drill_touches'], result['layer_labels'])
    # Both layers' pads should now have the same net id
    top_net = net_labels['top'][45, 70]
    bot_net = net_labels['bot'][50, 80]
    assert top_net == bot_net != 0


def test_inferred_generic_drill_ignores_holes_without_copper():
    """Generic drill fallback treats holes with no copper pads as NPTH."""
    layers = {
        'top': make_mask(w=80, h=80),
        'bot': make_mask(w=80, h=80),
    }
    drill = make_mask(w=80, h=80, shapes=[('ellipse', 35, 35, 45, 45)])

    result = extract_nets(layers, drill, connector_mode='infer')

    assert result['drill_touches'] == {}
    assert result['drill_classifications'][0]['plated'] is False
    assert result['drill_classifications'][0]['classification'] == 'likely_npth'
    assert 'no substantial copper pad ring' in result['drill_classifications'][0]['reason']


def test_inferred_generic_drill_rejects_trace_touches_without_complete_pad_ring():
    """Generic drill fallback ignores traces that touch without forming pads."""
    layers = {
        'top': make_mask(w=80, h=80, shapes=[('rect', 40, 35, 70, 45)]),
        'bot': make_mask(w=80, h=80, shapes=[('rect', 10, 35, 40, 45)]),
    }
    drill = make_mask(w=80, h=80, shapes=[('ellipse', 35, 35, 45, 45)])

    result = extract_nets(layers, drill, connector_mode='infer')

    assert result['drill_touches'] == {}
    assert result['drill_classifications'][0]['plated'] is False
    assert result['drill_classifications'][0]['classification'] == 'likely_npth'
    contacts = result['drill_classifications'][0]['contacts']
    assert any(c['contact'] == 'partial' for c in contacts)
    assert not any(c['pad_ring_contact'] for c in contacts)


def test_inferred_generic_drill_rejects_one_sided_high_pixel_ring():
    from pcbnets.nets import _classify_drill

    contacts = [
        {
            'layer': 'top',
            'component_ids': [1],
            'contact': 'partial',
            'pad_ring_fraction': 0.90,
            'pad_ring_angular_coverage': 0.30,
        },
        {
            'layer': 'bot',
            'component_ids': [1],
            'contact': 'partial',
            'pad_ring_fraction': 0.90,
            'pad_ring_angular_coverage': 0.30,
        },
    ]

    decision = _classify_drill(
        drill_id=1,
        contacts=contacts,
        layer_names=['top', 'bot'],
        radius_px=5.0,
        small_radius_px=8.0,
        large_radius_px=20.0,
        connector_mode='infer',
    )

    assert decision['plated'] is False
    assert decision['classification'] == 'likely_npth'


def test_inferred_generic_drill_tolerates_rasterised_pad_ring_threshold():
    from pcbnets.nets import _classify_drill

    contacts = [
        {
            'layer': 'top',
            'component_ids': [1],
            'contact': 'all_around',
            'pad_ring_fraction': 0.84,
        },
        {
            'layer': 'bot',
            'component_ids': [1],
            'contact': 'all_around',
            'pad_ring_fraction': 0.84,
        },
    ]

    decision = _classify_drill(
        drill_id=1,
        contacts=contacts,
        layer_names=['top', 'bot'],
        radius_px=5.0,
        small_radius_px=8.0,
        large_radius_px=20.0,
        connector_mode='infer',
    )

    assert decision['plated'] is True
    assert decision['classification'] == 'likely_pth'


def test_inferred_generic_drill_connects_top_and_bottom_annular_pads():
    """Generic drill fallback treats copper on both outer sides as likely PTH."""
    layers = {
        'top': make_mask(w=80, h=80, shapes=[('rect', 0, 0, 79, 79)]),
        'bot': make_mask(w=80, h=80, shapes=[('rect', 0, 0, 79, 79)]),
    }
    drill = make_mask(w=80, h=80, shapes=[('ellipse', 35, 35, 45, 45)])

    result = extract_nets(layers, drill, connector_mode='infer')

    assert len(result['drill_touches']) == 1
    assert result['drill_classifications'][0]['plated'] is True
    assert result['drill_classifications'][0]['reason'] == (
        'copper annulus/pad contact found on both outer copper layers'
    )


def test_explicit_pth_connects_all_around_copper():
    """Explicit PTH/via masks connect even when annular contact is all around."""
    layers = {
        'top': make_mask(w=80, h=80, shapes=[('rect', 0, 0, 79, 79)]),
        'bot': make_mask(w=80, h=80, shapes=[('rect', 0, 0, 79, 79)]),
    }
    drill = make_mask(w=80, h=80, shapes=[('ellipse', 35, 35, 45, 45)])

    result = extract_nets(layers, drill, connector_mode='explicit')

    assert len(result['drill_touches']) == 1
    assert result['drill_classifications'][0]['classification'] == 'explicit_pth'
    members = next(iter(result['drill_touches'].values()))
    assert ('top', 1) in members
    assert ('bot', 1) in members


def test_merge_nets_debug_explains_drill_path():
    """Debug metadata should explain the drill edge that merged local nets."""
    layers = {
        'top': make_mask(shapes=[('rect', 50, 30, 90, 70)]),
        'bot': make_mask(shapes=[('rect', 60, 40, 100, 80)]),
    }
    drill = make_mask(shapes=[('ellipse', 65, 45, 85, 65)])
    result = extract_nets(layers, drill, drill_grow_px=0)

    net_labels, debug = merge_nets_debug(
        result['drill_touches'],
        result['layer_labels'],
        result['drill_labels'],
        result['drill_classifications'],
    )
    assert net_labels['top'][45, 70] == net_labels['bot'][50, 80] != 0
    assert debug['drills'][0]['bbox'] == [65, 45, 86, 66]
    assert debug['drills'][0]['centroid'] == [75.0, 55.0]
    assert debug['drill_classifications'][0]['classification'] == 'explicit_pth'

    path = explain_merge_path(debug, ('top', 1), ('bot', 1))
    assert path == [{
        'from': {'layer': 'top', 'component': 1},
        'drill': 1,
        'to': {'layer': 'bot', 'component': 1},
    }]


def test_isolated_net_gets_an_id():
    """A copper region with no drills should still get a unique net id."""
    layers = {
        'top': make_mask(shapes=[
            ('rect', 10, 10, 30, 30),    # isolated, no drill
            ('rect', 100, 50, 120, 70),  # also isolated
        ]),
    }
    drill = make_mask()  # empty
    result = extract_nets(layers, drill, drill_grow_px=0)
    net_labels = merge_nets(result['drill_touches'], result['layer_labels'])
    # Two distinct nets on top
    unique = set(np.unique(net_labels['top'])) - {0}
    assert len(unique) == 2


def test_mask_shape_mismatch_raises(tmp_path):
    Image.new('L', (100, 50), 255).save(tmp_path / 'top.png')
    Image.new('L', (120, 50), 255).save(tmp_path / 'drill.png')
    with pytest.raises(ValueError, match='dimensions differ'):
        load_masks(tmp_path, ['top'], 'drill')


def test_build_grid_basic():
    layers = {'top': make_mask(), 'bot': make_mask()}
    fake_labels = {
        'top': np.zeros((100, 200), dtype=np.int32),
        'bot': np.zeros((100, 200), dtype=np.int32),
    }
    fake_labels['top'][20:40, 20:40] = 1
    fake_labels['bot'][20:40, 20:40] = 1

    grid, idmap, meta = build_grid_and_idmap(layers, fake_labels, cols=2)
    assert grid.size == (400, 100)  # 2 cols × 200, 1 row × 100
    assert idmap.size == grid.size
    assert meta['layers'] == ['top', 'bot']
    assert 'placements' in meta


def test_display_punches_drill_holes_out_of_copper_and_idmap():
    from pcbnets.cli import _punch_drill_holes_for_display

    arrs = {
        'top': np.ones((8, 8), dtype=bool),
    }
    labels = {
        'top': np.ones((8, 8), dtype=np.int32),
    }
    drill = np.zeros((8, 8), dtype=bool)
    drill[3:5, 3:5] = True

    display_images, display_labels = _punch_drill_holes_for_display(
        arrs, labels, drill, ['top']
    )

    display = np.asarray(display_images['top'].convert('L')) > 0
    assert display[2, 2]
    assert not display[3, 3]
    assert display_labels['top'][2, 2] == 1
    assert display_labels['top'][3, 3] == 0


def test_load_net_labels_for_render_accepts_nets_output(tmp_path):
    from pcbnets.cli import _load_net_labels_for_render

    arrs = {
        'top': np.zeros((3, 4), dtype=bool),
        'bot': np.zeros((3, 4), dtype=bool),
    }
    top = np.zeros((3, 4), dtype=np.int32)
    bot = np.ones((3, 4), dtype=np.int32)
    np.savez_compressed(tmp_path / 'net-labels.npz', top=top, bot=bot)

    loaded = _load_net_labels_for_render(tmp_path, ['top', 'bot'], arrs)

    assert list(loaded) == ['top', 'bot']
    np.testing.assert_array_equal(loaded['top'], top)
    np.testing.assert_array_equal(loaded['bot'], bot)


def test_load_net_labels_for_render_rejects_shape_mismatch(tmp_path):
    from pcbnets.cli import _load_net_labels_for_render

    arrs = {'top': np.zeros((3, 4), dtype=bool)}
    np.savez_compressed(tmp_path / 'net-labels.npz',
                        top=np.zeros((2, 4), dtype=np.int32))

    with pytest.raises(ValueError, match='expected'):
        _load_net_labels_for_render(tmp_path, ['top'], arrs)


def test_drill_identify_writes_split_masks_and_choices(tmp_path):
    import argparse
    import json
    from pcbnets.cli import cmd_drill_identify

    src = tmp_path / 'src'
    out = tmp_path / 'out'
    src.mkdir()
    make_mask(w=80, h=80, shapes=[
        ('ellipse', 12, 12, 28, 28),
    ]).save(src / 'F_Cu.png')
    make_mask(w=80, h=80, shapes=[
        ('ellipse', 12, 12, 28, 28),
    ]).save(src / 'B_Cu.png')
    make_mask(w=80, h=80, shapes=[
        ('ellipse', 15, 15, 25, 25),
        ('ellipse', 55, 55, 65, 65),
    ]).save(src / 'drill.png')

    args = argparse.Namespace(
        directory=str(src),
        output=str(out),
        choices=None,
        excellon=None,
        layers=None,
        drill='auto',
        threshold=0,
        dpi=1000,
        no_auto_invert=False,
        auto_align=False,
        invert=[],
        no_invert=[],
        offset=[],
        outer=None,
    )

    assert cmd_drill_identify(args) == 0

    manifest = json.loads((out / 'drill-identify.json').read_text())
    assert manifest['plated_count'] == 1
    assert manifest['npth_count'] == 1
    assert (out / 'PTH.png').exists()
    assert (out / 'via.png').exists()
    assert (out / 'NPTH.png').exists()

    pth = np.asarray(Image.open(out / 'PTH.png').convert('L')) > 0
    npth = np.asarray(Image.open(out / 'NPTH.png').convert('L')) > 0
    assert pth[20, 20]
    assert not pth[60, 60]
    assert npth[60, 60]


def test_drill_identify_choices_override_masks(tmp_path):
    from pcbnets.cli import _split_drill_masks

    labels = np.array([[0, 1, 2]], dtype=np.int32)
    classifications = [
        {'drill': 1, 'plated': True, 'classification': 'likely_pth'},
        {'drill': 2, 'plated': False, 'classification': 'likely_npth'},
    ]

    pth, npth, decisions = _split_drill_masks(
        labels,
        classifications,
        overrides={1: False, 2: True},
    )

    assert not pth[0, 1]
    assert pth[0, 2]
    assert npth[0, 1]
    assert decisions[0]['override'] is True
    assert decisions[1]['override'] is True



def test_excellon_mapping_uses_centroids_not_drill_id_order():
    from pcbnets.cli import _map_excellon_objects_to_drills

    class Obj:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    objects = [
        Obj(0, 0),    # bottom-left in Excellon space
        Obj(0, 100),  # top-left in Excellon space
    ]
    classifications = [
        {'drill': 1, 'plated': True, 'centroid': [0, 100]},
        {'drill': 2, 'plated': False, 'centroid': [0, 0]},
    ]

    assert _map_excellon_objects_to_drills(objects, classifications) == {
        1: 0,
        2: 1,
    }


def test_drill_identify_excellon_without_choices_writes_excellon_splits(tmp_path, monkeypatch):
    import argparse
    import json
    import sys
    import types
    from pcbnets.cli import cmd_drill_identify

    class FakeObj:
        def __init__(self, name, x, y):
            self.name = name
            self.x = x
            self.y = y

    class FakeExcellon:
        def __init__(self, objects):
            self.objects = list(objects)

        @classmethod
        def open(cls, path):
            return cls([FakeObj('plated-hole', 10, 30)])

        def save(self, path):
            path.write_text('\n'.join(obj.name for obj in self.objects))

    monkeypatch.setitem(sys.modules, 'gerbonara', types.SimpleNamespace(ExcellonFile=FakeExcellon))

    src = tmp_path / 'src'
    out = tmp_path / 'out'
    src.mkdir()
    make_mask(w=40, h=40, shapes=[
        ('ellipse', 7, 7, 23, 23),
    ]).save(src / 'F_Cu.png')
    make_mask(w=40, h=40, shapes=[
        ('ellipse', 7, 7, 23, 23),
    ]).save(src / 'B_Cu.png')
    make_mask(w=40, h=40, shapes=[
        ('ellipse', 10, 10, 20, 20),
    ]).save(src / 'drill.png')

    args = argparse.Namespace(
        directory=str(src),
        output=str(out),
        choices=None,
        excellon=str(tmp_path / 'all_drills.drl'),
        layers=None,
        drill='auto',
        threshold=0,
        dpi=1000,
        no_auto_invert=False,
        auto_align=False,
        invert=[],
        no_invert=[],
        offset=[],
        outer=None,
    )

    assert cmd_drill_identify(args) == 0

    manifest = json.loads((out / 'drill-identify.json').read_text())
    assert manifest['plated_count'] == 1
    assert (out / 'PTH.png').exists()
    assert (out / 'NPTH.png').exists()
    assert (out / 'PTH.drl').read_text() == 'plated-hole'
    assert (out / 'NPTH.drl').read_text() == ''
    assert manifest['drill_classifications'][0]['source_object_index'] == 0
    assert manifest['drill_classifications'][0]['source_x'] == 10
    assert manifest['drill_classifications'][0]['source_y'] == 30


def test_split_excellon_with_gerbonara_uses_object_index_choices(tmp_path, monkeypatch):
    import json
    import sys
    import types
    from pcbnets.cli import _split_excellon_with_gerbonara

    class FakeExcellon:
        def __init__(self, objects):
            self.objects = list(objects)

        @classmethod
        def open(cls, path):
            return cls(['hole-0', 'hole-1', 'hole-2'])

        def save(self, path):
            path.write_text('\n'.join(self.objects))

    monkeypatch.setitem(sys.modules, 'gerbonara', types.SimpleNamespace(ExcellonFile=FakeExcellon))

    choices = tmp_path / 'choices.json'
    choices.write_text(json.dumps({
        'drill_classifications': [
            {'object_index': 0, 'plated': True},
            {'object_index': 1, 'plated': False},
            {'object_index': 2, 'plated': True},
        ],
    }))

    pth_path, npth_path = _split_excellon_with_gerbonara(
        tmp_path / 'all_drills.drl',
        tmp_path / 'out',
        choices,
    )

    assert pth_path.read_text() == 'hole-0\nhole-2'
    assert npth_path.read_text() == 'hole-1'


def test_split_excellon_with_gerbonara_requires_object_indexes(tmp_path):
    import json
    from pcbnets.cli import _split_excellon_with_gerbonara

    choices = tmp_path / 'choices.json'
    choices.write_text(json.dumps({
        'drill_classifications': [
            {'drill': 1, 'plated': True},
        ],
    }))

    with pytest.raises(ValueError, match='object_index'):
        _split_excellon_with_gerbonara(
            tmp_path / 'all_drills.drl',
            tmp_path / 'out',
            choices,
        )


def test_id_encoding_roundtrip():
    """Net IDs encoded into RGB should decode back exactly."""
    labels = {'top': np.array([[1, 2, 300], [65536, 16777215, 0]], dtype=np.int32)}
    layers = {'top': Image.new('L', (3, 2), 0)}
    _, idmap, _ = build_grid_and_idmap(layers, labels, cols=1)
    arr = np.asarray(idmap.convert('RGB')).astype(np.uint32)
    decoded = arr[..., 0] | (arr[..., 1] << 8) | (arr[..., 2] << 16)
    np.testing.assert_array_equal(decoded, labels['top'].astype(np.uint32))


def _decode_idmap_labels(img):
    arr = np.asarray(img.convert('RGB')).astype(np.uint32)
    return arr[..., 0] | (arr[..., 1] << 8) | (arr[..., 2] << 16)


def test_scaled_idmap_prefers_trace_ids_over_blank_space():
    labels = {'top': np.array([[0, 0], [0, 42]], dtype=np.int32)}
    layers = {'top': Image.new('L', (2, 2), 0)}

    _, idmap, _ = build_grid_and_idmap(layers, labels, cols=1, scale=0.5)

    decoded = _decode_idmap_labels(idmap)
    assert decoded.shape == (1, 1)
    assert decoded[0, 0] == 42


def test_mip_idmap_prefers_trace_ids_over_blank_space(tmp_path):
    from pcbnets.mips import make_mips

    build = tmp_path / 'build'
    build.mkdir()
    Image.new('L', (2, 2), 0).save(build / 'grid.png')
    _, idmap, _ = build_grid_and_idmap(
        {'top': Image.new('L', (2, 2), 0)},
        {'top': np.array([[0, 0], [0, 7]], dtype=np.int32)},
        cols=1,
    )
    idmap.save(build / 'idmap.png')

    make_mips(build, levels=(2,))

    decoded = _decode_idmap_labels(Image.open(build / 'mips' / '2' / 'idmap.png'))
    assert decoded.shape == (1, 1)
    assert decoded[0, 0] == 7

def test_write_build_is_svg_only_and_writes_netmap(tmp_path):
    from pcbnets.cli import _write_build

    build = tmp_path / 'build'
    grid = Image.new('L', (32, 32), 255)
    idmap = Image.new('RGB', (32, 32), (0, 0, 0))
    meta = {'grid_w': 32, 'grid_h': 32, 'layers': ['F_Cu']}

    _write_build(build, grid, idmap, meta, {}, '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><path class="net-shape" data-net-id="1" data-layer="F_Cu" d="M0 0H1V1H0Z"/></svg>')

    assert (build / 'meta.json').is_file()
    assert (build / 'netmap.svg').is_file()
    assert not (build / 'grid.png').exists()
    assert not (build / 'idmap.png').exists()
    assert not (build / 'mips').exists()
    assert not (build / 'tiles').exists()
    assert 'data-net-id="1"' in (build / 'netmap.svg').read_text()


def test_labels_to_netmap_svg_groups_net_geometry_by_layer():
    from pcbnets.render import labels_to_netmap_svg

    labels = {
        'F_Cu': np.array([[1, 0, 1], [0, 2, 2]], dtype=np.int32),
        'B_Cu': np.array([[1, 1, 0], [0, 0, 0]], dtype=np.int32),
    }
    svg = labels_to_netmap_svg(labels, {
        'F_Cu': {'x': 0, 'y': 0, 'w': 3, 'h': 2},
        'B_Cu': {'x': 3, 'y': 0, 'w': 3, 'h': 2},
    }, 6, 2)

    assert 'viewBox="0 0 6 2"' in svg
    assert 'data-layer="F_Cu"' in svg
    assert 'data-layer="B_Cu"' in svg
    assert svg.count('data-net-id="1"') == 2
    assert 'class="net-shape"' in svg


def test_copy_visual_svgs_preserves_source_text(tmp_path):
    from pcbnets.cli import _copy_visual_svgs

    src_dir = tmp_path / 'src'
    build_dir = tmp_path / 'build'
    src_dir.mkdir()
    build_dir.mkdir()
    svg = (
        '<svg width="1303" height="885" viewBox="0 0 1303 885">\n'
        '  <!-- keep raw -->\n'
        '</svg>\n'
    )
    (src_dir / 'F_Cu.svg').write_text(svg)

    _copy_visual_svgs(src_dir, build_dir, ['F_Cu'])

    assert (build_dir / 'F_Cu.svg').read_text() == svg


def test_export_copies_svg_only_bundle(tmp_path):
    from pcbnets.cli import cmd_export
    import argparse
    import json

    build = tmp_path / 'build'
    build.mkdir()
    (build / 'netmap.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>')
    (build / 'F_Cu.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>')
    (build / 'meta.json').write_text(json.dumps({'layer_svgs': {'F_Cu': 'F_Cu.svg'}, 'netmap': 'netmap.svg'}))

    out = tmp_path / 'static'
    args = argparse.Namespace(
        build_dir=str(build), output=str(out), title=None, description=None,
    )
    assert cmd_export(args) == 0
    assert (out / 'netmap.svg').is_file()
    assert (out / 'F_Cu.svg').is_file()
    assert not (out / 'mips').exists()


def test_deploy_zip_contains_static_bundle(tmp_path):
    from zipfile import ZipFile
    from pcbnets.cli import cmd_deploy
    import argparse
    import json

    build = tmp_path / 'build'
    build.mkdir()
    (build / 'netmap.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>')
    (build / 'F_Cu.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>')
    (build / 'F_Silkscreen.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>')
    (build / 'F_Mask.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>')
    (build / 'meta.json').write_text(json.dumps({
        'layers': ['F_Cu'],
        'silk_layers': ['F_Silkscreen'],
        'mask_layers': ['F_Mask'],
        'layer_svgs': {'F_Cu': 'F_Cu.svg', 'F_Silkscreen': 'F_Silkscreen.svg', 'F_Mask': 'F_Mask.svg'},
        'netmap': 'netmap.svg',
    }))

    out = tmp_path / 'board.zip'
    args = argparse.Namespace(
        build_dir=str(build), output=str(out), title='Test Board',
        description=None, prefix=None,
    )
    assert cmd_deploy(args) == 0

    with ZipFile(out) as zf:
        names = set(zf.namelist())
        assert {'index.html', 'netmap.svg', 'meta.json'} <= names
        assert 'F_Cu.svg' in names
        assert 'F_Silkscreen.svg' in names
        assert 'F_Mask.svg' in names
        assert not any(name.endswith('.png') or name.startswith('mips/') or name.startswith('tiles/') for name in names)
        assert 'Test Board' in zf.read('index.html').decode('utf-8')


def test_deploy_zip_prefix(tmp_path):
    from zipfile import ZipFile
    from pcbnets.cli import cmd_deploy
    import argparse

    build = tmp_path / 'build'
    build.mkdir()
    (build / 'netmap.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>')
    (build / 'meta.json').write_text('{"netmap": "netmap.svg", "layer_svgs": {}}')

    out = tmp_path / 'board.zip'
    args = argparse.Namespace(
        build_dir=str(build), output=str(out), title=None,
        description=None, prefix='boards/test',
    )
    assert cmd_deploy(args) == 0

    with ZipFile(out) as zf:
        names = set(zf.namelist())
        assert 'boards/test/index.html' in names
        assert 'boards/test/netmap.svg' in names
        assert not any(name.endswith('.png') for name in names)
