"""Smoke tests for the core pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pcbnets import (
    UnionFind,
    build_grid_and_idmap,
    extract_nets,
    load_masks,
    merge_nets,
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


def test_id_encoding_roundtrip():
    """Net IDs encoded into RGB should decode back exactly."""
    labels = {'top': np.array([[1, 2, 300], [65536, 16777215, 0]], dtype=np.int32)}
    layers = {'top': Image.new('L', (3, 2), 0)}
    _, idmap, _ = build_grid_and_idmap(layers, labels, cols=1)
    arr = np.asarray(idmap.convert('RGB')).astype(np.uint32)
    decoded = arr[..., 0] | (arr[..., 1] << 8) | (arr[..., 2] << 16)
    np.testing.assert_array_equal(decoded, labels['top'].astype(np.uint32))


def test_deploy_zip_contains_static_bundle(tmp_path):
    from zipfile import ZipFile
    from pcbnets.cli import cmd_deploy
    import argparse
    import json

    build = tmp_path / 'build'
    build.mkdir()
    Image.new('L', (2, 2), 0).save(build / 'grid.png')
    Image.new('RGB', (2, 2), 0).save(build / 'idmap.png')
    (build / 'meta.json').write_text(json.dumps({
        'layers': ['F_Cu', 'B_Cu'],
        'silk_layers': ['F_Silkscreen'],
        'mask_layers': ['F_Mask'],
    }))
    Image.new('L', (2, 2), 0).save(build / 'F_Silkscreen.png')
    Image.new('L', (2, 2), 0).save(build / 'F_Mask.png')

    out = tmp_path / 'board.zip'
    args = argparse.Namespace(
        build_dir=str(build), output=str(out), title='Test Board',
        description=None, prefix=None,
    )
    assert cmd_deploy(args) == 0

    with ZipFile(out) as zf:
        names = set(zf.namelist())
        assert {'index.html', 'grid.png', 'idmap.png', 'meta.json'} <= names
        assert 'F_Silkscreen.png' in names
        assert 'F_Mask.png' in names
        assert 'Test Board' in zf.read('index.html').decode('utf-8')


def test_deploy_zip_prefix(tmp_path):
    from zipfile import ZipFile
    from pcbnets.cli import cmd_deploy
    import argparse

    build = tmp_path / 'build'
    build.mkdir()
    Image.new('L', (2, 2), 0).save(build / 'grid.png')
    Image.new('RGB', (2, 2), 0).save(build / 'idmap.png')
    (build / 'meta.json').write_text('{}')

    out = tmp_path / 'board.zip'
    args = argparse.Namespace(
        build_dir=str(build), output=str(out), title=None,
        description=None, prefix='boards/test',
    )
    assert cmd_deploy(args) == 0

    with ZipFile(out) as zf:
        names = set(zf.namelist())
        assert 'boards/test/index.html' in names
        assert 'boards/test/grid.png' in names
