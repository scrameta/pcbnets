"""Net extraction: connected components per layer, joined by drills."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from typing import Mapping

import numpy as np
from PIL import Image
from scipy.ndimage import find_objects, label


def _disk(radius: int) -> np.ndarray:
    """Disk-shaped structuring element of the given radius (in pixels)."""
    if radius < 1:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius


def _to_bool(img) -> np.ndarray:
    """Accept either a PIL image or a numpy boolean array."""
    if isinstance(img, np.ndarray):
        return img.astype(bool, copy=False)
    return np.asarray(img.convert('L')) > 0

import math
from types import SimpleNamespace


def drill_annulus_contact_ids(layer_lbl: np.ndarray,
                              layer_copper: np.ndarray,
                              prop,
                              gap_px: int = 1,
                              width_px: int = 4,
                              min_copper_frac: float = 0.10) -> list[int]:
    """
    Return copper component ids that touch the drill barrel.

    This tests a *narrow annulus* immediately outside the drill.
    For 1000 dpi PNGs, gap=1, width=4 is a reasonable starting point.

    Important:
      Do not test a big outer annulus. On plane layers, an unconnected
      anti-pad will still have copper farther away, which would cause
      false plane connections.
    """
    h, w = layer_lbl.shape

    cy, cx = prop.centroid
    r = math.sqrt(prop.area / math.pi)

    r0 = r + gap_px
    r1 = r + gap_px + width_px

    y0 = max(0, int(math.floor(cy - r1 - 2)))
    y1 = min(h, int(math.ceil(cy + r1 + 3)))
    x0 = max(0, int(math.floor(cx - r1 - 2)))
    x1 = min(w, int(math.ceil(cx + r1 + 3)))

    yy, xx = np.indices((y1 - y0, x1 - x0))
    yy = yy + y0
    xx = xx + x0

    dist = np.hypot(yy - cy, xx - cx)
    annulus = (dist >= r0) & (dist <= r1)

    if not annulus.any():
        return []

    local_copper = layer_copper[y0:y1, x0:x1] & annulus
    frac = local_copper.sum() / annulus.sum()

    if frac < min_copper_frac:
        return []

    ids = np.unique(layer_lbl[y0:y1, x0:x1][local_copper])
    ids = ids[ids != 0]

    return [int(i) for i in ids]

def extract_nets(copper_layers: dict[str, Image.Image],
                 drill: Image.Image,
                 drill_grow_px: int = 0,
                 progress: Callable[[str], None] | None = None) -> dict:
    # Do not dilate drills for connectivity. Dilation is exactly the kind
    # of thing that can jump across anti-pads/clearances.
    arr_drill = _to_bool(drill)
    # Intentionally ignore drill_grow_px for annulus/barrel-contact testing.
    # The annulus must be measured from the real drill radius; dilating first
    # moves the sampling ring outside pads/thermals and can break every via.

    copper_masks = {
        name: _to_bool(img)
        for name, img in copper_layers.items()
    }

    layer_labels = {}
    n_layers = len(copper_masks)
    for idx, (name, mask) in enumerate(copper_masks.items(), start=1):
        if progress:
            progress(f'labeling copper layer {idx}/{n_layers}: {name}')
        layer_labels[name] = label(mask)[0]

    if progress:
        progress('labeling drill mask')
    lbl_drill, n_drill = label(arr_drill)

    drill_touches = {}

    report_every = max(1, n_drill // 20)
    for drill_id, obj in enumerate(find_objects(lbl_drill), start=1):
        if progress and (drill_id == 1 or drill_id == n_drill
                         or drill_id % report_every == 0):
            progress(f'checking drill contacts {drill_id}/{n_drill}')
        if obj is None:
            continue
        local = lbl_drill[obj] == drill_id
        ys, xs = np.nonzero(local)
        if len(xs) == 0:
            continue
        cy = float(ys.mean() + obj[0].start)
        cx = float(xs.mean() + obj[1].start)
        prop = SimpleNamespace(label=drill_id, centroid=(cy, cx), area=int(len(xs)))
        members = set()

        for layer, layer_lbl in layer_labels.items():
            ids = drill_annulus_contact_ids(
                layer_lbl=layer_lbl,
                layer_copper=copper_masks[layer],
                prop=prop,
                gap_px=1,
                width_px=4,
                min_copper_frac=0.10,
            )

            for net_id in ids:
                members.add((layer, net_id))

        # This is the important NPTH/mechanical-hole guard.
        #
        # A through plated hole should normally have a visible annular pad
        # on top and/or bottom. If the only contacts are internal planes,
        # reject it as a vertical connector.
        layer_names = list(copper_layers.keys())
        outer_layers = {"F_Cu", "B_Cu"}
        if layer_names:
            outer_layers.add(layer_names[0])
            outer_layers.add(layer_names[-1])
        has_outer_contact = any(
            layer in outer_layers
            for layer, net_id in members
        )

        if has_outer_contact and members:
            drill_touches[drill_id] = members

    return {
        'layer_labels': layer_labels,
        'drill_labels': lbl_drill,
        'drill_touches': drill_touches,
    }


def merge_nets_debug(
    drill_touches: Mapping[int, set],
    layer_labels: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict]:
    """Merge components and return labels plus explainable merge metadata.

    The debug metadata is intentionally JSON-serialisable.  It records the
    original per-layer component nodes, drill edges between those nodes, and
    final merged groups, so later tools can answer "why are these two local
    copper islands on the same net?" without re-running image processing.
    """
    net_labels = merge_nets(drill_touches, layer_labels)

    component_to_net: dict[tuple[str, int], int] = {}
    components: list[dict] = []
    for layer, lbl in layer_labels.items():
        max_id = int(lbl.max())
        for component in range(1, max_id + 1):
            mask = lbl == component
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            net_id = int(net_labels[layer][ys[0], xs[0]])
            component_to_net[(layer, component)] = net_id
            components.append({
                'layer': layer,
                'component': component,
                'net': net_id,
                'area_px': int(len(xs)),
                'bbox': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            })

    drills: list[dict] = []
    groups: dict[int, list[dict]] = defaultdict(list)
    for comp in components:
        groups[int(comp['net'])].append({
            'layer': comp['layer'],
            'component': comp['component'],
        })

    for drill_id, members in sorted(drill_touches.items()):
        member_list = [
            {'layer': layer, 'component': int(component),
             'net': int(component_to_net.get((layer, int(component)), 0))}
            for layer, component in sorted(members, key=lambda m: (m[0], m[1]))
        ]
        drills.append({
            'drill': int(drill_id),
            'members': member_list,
            'nets': sorted({m['net'] for m in member_list if m['net']}),
        })

    debug = {
        'components': components,
        'drills': drills,
        'merged_nets': [
            {'net': int(net), 'members': members}
            for net, members in sorted(groups.items())
        ],
    }
    return net_labels, debug


def explain_merge_path(debug: Mapping, start: tuple[str, int], end: tuple[str, int]) -> list[dict]:
    """Return a shortest drill-by-drill path connecting two local components."""
    graph: dict[tuple[str, int], list[tuple[tuple[str, int], int]]] = defaultdict(list)
    for drill in debug.get('drills', []):
        members = [
            (m['layer'], int(m['component']))
            for m in drill.get('members', [])
        ]
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                graph[a].append((b, int(drill['drill'])))
                graph[b].append((a, int(drill['drill'])))

    q = deque([start])
    prev: dict[tuple[str, int], tuple[tuple[str, int], int] | None] = {start: None}
    while q:
        node = q.popleft()
        if node == end:
            break
        for nxt, drill_id in graph.get(node, []):
            if nxt not in prev:
                prev[nxt] = (node, drill_id)
                q.append(nxt)

    if end not in prev:
        return []

    nodes = []
    cur = end
    while cur != start:
        parent, drill_id = prev[cur]
        nodes.append((parent, drill_id, cur))
        cur = parent
    nodes.reverse()
    return [
        {
            'from': {'layer': a[0], 'component': a[1]},
            'drill': drill_id,
            'to': {'layer': b[0], 'component': b[1]},
        }
        for a, drill_id, b in nodes
    ]

class UnionFind:
    """Tiny union-find with path compression. Keys can be any hashable."""

    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression.
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def merge_nets(
    drill_touches: Mapping[int, set],
    layer_labels: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Merge per-layer components via drills into board-wide nets.

    Returns ``{layer_name: int32 ndarray}`` arrays with the same shapes as
    ``layer_labels`` but where the same integer means the same electrical
    net across layers. Net ids are densely numbered from 1; 0 = background.
    """
    uf = UnionFind()

    # Seed every (layer, component) so isolated nets also get an entry.
    for layer, lbl in layer_labels.items():
        max_id = int(lbl.max())
        for component in range(1, max_id + 1):
            uf.find((layer, component))

    # Union everything a drill touches.
    for members in drill_touches.values():
        members = list(members)
        for m in members[1:]:
            uf.union(members[0], m)

    # Assign a dense net id (1, 2, 3, ...) to each root, in a stable order
    # so re-runs produce the same colouring.
    root_to_net: dict = {}
    next_id = 1
    for node in sorted(uf.parent, key=lambda k: (k[0], k[1])):
        root = uf.find(node)
        if root not in root_to_net:
            root_to_net[root] = next_id
            next_id += 1

    net_labels: dict[str, np.ndarray] = {}
    for layer, lbl in layer_labels.items():
        max_id = int(lbl.max())
        lut = np.zeros(max_id + 1, dtype=np.int32)
        for component in range(1, max_id + 1):
            lut[component] = root_to_net[uf.find((layer, component))]
        net_labels[layer] = lut[lbl]

    return net_labels
