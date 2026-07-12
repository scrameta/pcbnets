# pcbnets WebGL viewer — Phase 1 prototype

Drop `index_webgl.html` next to `meta.json` in a generated web/ directory
(alongside the layer SVGs and netmap.svg) and open it — same serving
contract as the classic viewer, zero pipeline changes.

## What it does differently
- Parses the gerbv layer SVGs client-side into GPU primitives:
  * strokes -> instanced SDF capsules (round caps in the fragment shader,
    crisp at any zoom)
  * circular pad flashes / drills -> SDF discs (detected per subpath, with a
    bbox-isolation guard so even-odd holes stay exactly correct)
  * remaining fills -> stencil-INVERT parity + cover (handles arbitrary
    even-odd polygons, no tessellation)
  * clear polarity (#010101) -> erase blending, in document order; layers
    with corrections.invert get a full-board base + flipped polarities
- Composites per frame per layer window: substrate, copper, soldermask
  (inverted tint), silk with mask+drill knockout, drills, net highlight.
- Netmap run-length rects -> per-layer Uint16 label grids: picking is an
  array lookup, highlight is a shader compare (always visible, free).
- Pan/zoom updates a uniform; every frame is a full vector redraw. No
  previews, no settle, no interruptibility machinery.

## Testing
`node test_parser.js` runs the parser suite against the real Sparrow and
CTPCI files (path grammar, arc flattening, polarity, background skip,
circle conversion, netmap grids), including a CPU parity rasteriser that
simulates the GPU stencil pipeline and matches an independently computed
even-odd reference with zero interior errors. The same suite runs against
the script extracted from index_webgl.html.

## Numbers (Sparrow, 6 copper + masks + silk + drills)
- 153k capsule instances + 70k stencil triangles total
- layer parse ~1.1s, netmap parse ~0.5s (async after first paint)
- label grids: 6 x 2606x1770 u16 (halved on mobile)

## Knobs
- DPR_CAP, NETMAP_SHIFT (mobile), tol in parseGerbvLayer (flattening)

## Known gaps (deliberate, phase 1)
- Obround pads render via stencil (could become capsules later)
- corrections.offset is displayed but not applied (parity with classic)
- No keyboard arrow panning; no per-layer visibility toggles yet
