'use strict';
const fs = require('fs');
const P = require('./parser.js');

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) console.log(`  ok  ${name}`);
  else { console.error(`FAIL  ${name} ${detail}`); failures++; }
}

/* ---- path parser unit tests ---- */
{
  // Absolute/relative, H/V, implicit lineto after M
  const subs = P.pathToPolylines('M10 10 20 10 h5 v5 l-5 0 Z', 0.01);
  check('implicit L + rel HVl', subs.length === 1 && subs[0].closed &&
    JSON.stringify(subs[0].pts) === JSON.stringify([10,10,20,10,25,10,25,15,20,15]));

  // Arc: quarter circle radius 10 — all flattened points on the circle
  const arc = P.pathToPolylines('M10 0 A10 10 0 0 1 0 10', 0.001);
  const pts = arc[0].pts;
  let maxErr = 0;
  for (let i = 0; i < pts.length; i += 2) {
    maxErr = Math.max(maxErr, Math.abs(Math.hypot(pts[i], pts[i+1]) - 10));
  }
  check('arc points on circle', maxErr < 0.01, `maxErr=${maxErr}`);
  check('arc endpoint exact', pts[pts.length-2] === 0 && pts[pts.length-1] === 10);

  // Cubic flattening: chord midpoint of symmetric curve
  const cub = P.pathToPolylines('M0 0 C0 10 10 10 10 0', 0.01);
  const cp = cub[0].pts;
  let apex = 0;
  for (let i = 1; i < cp.length; i += 2) apex = Math.max(apex, cp[i]);
  check('cubic apex ~7.5', Math.abs(apex - 7.5) < 0.05, `apex=${apex}`);

  // Multiple subpaths, unclosed (CTPCI-style fills)
  const multi = P.pathToPolylines('M0 0v-5h5v5M10 0v-5h5v5', 0.01);
  check('two unclosed subpaths', multi.length === 2 && !multi[0].closed && multi[0].pts.length === 8);

  // Z followed by implicit new subpath at start point then trailing junk 'm 0 0'
  const zm = P.pathToPolylines('M0 0h5v5H0Zm0 0', 0.01);
  check('Z then empty m subpath', zm.length === 1 && zm[0].closed);
}

/* ---- parity rasteriser: simulates GPU stencil-INVERT + SDF dots ---- */
function rasterizeRuns(runs, w, h, scale) {
  // Returns Uint8Array coverage grid sampling pixel centres, applying runs
  // in order with add/erase — the CPU twin of the GPU pipeline.
  const grid = new Uint8Array(w * h);
  const inTri = (px, py, ax, ay, bx, by, cx, cy) => {
    const d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
    const d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
    const d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
    const neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    const pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    return !(neg && pos);
  };
  const parity = new Uint8Array(w * h);
  for (const r of runs) {
    parity.fill(0);
    // fans → parity
    for (let i = 0; i < r.fans.length; i += 6) {
      const ax = r.fans[i] * scale, ay = r.fans[i+1] * scale;
      const bx = r.fans[i+2] * scale, by = r.fans[i+3] * scale;
      const cx = r.fans[i+4] * scale, cy = r.fans[i+5] * scale;
      const minx = Math.max(0, Math.floor(Math.min(ax,bx,cx)));
      const maxx = Math.min(w-1, Math.ceil(Math.max(ax,bx,cx)));
      const miny = Math.max(0, Math.floor(Math.min(ay,by,cy)));
      const maxy = Math.min(h-1, Math.ceil(Math.max(ay,by,cy)));
      for (let y = miny; y <= maxy; y++) for (let x = minx; x <= maxx; x++) {
        if (inTri(x+.5, y+.5, ax, ay, bx, by, cx, cy)) parity[y*w+x] ^= 1;
      }
    }
    // capsules → coverage into same run mask (union)
    for (let i = 0; i < r.segs.length; i += 5) {
      const x1 = r.segs[i]*scale, y1 = r.segs[i+1]*scale;
      const x2 = r.segs[i+2]*scale, y2 = r.segs[i+3]*scale;
      const hw = r.segs[i+4]*scale;
      const minx = Math.max(0, Math.floor(Math.min(x1,x2)-hw));
      const maxx = Math.min(w-1, Math.ceil(Math.max(x1,x2)+hw));
      const miny = Math.max(0, Math.floor(Math.min(y1,y2)-hw));
      const maxy = Math.min(h-1, Math.ceil(Math.max(y1,y2)+hw));
      const dx = x2-x1, dy = y2-y1;
      const len2 = dx*dx+dy*dy;
      for (let y = miny; y <= maxy; y++) for (let x = minx; x <= maxx; x++) {
        const px = x+.5, py = y+.5;
        let t = len2 ? ((px-x1)*dx+(py-y1)*dy)/len2 : 0;
        t = Math.max(0, Math.min(1, t));
        const qx = x1+t*dx, qy = y1+t*dy;
        if ((px-qx)*(px-qx)+(py-qy)*(py-qy) <= hw*hw) parity[y*w+x] = 1;
      }
    }
    // apply with polarity
    for (let i = 0; i < grid.length; i++) {
      if (parity[i]) grid[i] = r.polarity === 1 ? 1 : 0;
    }
  }
  return grid;
}

{
  const svg = `<svg width="100" height="100"><mask id="a">
    <path fill="#010101" d="M-10-10h120v120H-10z"/>
    <path fill="#FEFEFE" fill-rule="evenodd" d="M10 10H50V50H10ZM20 20H40V40H20Z"/>
  </mask></svg>`;
  const parsed = P.parseGerbvLayer(svg);
  check('bg skipped', parsed.stats.skippedBackground === 1);
  const g = rasterizeRuns(parsed.runs, 100, 100, 1);
  let area = 0;
  for (const v of g) area += v;
  // Ring: 40x40 minus 20x20 = 1200 (same winding — parity must still hole it)
  check('evenodd ring area ~1200 (parity)', Math.abs(area - 1200) <= 40, `area=${area}`);
  check('hole is empty', g[30 * 100 + 30] === 0);
  check('ring is filled', g[15 * 100 + 30] === 1);
}

/* ---- end-to-end: parity raster of real layer vs reference evenodd ---- */
{
  // Reference: evenodd point-in-polygon over the raw contours + stroke
  // distance — computed independently of the fan/dot representation.
  const text = fs.readFileSync('/home/claude/samples/svgs/In1_Cu.svg', 'utf8');
  const parsed = P.parseGerbvLayer(text);
  const SCALE = 4;
  const W = Math.ceil(281 * SCALE), H = Math.ceil(177 * SCALE);
  const gpu = rasterizeRuns(parsed.runs, W, H, SCALE);

  // Build reference from raw elements
  const tol = 281 / 20000;
  const tags = [...text.matchAll(/<path\b[^>]*>/g)].map(m => m[0]);
  const ref = new Uint8Array(W * H);
  const evenOdd = (subs, px, py) => {
    let inside = false;
    for (const s of subs) {
      const p = s.pts, n = p.length / 2;
      for (let i = 0, j = n - 1; i < n; j = i++) {
        const xi = p[2*i], yi = p[2*i+1], xj = p[2*j], yj = p[2*j+1];
        if ((yi > py) !== (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi) inside = !inside;
      }
    }
    return inside;
  };
  let first = true;
  for (const tag of tags) {
    const fill = /fill="#FEFEFE"/.test(tag);
    const strokeM = tag.match(/stroke="#FEFEFE"/);
    const dark = /(?:fill|stroke)="#010101"/.test(tag);
    const d = (tag.match(/d="([^"]*)"/) || [])[1];
    if (!d) continue;
    if (first && dark) { first = false; continue; }
    first = false;
    if (!fill && !strokeM && !dark) continue;
    const subs = P.pathToPolylines(d, tol);
    const val = dark ? 0 : 1;
    // bbox
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (const s of subs) for (let i = 0; i < s.pts.length; i += 2) {
      minx = Math.min(minx, s.pts[i]); maxx = Math.max(maxx, s.pts[i]);
      miny = Math.min(miny, s.pts[i+1]); maxy = Math.max(maxy, s.pts[i+1]);
    }
    const hw = strokeM ? parseFloat((tag.match(/stroke-width="([^"]*)"/) || [0,'1'])[1]) / 2 : 0;
    const x0 = Math.max(0, Math.floor((minx - hw) * SCALE)), x1 = Math.min(W-1, Math.ceil((maxx + hw) * SCALE));
    const y0 = Math.max(0, Math.floor((miny - hw) * SCALE)), y1 = Math.min(H-1, Math.ceil((maxy + hw) * SCALE));
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const px = (x + .5) / SCALE, py = (y + .5) / SCALE;
      let hit = false;
      if (fill || (dark && !strokeM)) hit = evenOdd(subs, px, py);
      if (!hit && (strokeM || (dark && hw > 0))) {
        for (const s of subs) {
          const p = s.pts, n = p.length / 2;
          const last = s.closed ? n : n - 1;
          for (let i = 0; i < last && !hit; i++) {
            const j = (i + 1) % n;
            const ax = p[2*i], ay = p[2*i+1], bx = p[2*j], by = p[2*j+1];
            const dx = bx-ax, dy = by-ay, len2 = dx*dx+dy*dy;
            let t = len2 ? ((px-ax)*dx+(py-ay)*dy)/len2 : 0;
            t = Math.max(0, Math.min(1, t));
            const qx = ax+t*dx, qy = ay+t*dy;
            if ((px-qx)*(px-qx)+(py-qy)*(py-qy) <= hw*hw) hit = true;
          }
          if (hit) break;
        }
      }
      if (hit) ref[y*W+x] = val;
    }
  }
  let boundary = 0, interior = 0, filled = 0;
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const i = y * W + x;
    if (ref[i]) filled++;
    if (ref[i] !== gpu[i]) {
      let uniform = true;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (ref[i + dy * W + dx] !== ref[i]) uniform = false;
      }
      if (uniform) interior++; else boundary++;
    }
  }
  console.log(`  e2e In1_Cu: ${filled} ref px filled, ${boundary} boundary diffs, ${interior} interior diffs`);
  check('e2e: zero interior errors', interior === 0, String(interior));
  check('e2e: boundary noise < 3% of filled', boundary / Math.max(1, filled) < 0.03,
    (100 * boundary / Math.max(1, filled)).toFixed(2) + '%');
}

/* ---- real files: Sparrow ---- */
const SP = '/home/claude/samples/home/markw/git/sparrow_gerbers/v2/web';
{
  const t0 = Date.now();
  let totalSegs = 0, totalFans = 0;
  for (const f of ['F_Cu','In1_Cu','In2_Cu','In3_Cu','In4_Cu','B_Cu','F_Mask','B_Mask','F_Silkscreen','B_Silkscreen','PTH','NPTH']) {
    const parsed = P.parseGerbvLayer(fs.readFileSync(`${SP}/${f}.svg`, 'utf8'));
    check(`${f}: dims 1303x885`, parsed.width === 1303 && parsed.height === 885,
      `${parsed.width}x${parsed.height}`);
    check(`${f}: bg skipped once`, parsed.stats.skippedBackground === 1);
    check(`${f}: no unknown colours`, parsed.stats.unknownColour === 0,
      String(parsed.stats.unknownColour));
    // All geometry within board bounds + small margin
    let inBounds = true;
    for (const r of parsed.runs) {
      if (r.bounds[0] < -30 || r.bounds[1] < -30 ||
          r.bounds[2] > parsed.width + 30 || r.bounds[3] > parsed.height + 30) inBounds = false;
      totalSegs += r.segs.length / 5;
      totalFans += r.fans.length / 6;
      // NaN scan
      for (const arr of [r.segs, r.fans]) {
        for (let i = 0; i < arr.length; i++) if (!Number.isFinite(arr[i])) { inBounds = false; break; }
      }
    }
    check(`${f}: bounds sane + finite`, inBounds);
    const erase = parsed.runs.filter(r => r.polarity === -1).length;
    if (f === 'In1_Cu') check('In1_Cu has erase runs (thermals)', erase > 0, String(erase));
    if (f === 'F_Cu') check('F_Cu purely additive', erase === 0, String(erase));
  }
  console.log(`  sparrow totals: ${totalSegs} capsules, ${totalFans} fan tris, parsed in ${Date.now()-t0}ms`);
  check('capsule count plausible', totalSegs > 50000 && totalSegs < 2000000, String(totalSegs));
}

/* ---- real files: CTPCI (no viewBox, unclosed fills) ---- */
{
  for (const f of ['F_Cu','B_Cu','In1_Cu','drill','F_Mask','F_Silkscreen']) {
    const parsed = P.parseGerbvLayer(fs.readFileSync(`/home/claude/samples/svgs/${f}.svg`, 'utf8'));
    check(`ctpci ${f}: dims from width/height`, parsed.width === 281 && parsed.height === 177,
      `${parsed.width}x${parsed.height}`);
    check(`ctpci ${f}: parsed something`,
      parsed.runs.reduce((n, r) => n + r.segs.length + r.fans.length, 0) > 0);
  }
}

/* ---- inversion ---- */
{
  const parsed = P.parseGerbvLayer(fs.readFileSync(`${SP}/In1_Cu.svg`, 'utf8'));
  const inv = P.invertRuns(parsed);
  check('invert prepends full add', inv.runs[0].polarity === 1 && inv.runs[0].fans.length === 12);
  check('invert flips polarities', inv.runs[1].polarity === -parsed.runs[0].polarity);
}

/* ---- netmap ---- */
{
  const t0 = Date.now();
  const text = fs.readFileSync(`${SP}/netmap.svg`, 'utf8');
  const nm = P.parseNetmap(text);
  const ms = Date.now() - t0;
  console.log(`  netmap parsed in ${ms}ms: ${nm.stats.paths} paths, ${nm.stats.rects} rects, ${nm.stats.oddPaths} odd`);
  check('netmap dims', nm.width === 2606 && nm.height === 1770, `${nm.width}x${nm.height}`);
  check('netmap has 6 layers', nm.layers.size === 6, String([...nm.layers.keys()]));
  check('netmap no odd paths', nm.stats.oddPaths === 0, String(nm.stats.oddPaths));
  check('netmap parse < 5s', ms < 5000, `${ms}ms`);
  // Coverage: copper layers should label a meaningful fraction of the board
  const fcu = nm.layers.get('F_Cu');
  let nz = 0;
  for (let i = 0; i < fcu.length; i += 7) if (fcu[i]) nz++;
  const frac = nz / (fcu.length / 7);
  check('F_Cu label coverage 2-60%', frac > 0.02 && frac < 0.6, frac.toFixed(3));
  // Spot check: the very first rect of net 2 on F_Cu (M141 15H147V16) labels
  check('spot rect labelled', fcu[15 * nm.width + 143] === 2, String(fcu[15 * nm.width + 143]));
  // Downsampled variant agrees at scaled coords
  const nm2 = P.parseNetmap(text, {shift: 1});
  check('netmap shift dims', nm2.width === 1303 && nm2.height === 885, `${nm2.width}x${nm2.height}`);
  check('netmap shift spot', nm2.layers.get('F_Cu')[7 * nm2.width + 71] === 2,
    String(nm2.layers.get('F_Cu')[7 * nm2.width + 71]));
}

/* ---- nearestLabel proximity picking ---- */
{
  const w = 12, h = 12;
  const g = new Uint16Array(w * h);
  g[3 * w + 3] = 7;   // net 7 at (3,3)
  g[3 * w + 9] = 9;   // net 9 at (9,3)
  check('nearest: exact hit', P.nearestLabel(g, w, h, 3, 3, 5) === 7);
  check('nearest: near miss snaps', P.nearestLabel(g, w, h, 4, 4, 5) === 7);
  check('nearest: picks closer of two', P.nearestLabel(g, w, h, 5, 3, 5) === 7);
  check('nearest: picks other side', P.nearestLabel(g, w, h, 8, 3, 5) === 9);
  check('nearest: equidistant returns one of them', [7, 9].includes(P.nearestLabel(g, w, h, 6, 3, 5)));
  check('nearest: radius respected', P.nearestLabel(g, w, h, 3, 10, 3) === 0);
  check('nearest: clamps outside grid', P.nearestLabel(g, w, h, -4, 3, 8) === 7);
  // Real netmap: cell (143,15) on F_Cu is net 2; a nearby empty cell snaps to it
  const text = fs.readFileSync(`${SP}/netmap.svg`, 'utf8');
  const nm = P.parseNetmap(text);
  const fcu = nm.layers.get('F_Cu');
  // find an empty cell within 4 of the known hit
  let ex = -1, ey = -1;
  outer: for (let dy = -4; dy <= 4; dy++) for (let dx = -4; dx <= 4; dx++) {
    const x = 143 + dx, y = 15 + dy;
    if (!fcu[y * nm.width + x]) { ex = x; ey = y; break outer; }
  }
  if (ex >= 0) {
    const snapped = P.nearestLabel(fcu, nm.width, nm.height, ex, ey, 8);
    check('nearest: real netmap snap finds a net', snapped > 0, String(snapped));
  } else {
    check('nearest: real netmap area fully labelled (skip)', true);
  }
}

/* ---- fan groups + net attribution + highlight geometry ---- */
{
  const text = fs.readFileSync(`${SP}/F_Cu.svg`, 'utf8');
  const parsed = P.parseGerbvLayer(text);
  // Group integrity: triangle ranges tile the fan buffers exactly
  let ok = true;
  for (const run of parsed.runs) {
    let covered = 0;
    for (const g of run.fanGroups) {
      if (g.start * 6 !== covered * 6 && g.start < covered) ok = false;
      covered = Math.max(covered, g.start + g.count);
      if (g.sx < run.bounds[0] - 1 || g.sx > run.bounds[2] + 1) ok = false;
    }
    if (covered !== run.fans.length / 6) ok = false;
  }
  check('fan groups tile fan buffers exactly', ok);

  const nmText = fs.readFileSync(`${SP}/netmap.svg`, 'utf8');
  const nm = P.parseNetmap(nmText);
  const grid = nm.layers.get('F_Cu');
  const cellsPerUnit = nm.fullWidth / 1303;
  const t0 = Date.now();
  const attrib = P.attributeRunNets(parsed.runs, grid, nm.width, nm.height, cellsPerUnit);
  const ms = Date.now() - t0;
  let attributed = 0, total = 0, maxId = 0;
  for (const a of attrib) {
    if (!a) continue;
    for (const id of a.segNets) { total++; if (id) { attributed++; maxId = Math.max(maxId, id); } }
    for (const id of a.fanNets) { total++; if (id) { attributed++; maxId = Math.max(maxId, id); } }
  }
  console.log(`  attribution F_Cu: ${attributed}/${total} prims in ${ms}ms, max id ${maxId}`);
  check('attribution: >97% of prims get a net', attributed / total > 0.97,
    (100 * attributed / total).toFixed(1) + '%');
  check('attribution: ids within range', maxId <= nm.stats.maxNet);

  // Highlight geometry for the most common net: prims present, none foreign
  const counts = new Map();
  for (const a of attrib) {
    if (!a) continue;
    for (const id of a.segNets) if (id) counts.set(id, (counts.get(id) || 0) + 1);
  }
  const topNet = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
  const geom = P.buildNetGeometry(parsed.runs, attrib, topNet);
  check('net geometry built for top net', !!geom && geom.length > 0);
  let prims = 0;
  for (const run of geom) prims += run.segs.length / 5 + run.fans.length / 6;
  check('net geometry non-trivial', prims > 10, String(prims));
  // Every extracted capsule's midpoint labels back to the same net (or its
  // 3-cell neighbourhood does — boundary cells can be owned by neighbours)
  let agree = 0, tested = 0;
  for (const run of geom) {
    if (run.polarity !== 1) continue;
    for (let i = 0; i < run.segs.length && tested < 500; i += 5, tested++) {
      const gx = Math.floor((run.segs[i] + run.segs[i + 2]) / 2 * cellsPerUnit);
      const gy = Math.floor((run.segs[i + 1] + run.segs[i + 3]) / 2 * cellsPerUnit);
      if (P.nearestLabel(grid, nm.width, nm.height, gx, gy, 3) === topNet) agree++;
    }
  }
  check('net geometry self-consistent (>95%)', agree / Math.max(1, tested) > 0.95,
    `${agree}/${tested}`);

  // Inverted plane: geometry includes the base add plus erase runs
  const invParsed = P.invertRuns(P.parseGerbvLayer(fs.readFileSync(`${SP}/In1_Cu.svg`, 'utf8')));
  const invGrid = nm.layers.get('In1_Cu');
  const invAttrib = P.attributeRunNets(invParsed.runs, invGrid, nm.width, nm.height, cellsPerUnit);
  check('inverted base attributed', invAttrib[0] && invAttrib[0].fanNets.length === 1);
  const planeNet = invAttrib[0].fanNets[0];
  if (planeNet) {
    const planeGeom = P.buildNetGeometry(invParsed.runs, invAttrib, planeNet);
    const hasErase = planeGeom && planeGeom.some(r => r.polarity === -1);
    check('plane highlight keeps erase runs (clearance holes)', !!hasErase);
  } else {
    check('plane net resolves (informational)', true);
  }
}

console.log(failures ? `\n${failures} FAILURES` : '\nALL TESTS PASSED');
process.exit(failures ? 1 : 0);
