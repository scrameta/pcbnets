// pcbnets WebGL viewer — geometry parser.
// DOM-free on purpose: runs identically in the browser and under node for
// testing against real gerbv output. Parses gerbv SVG layers (mask-based,
// white=add / #010101=erase) into GPU-ready primitive runs, and pcbnets
// netmap SVGs (run-length rect paths) into per-layer net-id grids.

'use strict';

/* ------------------------------------------------------------------ */
/* SVG path data → polylines                                           */
/* ------------------------------------------------------------------ */

// Tokenize a path `d` string into commands + numbers. Returns flat array
// alternating command chars and their numeric arguments.
function tokenizePath(d) {
  const tokens = [];
  const re = /([MLHVCSQTAZmlhvcsqtaz])|(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)/g;
  let m;
  while ((m = re.exec(d)) !== null) tokens.push(m[1] !== undefined ? m[1] : parseFloat(m[2]));
  return tokens;
}

function subdivideCubic(x0, y0, x1, y1, x2, y2, x3, y3, tol, depth, out) {
  // Flatness: max distance of control points from the chord.
  const dx = x3 - x0, dy = y3 - y0;
  const d1 = Math.abs((x1 - x0) * dy - (y1 - y0) * dx);
  const d2 = Math.abs((x2 - x0) * dy - (y2 - y0) * dx);
  const len = Math.hypot(dx, dy) || 1e-12;
  if (depth >= 16 || (d1 + d2) / len <= tol) {
    out.push(x3, y3);
    return;
  }
  const x01 = (x0 + x1) / 2, y01 = (y0 + y1) / 2;
  const x12 = (x1 + x2) / 2, y12 = (y1 + y2) / 2;
  const x23 = (x2 + x3) / 2, y23 = (y2 + y3) / 2;
  const x012 = (x01 + x12) / 2, y012 = (y01 + y12) / 2;
  const x123 = (x12 + x23) / 2, y123 = (y12 + y23) / 2;
  const xm = (x012 + x123) / 2, ym = (y012 + y123) / 2;
  subdivideCubic(x0, y0, x01, y01, x012, y012, xm, ym, tol, depth + 1, out);
  subdivideCubic(xm, ym, x123, y123, x23, y23, x3, y3, tol, depth + 1, out);
}

// Endpoint → centre parametrisation per W3C spec, then subdivide by angle.
function flattenArc(x0, y0, rx, ry, xrot, largeArc, sweep, x1, y1, tol, out) {
  rx = Math.abs(rx); ry = Math.abs(ry);
  if (rx < 1e-12 || ry < 1e-12 || (x0 === x1 && y0 === y1)) { out.push(x1, y1); return; }
  const phi = xrot * Math.PI / 180;
  const cosp = Math.cos(phi), sinp = Math.sin(phi);
  const dx2 = (x0 - x1) / 2, dy2 = (y0 - y1) / 2;
  const x1p = cosp * dx2 + sinp * dy2;
  const y1p = -sinp * dx2 + cosp * dy2;
  const lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
  if (lam > 1) { const s = Math.sqrt(lam); rx *= s; ry *= s; }
  const sign = largeArc !== sweep ? 1 : -1;
  const num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p;
  const den = rx * rx * y1p * y1p + ry * ry * x1p * x1p;
  const coef = sign * Math.sqrt(Math.max(0, num / den));
  const cxp = coef * (rx * y1p) / ry;
  const cyp = coef * (-ry * x1p) / rx;
  const cx = cosp * cxp - sinp * cyp + (x0 + x1) / 2;
  const cy = sinp * cxp + cosp * cyp + (y0 + y1) / 2;
  const angle = (ux, uy, vx, vy) => {
    const dot = ux * vx + uy * vy;
    const len = Math.hypot(ux, uy) * Math.hypot(vx, vy) || 1e-12;
    let a = Math.acos(Math.min(1, Math.max(-1, dot / len)));
    if (ux * vy - uy * vx < 0) a = -a;
    return a;
  };
  const th1 = angle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry);
  let dth = angle((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry);
  if (!sweep && dth > 0) dth -= 2 * Math.PI;
  if (sweep && dth < 0) dth += 2 * Math.PI;
  // Angular step from chord tolerance: chord error ≈ r(1-cos(step/2)).
  const r = Math.max(rx, ry);
  const step = Math.max(0.05, 2 * Math.acos(Math.min(1, Math.max(-1, 1 - tol / Math.max(r, 1e-9)))));
  const n = Math.max(1, Math.min(256, Math.ceil(Math.abs(dth) / step)));
  for (let i = 1; i <= n; i++) {
    const th = th1 + dth * (i / n);
    const ex = rx * Math.cos(th), ey = ry * Math.sin(th);
    out.push(cosp * ex - sinp * ey + cx, sinp * ex + cosp * ey + cy);
  }
  // Snap the final point exactly to the endpoint.
  out[out.length - 2] = x1;
  out[out.length - 1] = y1;
}

// Parse `d` into subpaths: [{pts: [x,y,...], closed: bool}, ...]
// Curves/arcs are flattened with the given chord tolerance (board units).
function pathToPolylines(d, tol) {
  const t = tokenizePath(d);
  const subs = [];
  let pts = null;
  let cx = 0, cy = 0;       // current point
  let sx = 0, sy = 0;       // subpath start
  let px = 0, py = 0;       // previous control point (for S/T)
  let prevCmd = '';
  let i = 0;
  const num = () => t[i++];
  const startSub = (x, y) => {
    if (pts && pts.length >= 4) subs.push({pts, closed: false});
    pts = [x, y];
    sx = x; sy = y;
  };
  while (i < t.length) {
    let cmd = t[i];
    if (typeof cmd === 'string') { i++; } else { cmd = prevCmd === 'M' ? 'L' : prevCmd === 'm' ? 'l' : prevCmd; }
    const rel = cmd >= 'a';
    switch (cmd.toUpperCase()) {
      case 'M': { let x = num(), y = num(); if (rel) { x += cx; y += cy; } startSub(x, y); cx = x; cy = y; break; }
      case 'L': { let x = num(), y = num(); if (rel) { x += cx; y += cy; } pts.push(x, y); cx = x; cy = y; break; }
      case 'H': { let x = num(); if (rel) x += cx; pts.push(x, cy); cx = x; break; }
      case 'V': { let y = num(); if (rel) y += cy; pts.push(cx, y); cy = y; break; }
      case 'C': {
        let x1 = num(), y1 = num(), x2 = num(), y2 = num(), x = num(), y = num();
        if (rel) { x1 += cx; y1 += cy; x2 += cx; y2 += cy; x += cx; y += cy; }
        subdivideCubic(cx, cy, x1, y1, x2, y2, x, y, tol, 0, pts);
        px = x2; py = y2; cx = x; cy = y; break;
      }
      case 'S': {
        let x2 = num(), y2 = num(), x = num(), y = num();
        if (rel) { x2 += cx; y2 += cy; x += cx; y += cy; }
        const refl = /[CScs]/.test(prevCmd);
        const x1 = refl ? 2 * cx - px : cx, y1 = refl ? 2 * cy - py : cy;
        subdivideCubic(cx, cy, x1, y1, x2, y2, x, y, tol, 0, pts);
        px = x2; py = y2; cx = x; cy = y; break;
      }
      case 'Q': {
        let qx = num(), qy = num(), x = num(), y = num();
        if (rel) { qx += cx; qy += cy; x += cx; y += cy; }
        // Elevate quadratic to cubic.
        const c1x = cx + 2 / 3 * (qx - cx), c1y = cy + 2 / 3 * (qy - cy);
        const c2x = x + 2 / 3 * (qx - x), c2y = y + 2 / 3 * (qy - y);
        subdivideCubic(cx, cy, c1x, c1y, c2x, c2y, x, y, tol, 0, pts);
        px = qx; py = qy; cx = x; cy = y; break;
      }
      case 'T': {
        let x = num(), y = num();
        if (rel) { x += cx; y += cy; }
        const refl = /[QTqt]/.test(prevCmd);
        const qx = refl ? 2 * cx - px : cx, qy = refl ? 2 * cy - py : cy;
        const c1x = cx + 2 / 3 * (qx - cx), c1y = cy + 2 / 3 * (qy - cy);
        const c2x = x + 2 / 3 * (qx - x), c2y = y + 2 / 3 * (qy - y);
        subdivideCubic(cx, cy, c1x, c1y, c2x, c2y, x, y, tol, 0, pts);
        px = qx; py = qy; cx = x; cy = y; break;
      }
      case 'A': {
        let rx = num(), ry = num(), rot = num(), laf = num(), swf = num(), x = num(), y = num();
        if (rel) { x += cx; y += cy; }
        flattenArc(cx, cy, rx, ry, rot, !!laf, !!swf, x, y, tol, pts);
        cx = x; cy = y; break;
      }
      case 'Z': {
        if (pts && pts.length >= 4) { subs.push({pts, closed: true}); }
        pts = null;
        cx = sx; cy = sy;
        // A new subpath may start implicitly at the old start point.
        pts = [sx, sy];
        break;
      }
      default: i++; // unknown token; skip defensively
    }
    prevCmd = cmd;
  }
  if (pts && pts.length >= 4) subs.push({pts, closed: false});
  return subs;
}

/* ------------------------------------------------------------------ */
/* gerbv layer SVG → primitive runs                                    */
/* ------------------------------------------------------------------ */

function attrOf(tag, name) {
  const m = tag.match(new RegExp(`${name}="([^"]*)"`));
  return m ? m[1] : null;
}

function colourPolarity(value) {
  if (!value || value === 'none') return 0;
  const v = value.toLowerCase();
  if (v === '#fefefe' || v === '#fff' || v === '#ffffff' || v === 'white') return 1;
  if (v === '#010101' || v === '#000' || v === '#000000' || v === 'black') return -1;
  return 0;
}

// Parse a gerbv layer SVG. Returns:
// { width, height, runs, stats }
// runs: [{polarity, segs: Float32Array(5*n), fans: Float32Array(6*m),
//         bounds: [minx,miny,maxx,maxy]}], in document order, polarity-merged.
function parseGerbvLayer(text, options = {}) {
  const rootMatch = text.match(/<svg\b[^>]*>/);
  if (!rootMatch) throw new Error('no <svg> root');
  const root = rootMatch[0];
  let width, height;
  const vb = attrOf(root, 'viewBox');
  if (vb) {
    const parts = vb.trim().split(/[\s,]+/).map(Number);
    width = parts[2]; height = parts[3];
  } else {
    width = parseFloat(attrOf(root, 'width'));
    height = parseFloat(attrOf(root, 'height'));
  }
  if (!(width > 0 && height > 0)) throw new Error('no usable dimensions');

  const tol = options.tol || Math.max(width, height) / 20000;

  // Artwork lives inside the (single) mask; fall back to whole doc if absent.
  const maskStart = text.indexOf('<mask');
  let content;
  if (maskStart >= 0) {
    const open = text.indexOf('>', maskStart) + 1;
    const close = text.indexOf('</mask>', open);
    content = text.slice(open, close);
  } else {
    content = text.slice(rootMatch.index + root.length);
  }

  const stats = {elements: 0, segments: 0, fanTris: 0, skippedBackground: 0, unknownColour: 0, dots: 0};
  const runs = [];
  let run = null;
  const openRun = polarity => {
    if (!run || run.polarity !== polarity) {
      run = {polarity, segs: [], fans: [], fanGroups: [],
             bounds: [Infinity, Infinity, -Infinity, -Infinity]};
      runs.push(run);
    }
    return run;
  };
  const grow = (b, x, y) => {
    if (x < b[0]) b[0] = x; if (y < b[1]) b[1] = y;
    if (x > b[2]) b[2] = x; if (y > b[3]) b[3] = y;
  };

  const tagRe = /<(path|circle|rect|line|polyline|polygon|ellipse)\b[^>]*>/g;
  let m;
  let firstElement = true;
  while ((m = tagRe.exec(content)) !== null) {
    const tag = m[0];
    const kind = m[1];
    stats.elements++;
    const fillPol = colourPolarity(attrOf(tag, 'fill'));
    const strokePol = colourPolarity(attrOf(tag, 'stroke'));
    if (!fillPol && !strokePol) { stats.unknownColour++; firstElement = false; continue; }

    // Collect geometry as subpaths.
    let subs = null;
    if (kind === 'path') {
      const d = attrOf(tag, 'd');
      if (!d) continue;
      subs = pathToPolylines(d, tol);
    } else if (kind === 'circle' || kind === 'ellipse') {
      const cx = parseFloat(attrOf(tag, 'cx') || '0');
      const cy = parseFloat(attrOf(tag, 'cy') || '0');
      const rx = parseFloat(kind === 'circle' ? attrOf(tag, 'r') : attrOf(tag, 'rx'));
      const ry = parseFloat(kind === 'circle' ? attrOf(tag, 'r') : attrOf(tag, 'ry'));
      const n = Math.max(16, Math.min(128, Math.ceil(2 * Math.PI / Math.max(0.05, 2 * Math.acos(Math.max(-1, 1 - tol / Math.max(rx, ry, 1e-9)))))));
      const pts = [];
      for (let i = 0; i < n; i++) {
        const a = 2 * Math.PI * i / n;
        pts.push(cx + rx * Math.cos(a), cy + ry * Math.sin(a));
      }
      subs = [{pts, closed: true}];
    } else if (kind === 'rect') {
      const x = parseFloat(attrOf(tag, 'x') || '0');
      const y = parseFloat(attrOf(tag, 'y') || '0');
      const w = parseFloat(attrOf(tag, 'width'));
      const h = parseFloat(attrOf(tag, 'height'));
      subs = [{pts: [x, y, x + w, y, x + w, y + h, x, y + h], closed: true}];
    } else if (kind === 'line') {
      const x1 = parseFloat(attrOf(tag, 'x1') || '0'), y1 = parseFloat(attrOf(tag, 'y1') || '0');
      const x2 = parseFloat(attrOf(tag, 'x2') || '0'), y2 = parseFloat(attrOf(tag, 'y2') || '0');
      subs = [{pts: [x1, y1, x2, y2], closed: false}];
    } else if (kind === 'polyline' || kind === 'polygon') {
      const pts = (attrOf(tag, 'points') || '').trim().split(/[\s,]+/).map(Number);
      subs = [{pts, closed: kind === 'polygon'}];
    }
    if (!subs || !subs.length) { firstElement = false; continue; }

    // gerbv's mask base: the FIRST element, a dark fill covering the board.
    if (firstElement && fillPol === -1) {
      let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
      for (const s of subs) for (let i = 0; i < s.pts.length; i += 2) {
        minx = Math.min(minx, s.pts[i]); maxx = Math.max(maxx, s.pts[i]);
        miny = Math.min(miny, s.pts[i + 1]); maxy = Math.max(maxy, s.pts[i + 1]);
      }
      if (minx <= 0 && miny <= 0 && maxx >= width && maxy >= height) {
        stats.skippedBackground++;
        firstElement = false;
        continue;
      }
    }
    firstElement = false;

    if (strokePol) {
      const hw = parseFloat(attrOf(tag, 'stroke-width') || '1') / 2;
      const r = openRun(strokePol);
      for (const s of subs) {
        const p = s.pts;
        const n = p.length / 2;
        if (n === 1) {
          r.segs.push(p[0], p[1], p[0], p[1], hw);
          grow(r.bounds, p[0] - hw, p[1] - hw); grow(r.bounds, p[0] + hw, p[1] + hw);
          stats.dots++;
          continue;
        }
        const last = s.closed ? n : n - 1;
        for (let i = 0; i < last; i++) {
          const j = (i + 1) % n;
          const x1 = p[2 * i], y1 = p[2 * i + 1], x2 = p[2 * j], y2 = p[2 * j + 1];
          r.segs.push(x1, y1, x2, y2, hw);
          grow(r.bounds, Math.min(x1, x2) - hw, Math.min(y1, y2) - hw);
          grow(r.bounds, Math.max(x1, x2) + hw, Math.max(y1, y2) + hw);
          stats.segments++;
        }
      }
    }

    if (fillPol) {
      const r = openRun(fillPol);
      // Circle subpaths (pad flashes, drills — gerbv's two-cubic circles)
      // become SDF dot capsules: exact discs, 1 instance instead of ~20-100
      // stencil triangles. Parity guard: only when the circle's bbox doesn't
      // intersect any sibling subpath's bbox, so even-odd holes and
      // overlapping shapes keep the stencil path and stay exactly correct.
      const boxes = subs.map(s => {
        const p = s.pts;
        let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
        for (let i = 0; i < p.length; i += 2) {
          if (p[i] < minx) minx = p[i]; if (p[i] > maxx) maxx = p[i];
          if (p[i + 1] < miny) miny = p[i + 1]; if (p[i + 1] > maxy) maxy = p[i + 1];
        }
        return [minx, miny, maxx, maxy];
      });
      const isolated = index => {
        const a = boxes[index];
        for (let k = 0; k < boxes.length; k++) {
          if (k === index) continue;
          const b = boxes[k];
          if (subs[k].pts.length < 6) continue; // degenerate artifacts
          if (a[0] <= b[2] && a[2] >= b[0] && a[1] <= b[3] && a[3] >= b[1]) return false;
        }
        return true;
      };
      const asCircle = s => {
        const p = s.pts;
        const n = p.length / 2;
        if (!s.closed && n < 8) return null;
        if (n < 8) return null;
        let cx = 0, cy = 0;
        for (let i = 0; i < p.length; i += 2) { cx += p[i]; cy += p[i + 1]; }
        cx /= n; cy /= n;
        let rSum = 0;
        for (let i = 0; i < p.length; i += 2) rSum += Math.hypot(p[i] - cx, p[i + 1] - cy);
        const radius = rSum / n;
        if (!(radius > 0)) return null;
        let dev = 0;
        for (let i = 0; i < p.length; i += 2) {
          dev = Math.max(dev, Math.abs(Math.hypot(p[i] - cx, p[i + 1] - cy) - radius));
        }
        // 4% covers gerbv's 4/3 two-cubic circle approximation (~2.7% radial
        // error) while rejecting rects (41%) and obrounds.
        return dev <= Math.max(tol * 3, 0.04 * radius) ? {cx, cy, radius} : null;
      };

      const fanSubs = [];
      for (let si = 0; si < subs.length; si++) {
        const s = subs[si];
        if (s.pts.length < 6) continue; // 'Zm0 0' artifacts and stray moves
        const circle = asCircle(s);
        if (circle && isolated(si)) {
          r.segs.push(circle.cx, circle.cy, circle.cx, circle.cy, circle.radius);
          grow(r.bounds, circle.cx - circle.radius, circle.cy - circle.radius);
          grow(r.bounds, circle.cx + circle.radius, circle.cy + circle.radius);
          stats.dots++;
        } else {
          fanSubs.push(s);
        }
      }

      // Even-odd fill via stencil-invert fan: anchor at the first remaining
      // vertex; every edge (with implicit closing) contributes one triangle
      // (anchor, vi, vj). Order-independent, hole-correct.
      if (fanSubs.length) {
        const ax = fanSubs[0].pts[0], ay = fanSubs[0].pts[1];
        for (const s of fanSubs) {
          const p = s.pts;
          const n = p.length / 2;
          const startTri = r.fans.length / 6;
          const bb = boxes[subs.indexOf(s)];
          for (let i = 0; i < n; i++) {
            const j = (i + 1) % n; // fills always auto-close
            const x1 = p[2 * i], y1 = p[2 * i + 1], x2 = p[2 * j], y2 = p[2 * j + 1];
            if (x1 === x2 && y1 === y2) continue;
            r.fans.push(ax, ay, x1, y1, x2, y2);
            grow(r.bounds, x1, y1); grow(r.bounds, x2, y2);
            stats.fanTris++;
          }
          const count = r.fans.length / 6 - startTri;
          if (count > 0) {
            // Sample point (bbox centre) lets the viewer attribute this
            // subpath to an electrical net via the label grid.
            r.fanGroups.push({start: startTri, count,
                              sx: (bb[0] + bb[2]) / 2, sy: (bb[1] + bb[3]) / 2,
                              bounds: bb});
          }
        }
        grow(r.bounds, ax, ay);
      }
    }
  }

  for (const r of runs) {
    r.segs = new Float32Array(r.segs);
    r.fans = new Float32Array(r.fans);
  }
  return {width, height, runs, stats};
}

// Apply an inversion correction: full-board add first, then all polarities
// flipped (equivalent to swapping white/black in the gerbv mask).
function invertRuns(parsed) {
  const {width, height} = parsed;
  const base = {
    polarity: 1,
    segs: new Float32Array(0),
    fans: new Float32Array([0, 0, width, 0, width, height, 0, 0, width, height, 0, height]),
    fanGroups: [{start: 0, count: 2, sx: width / 2, sy: height / 2,
                 bounds: [0, 0, width, height]}],
    bounds: [0, 0, width, height],
  };
  const flipped = parsed.runs.map(r => ({...r, polarity: -r.polarity}));
  return {...parsed, runs: [base, ...flipped]};
}

// Assign an electrical net id to every additive primitive by sampling the
// per-layer label grid. cellsPerUnit converts board units to grid cells.
// Capsules sample their midpoint (endpoints as fallback); fan groups sample
// their bbox centre — except very large fills (plane bases), which take a
// majority vote of five spread samples so a clearance or small island under
// the centre can't misattribute the whole plane.
function attributeRunNets(runs, grid, gw, gh, cellsPerUnit) {
  const lookup = (bx, by) => {
    const gx = Math.floor(bx * cellsPerUnit), gy = Math.floor(by * cellsPerUnit);
    return nearestLabel(grid, gw, gh, gx, gy, 3);
  };
  return runs.map(run => {
    if (run.polarity !== 1) return null;
    const segCount = run.segs.length / 5;
    const segNets = new Uint16Array(segCount);
    for (let i = 0; i < segCount; i++) {
      const o = i * 5;
      let id = lookup((run.segs[o] + run.segs[o + 2]) / 2, (run.segs[o + 1] + run.segs[o + 3]) / 2);
      if (!id) id = lookup(run.segs[o], run.segs[o + 1]);
      if (!id) id = lookup(run.segs[o + 2], run.segs[o + 3]);
      segNets[i] = id;
    }
    const fanNets = new Uint16Array(run.fanGroups.length);
    const gridArea = (gw / cellsPerUnit) * (gh / cellsPerUnit);
    run.fanGroups.forEach((group, gi) => {
      const b = group.bounds;
      const area = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
      if (area > gridArea * 0.2) {
        const xs = [group.sx, (b[0] * 3 + b[2]) / 4, (b[0] + b[2] * 3) / 4, group.sx, group.sx];
        const ys = [group.sy, group.sy, group.sy, (b[1] * 3 + b[3]) / 4, (b[1] + b[3] * 3) / 4];
        const votes = new Map();
        for (let i = 0; i < 5; i++) {
          const id = lookup(xs[i], ys[i]);
          if (id) votes.set(id, (votes.get(id) || 0) + 1);
        }
        let best = 0, bestVotes = 0;
        for (const [id, v] of votes) if (v > bestVotes) { best = id; bestVotes = v; }
        fanNets[gi] = best;
      } else {
        fanNets[gi] = lookup(group.sx, group.sy);
      }
    });
    return {segNets, fanNets};
  });
}

// Build the highlight geometry for one layer + net: the net's additive
// primitives, in original order, interleaved with ALL erase primitives —
// clearances legitimately subtract from every net's copper, which makes
// inverted plane layers highlight correctly (plane minus its holes).
function buildNetGeometry(runs, attribution, netId) {
  const out = [];
  let any = false;
  for (let ri = 0; ri < runs.length; ri++) {
    const run = runs[ri];
    if (run.polarity === -1) {
      if (run.segs.length || run.fans.length) {
        out.push({polarity: -1, segs: run.segs, fans: run.fans, bounds: run.bounds});
      }
      continue;
    }
    const attrib = attribution[ri];
    if (!attrib) continue;
    const segsOut = [];
    const bounds = [Infinity, Infinity, -Infinity, -Infinity];
    const grow = (x, y) => {
      if (x < bounds[0]) bounds[0] = x; if (y < bounds[1]) bounds[1] = y;
      if (x > bounds[2]) bounds[2] = x; if (y > bounds[3]) bounds[3] = y;
    };
    for (let i = 0; i < attrib.segNets.length; i++) {
      if (attrib.segNets[i] !== netId) continue;
      const o = i * 5;
      const hw = run.segs[o + 4];
      segsOut.push(run.segs[o], run.segs[o + 1], run.segs[o + 2], run.segs[o + 3], hw);
      grow(Math.min(run.segs[o], run.segs[o + 2]) - hw, Math.min(run.segs[o + 1], run.segs[o + 3]) - hw);
      grow(Math.max(run.segs[o], run.segs[o + 2]) + hw, Math.max(run.segs[o + 1], run.segs[o + 3]) + hw);
    }
    const fanChunks = [];
    let fanFloats = 0;
    run.fanGroups.forEach((group, gi) => {
      if (attrib.fanNets[gi] !== netId) return;
      fanChunks.push(run.fans.subarray(group.start * 6, (group.start + group.count) * 6));
      fanFloats += group.count * 6;
      grow(group.bounds[0], group.bounds[1]);
      grow(group.bounds[2], group.bounds[3]);
    });
    if (!segsOut.length && !fanFloats) continue;
    const fans = new Float32Array(fanFloats);
    let off = 0;
    for (const chunk of fanChunks) { fans.set(chunk, off); off += chunk.length; }
    out.push({polarity: 1, segs: new Float32Array(segsOut), fans, bounds});
    any = true;
  }
  return any ? out : null;
}

/* ------------------------------------------------------------------ */
/* netmap SVG → per-layer Uint16 net-id grids                          */
/* ------------------------------------------------------------------ */

// The netmap encodes a label raster as run-length rects:
//   M x y H x2 V y2 H x Z  (repeated) — axis-aligned, absolute commands.
// `shift` right-shifts coordinates (downsample by 2^shift) for mobile.
function parseNetmap(text, {shift = 0} = {}) {
  const vbMatch = text.match(/viewBox="([\d.\s-]+)"/);
  if (!vbMatch) throw new Error('netmap: no viewBox');
  const vb = vbMatch[1].trim().split(/\s+/).map(Number);
  const fullW = Math.round(vb[2]), fullH = Math.round(vb[3]);
  const w = fullW >> shift, h = fullH >> shift;

  const layers = new Map();
  const stats = {paths: 0, rects: 0, oddPaths: 0, maxNet: 0};

  // Iterate <g data-layer="..."> sections.
  const groupRe = /<g\s+data-layer="([^"]+)"\s*>/g;
  let gm;
  const groupStarts = [];
  while ((gm = groupRe.exec(text)) !== null) groupStarts.push({name: gm[1], at: gm.index + gm[0].length});
  for (let gi = 0; gi < groupStarts.length; gi++) {
    const {name, at} = groupStarts[gi];
    const end = gi + 1 < groupStarts.length
      ? text.lastIndexOf('</g>', groupStarts[gi + 1].at)
      : text.lastIndexOf('</g>');
    const section = text.slice(at, end);
    let grid = layers.get(name);
    if (!grid) { grid = new Uint16Array(w * h); layers.set(name, grid); }

    const pathRe = /<path\b[^>]*data-net-id="(\d+)"[^>]*\sd="([^"]*)"/g;
    let pm;
    while ((pm = pathRe.exec(section)) !== null) {
      const id = Number(pm[1]);
      if (id > 65535) throw new Error('net id exceeds u16');
      if (id > stats.maxNet) stats.maxNet = id;
      stats.paths++;
      const d = pm[2];
      // Tight scanner for the M x y H a V b H c Z pattern.
      let i = 0;
      const n = d.length;
      const readNum = () => {
        while (i < n && (d[i] === ' ' || d[i] === ',')) i++;
        let s = i;
        while (i < n && (d.charCodeAt(i) === 46 || d.charCodeAt(i) === 45 ||
               (d.charCodeAt(i) >= 48 && d.charCodeAt(i) <= 57))) i++;
        return parseFloat(d.slice(s, i));
      };
      let ok = true;
      while (i < n && ok) {
        while (i < n && (d[i] === ' ' || d[i] === ',')) i++;
        if (i >= n) break;
        if (d[i] !== 'M') { ok = false; break; }
        i++;
        const x = readNum(), y = readNum();
        if (i >= n || d[i] !== 'H') { ok = false; break; }
        i++;
        const x2 = readNum();
        if (i >= n || d[i] !== 'V') { ok = false; break; }
        i++;
        const y2 = readNum();
        if (i >= n || d[i] !== 'H') { ok = false; break; }
        i++;
        readNum(); // back to x
        if (i >= n || d[i] !== 'Z') { ok = false; break; }
        i++;
        // Fill rect [x..x2) × [y..y2)
        const gx0 = Math.min(w - 1, Math.max(0, Math.round(x) >> shift));
        const gy0 = Math.min(h - 1, Math.max(0, Math.round(y) >> shift));
        const gx1 = Math.min(w, Math.max(gx0 + 1, Math.round(x2) >> shift));
        const gy1 = Math.min(h, Math.max(gy0 + 1, Math.round(y2) >> shift));
        for (let yy = gy0; yy < gy1; yy++) {
          grid.fill(id, yy * w + gx0, yy * w + gx1);
        }
        stats.rects++;
      }
      if (!ok) stats.oddPaths++;
    }
  }
  return {width: w, height: h, fullWidth: fullW, fullHeight: fullH, shift, layers, stats};
}

// Nearest non-zero label within maxR cells (Euclidean) of (gx, gy), or 0.
// Scans expanding Chebyshev rings, stopping once no closer hit is possible.
function nearestLabel(grid, w, h, gx, gy, maxR) {
  gx = Math.min(w - 1, Math.max(0, gx));
  gy = Math.min(h - 1, Math.max(0, gy));
  const at = (x, y) => (x >= 0 && y >= 0 && x < w && y < h) ? grid[y * w + x] : 0;
  const centre = at(gx, gy);
  if (centre) return centre;
  let best = 0;
  let bestD2 = Infinity;
  for (let r = 1; r <= maxR; r++) {
    if (r * r > bestD2) break;
    for (let i = -r; i <= r; i++) {
      const ring = [[gx + i, gy - r], [gx + i, gy + r], [gx - r, gy + i], [gx + r, gy + i]];
      for (const [x, y] of ring) {
        const v = at(x, y);
        if (!v) continue;
        const d2 = (x - gx) * (x - gx) + (y - gy) * (y - gy);
        if (d2 < bestD2) { bestD2 = d2; best = v; }
      }
    }
  }
  return best;
}

if (typeof module !== 'undefined') {
  module.exports = {tokenizePath, pathToPolylines, parseGerbvLayer, invertRuns, parseNetmap, flattenArc, nearestLabel, attributeRunNets, buildNetGeometry};
}
