"""Generate a synthetic 4-layer board for testing pcbnets end-to-end.

Includes top_silk.png to demonstrate the silk overlay in the viewer.
All layers are simple signal-style (low fill), so auto-polarity-detection
reports no action — see test_audit.py for cases that exercise the
inversion heuristic directly.

Run with the package installed:

    python examples/make_example.py
    pcbnets audit  examples/synthetic
    pcbnets render examples/synthetic -o examples/build
    pcbnets serve  examples/build
"""

from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw, ImageFont

OUT = pathlib.Path(__file__).parent / 'synthetic'
OUT.mkdir(exist_ok=True)

W, H = 600, 400


def new_mask() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new('L', (W, H), 0)
    return img, ImageDraw.Draw(img)


# --- top: net A (left) and net B (right) ---
top, d = new_mask()
d.rectangle((50, 50, 250, 80),   fill=255)
d.rectangle((230, 60, 260, 200), fill=255)
d.ellipse((40, 40, 80, 80),      fill=255)
d.ellipse((240, 190, 280, 230),  fill=255)
d.rectangle((350, 50, 550, 80),  fill=255)
d.ellipse((340, 40, 380, 80),    fill=255)
d.ellipse((520, 60, 560, 100),   fill=255)

# --- inner1: net A long horizontal trace ---
inner1, d = new_mask()
d.rectangle((40, 220, 500, 250), fill=255)
d.ellipse((240, 200, 280, 240),  fill=255)
d.ellipse((480, 220, 520, 260),  fill=255)

# --- inner2: net B vertical link between top and bottom ---
inner2, d = new_mask()
d.rectangle((300, 280, 540, 310), fill=255)
d.ellipse((520, 80, 560, 120),   fill=255)
d.rectangle((530, 90, 545, 295), fill=255)

# --- bottom: bits of each net + an isolated trace ---
bottom, d = new_mask()
d.ellipse((480, 220, 520, 260), fill=255)
d.rectangle((100, 320, 300, 350), fill=255)
d.ellipse((480, 100, 540, 160), fill=255)

# --- drill: vias ---
drill, d = new_mask()
def via(x, y, r=12):
    d.ellipse((x - r, y - r, x + r, y + r), fill=255)
via(60, 60)
via(260, 210)
via(500, 240)
via(360, 60)
via(540, 100)
via(540, 290)

# --- top silk: component outlines + reference designators ---
top_silk, d = new_mask()
try:
    font = ImageFont.load_default()
    d.text((45, 12), 'R1',  fill=255, font=font)
    d.text((345, 12), 'C1', fill=255, font=font)
    d.text((265, 235), 'TP1', fill=255, font=font)
    d.rectangle((40, 25, 90, 40),   outline=255)
    d.rectangle((340, 25, 390, 40), outline=255)
    d.rectangle((230, 220, 290, 245), outline=255)
except Exception:
    pass


for name, img in [
    ('top', top), ('inner1', inner1), ('inner2', inner2),
    ('bottom', bottom), ('drill', drill), ('top_silk', top_silk),
]:
    img.save(OUT / f'{name}.png')
    print(f'wrote {OUT / f"{name}.png"}')

print()
print('Now run:')
print(f'  pcbnets audit  {OUT}')
print(f'  pcbnets render {OUT} -o {OUT.parent}/build')
print(f'  pcbnets serve  {OUT.parent}/build')
