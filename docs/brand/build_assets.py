"""Generate the pyinterp logos, favicons and cards.

The SVG files are written by hand from the geometry below; the wordmark is
outlined from Inter (SIL OFL), so the font is only needed to regenerate them.

Usage::

    python docs/brand/build_assets.py path/to/Inter[opsz,wght].ttf

Requirements: fonttools, uharfbuzz, cairosvg and pillow. Inter is available
from https://github.com/google/fonts/tree/main/ofl/inter.
"""

import io
import pathlib
import sys

import cairosvg
import uharfbuzz as hb
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from PIL import Image

HERE = pathlib.Path(__file__).absolute().parent
STATIC = HERE.parent / "source" / "_static"

#: Brand tokens (see docs/brand.md).
LIGHT = {
    "ink": "#0B2540",
    "ink-muted": "#4F6680",
    "ocean-900": "#0B4F7C",
    "ocean-700": "#0E6BA8",
    "ocean-500": "#2E9CCA",
    "ocean-300": "#9ED3EC",
    "point": "#D9700F",
    "paper": "#FFFFFF",
    "surface": "#F3F6F9",
}
DARK = {
    "ink": "#E6EEF5",
    "ink-muted": "#9DB0C3",
    "ocean-900": "#A3D9F7",
    "ocean-700": "#5DB3E8",
    "ocean-500": "#2F82B8",
    "ocean-300": "#1B4B6C",
    "point": "#FFA54F",
    "paper": "#0F1720",
    "surface": "#16212C",
}

#: Wordmark settings: Inter, SemiBold, display optical size, -0.02 em.
WORDMARK = {"wght": 600, "opsz": 32}
#: Tagline settings, on the social card.
TAGLINE = {"wght": 450, "opsz": 32}
TRACKING = -0.02


# --- Mark ---------------------------------------------------------------------
# All coordinates are on a 24-unit grid. The cell spans (4, 4)-(20, 20) and the
# interpolated point P sits at (9, 10), so the four rectangles have the areas
# 30, 66, 50 and 110: the largest one gets the darkest step of the ramp.


def mark(c: dict[str, str], *, favicon: bool = False, mono: bool = False,
         classes: bool = False) -> str:
    """Return the SVG elements of the mark, in a 24-unit square."""
    if favicon:
        # Larger cell, no overshoots: legible at 16 px.
        x0, x1, px, py, sw, rn, rp, halo = 2, 22, 8, 9, 1.5, 1.75, 2.25, 1
        overshoot = 0
    else:
        x0, x1, px, py, sw, rn, rp, halo = 4, 20, 9, 10, 1.25, 1.5, 1.75, 0.75
        overshoot = 3

    def fill(token: str) -> str:
        if classes:
            return f'class="{token}"'
        if mono:
            tint = {"ocean-900": 0.85, "ocean-700": 0.55,
                    "ocean-500": 0.35, "ocean-300": 0.15}[token]
            return f'fill="{c["ink"]}" fill-opacity="{tint}"'
        return f'fill="{c[token]}"'

    ink = 'class="ink"' if classes else f'fill="{c["ink"]}"'
    w0, w1 = px - x0, x1 - px
    h0, h1 = py - x0, x1 - py
    parts = [
        f'<rect x="{x0}" y="{x0}" width="{w0}" height="{h0}" {fill("ocean-300")}/>',
        f'<rect x="{px}" y="{x0}" width="{w1}" height="{h0}" {fill("ocean-700")}/>',
        f'<rect x="{x0}" y="{py}" width="{w0}" height="{h1}" {fill("ocean-500")}/>',
        f'<rect x="{px}" y="{py}" width="{w1}" height="{h1}" {fill("ocean-900")}/>',
    ]
    # Dividing lines through P, drawn as gaps in the background colour.
    gap = 'class="gap"' if classes else f'stroke="{c["paper"]}"'
    parts.append(
        f'<path d="M{px} {x0}V{x1}M{x0} {py}H{x1}" {gap} stroke-width="1"/>'
    )
    # Cell outline, extended past the corners to suggest the grid.
    a, b = x0 - overshoot, x1 + overshoot
    stroke = 'class="line"' if classes else f'stroke="{c["ink"]}"'
    parts.append(
        f'<path d="M{a} {x0}H{b}M{a} {x1}H{b}M{x0} {a}V{b}M{x1} {a}V{b}" '
        f'{stroke} stroke-width="{sw}" fill="none"/>'
    )
    for x, y in ((x0, x0), (x1, x0), (x0, x1), (x1, x1)):
        parts.append(f'<circle cx="{x}" cy="{y}" r="{rn}" {ink}/>')
    halo_fill = 'class="halo"' if classes else f'fill="{c["paper"]}"'
    point = 'class="point"' if classes else (
        f'fill="{c["ink"]}"' if mono else f'fill="{c["point"]}"')
    parts.append(f'<circle cx="{px}" cy="{py}" r="{rp + halo}" {halo_fill}/>')
    parts.append(f'<circle cx="{px}" cy="{py}" r="{rp}" {point}/>')
    return "".join(parts)


def svg(view_box: str, body: str, title: str = "pyinterp") -> str:
    """Wrap SVG elements in a root element."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink='
        f'"http://www.w3.org/1999/xlink" viewBox="{view_box}" role="img">'
        f"<title>{title}</title>{body}</svg>\n"
    )


# --- Wordmark -------------------------------------------------------------------


class _RelPen(BasePen):
    """Pen writing a compact SVG path with relative commands."""

    def __init__(self, glyph_set) -> None:  # noqa: ANN001
        super().__init__(glyph_set)
        self.d: list[str] = []
        self.cur = (0, 0)
        self.start = (0, 0)

    @staticmethod
    def _n(v: float) -> str:
        return str(round(v))

    def _rel(self, *pts: tuple[float, float]) -> str:
        x0, y0 = self.cur
        out = []
        for x, y in pts:
            out.append(f"{self._n(x - x0)} {self._n(y - y0)}")
        self.cur = pts[-1]
        return " ".join(out).replace(" -", "-")

    def _moveTo(self, pt):  # noqa: N802
        self.d.append("m" + self._rel(pt))
        self.start = pt

    def _lineTo(self, pt):  # noqa: N802
        self.d.append("l" + self._rel(pt))

    def _qCurveToOne(self, p1, p2):  # noqa: N802
        x0, y0 = self.cur
        self.d.append(
            "q" + f"{self._n(p1[0] - x0)} {self._n(p1[1] - y0)} "
            f"{self._n(p2[0] - x0)} {self._n(p2[1] - y0)}".replace(" -", "-")
        )
        self.cur = p2

    def _curveToOne(self, p1, p2, p3):  # noqa: N802
        x0, y0 = self.cur
        vals = [p1, p2, p3]
        self.d.append(
            "c" + " ".join(
                f"{self._n(x - x0)} {self._n(y - y0)}" for x, y in vals
            ).replace(" -", "-")
        )
        self.cur = p3

    def _closePath(self):  # noqa: N802
        self.d.append("z")
        self.cur = self.start


class Wordmark:
    """Outline text set in Inter at the wordmark settings."""

    def __init__(self, path: pathlib.Path, axes: dict[str, float]) -> None:
        font = instancer.instantiateVariableFont(TTFont(path), axes)
        buf = io.BytesIO()
        font.save(buf)
        self.font = TTFont(io.BytesIO(buf.getvalue()))
        self.glyphs = self.font.getGlyphSet()
        self.upem = self.font["head"].unitsPerEm
        self.hb_font = hb.Font(hb.Face(buf.getvalue()))
        self.order = self.font.getGlyphOrder()

    def shape(self, text: str, tracking: float = 0.0):
        """Return (glyph name, x offset) pairs and the advance width."""
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb_font, buf, {"kern": True, "liga": False})
        x = 0
        out = []
        for i, (info, pos) in enumerate(zip(buf.glyph_infos, buf.glyph_positions)):
            out.append((self.order[info.codepoint], x + pos.x_offset))
            x += pos.x_advance
            if i < len(buf.glyph_infos) - 1:
                x += tracking * self.upem
        return out, x

    def outline(self, name: str) -> str:
        pen = _RelPen(self.glyphs)
        self.glyphs[name].draw(pen)
        return "".join(pen.d)

    def bounds(self, text: str) -> tuple[float, float]:
        """Return the (ymin, ymax) of the text, in font units."""
        from fontTools.pens.boundsPen import BoundsPen

        lo, hi = 0.0, 0.0
        for name, _ in self.shape(text)[0]:
            pen = BoundsPen(self.glyphs)
            self.glyphs[name].draw(pen)
            if pen.bounds:
                lo, hi = min(lo, pen.bounds[1]), max(hi, pen.bounds[3])
        return lo, hi

    def svg(self, runs: list[tuple[str, str]], x: float, baseline: float,
            size: float, prefix: str = "g") -> tuple[str, str, float]:
        """Return (defs, body, width) for text runs [(text, fill), ...].

        Each glyph is defined once and reused, to keep the files small.
        """
        scale = size / self.upem
        defs: dict[str, str] = {}
        uses = []
        text = "".join(t for t, _ in runs)
        glyphs, advance = self.shape(text, TRACKING)
        k = 0
        for chunk, fill in runs:
            group = []
            for _ in chunk:
                name, gx = glyphs[k]
                k += 1
                gid = f"{prefix}{len(defs)}" if name not in defs else defs[name]
                if name not in defs:
                    defs[name] = gid
                group.append(f'<use xlink:href="#{gid}" x="{round(gx)}"/>')
            uses.append(f'<g fill="{fill}">{"".join(group)}</g>')
        d = "".join(
            f'<path id="{gid}" d="{self.outline(name)}"/>'
            for name, gid in defs.items()
        )
        s = f"{scale:.6g}"
        body = (
            f'<g transform="matrix({s} 0 0 -{s} {x:g} {baseline:g})">'
            f'{"".join(uses)}</g>'
        )
        return d, body, advance * scale


# --- Files -------------------------------------------------------------------


def lockup(wm: Wordmark, c: dict[str, str]) -> str:
    """Horizontal lockup: mark, gap of 5 units, wordmark."""
    size = 17.0
    lo, hi = wm.bounds("pyinterp")
    scale = size / wm.upem
    # Centre the ink of the word on the centre of the cell (y = 12).
    baseline = 12 + (hi + lo) / 2 * scale
    x = 23 + 5
    defs, body, width = wm.svg(
        [("py", c["ink-muted"]), ("interp", c["ink"])], x, baseline, size
    )
    w = round(x + width + 1, 1)
    return svg(f"0 0 {w:g} 24", f"<defs>{defs}</defs>{mark(c)}{body}")


def favicon() -> str:
    style = "".join(
        [
            "<style>",
            *(
                f".{k}{{fill:{LIGHT[k]}}}"
                for k in ("ocean-900", "ocean-700", "ocean-500", "ocean-300",
                          "point", "ink")
            ),
            f".line{{stroke:{LIGHT['ink']}}}.halo{{fill:{LIGHT['paper']}}}",
            f".gap{{stroke:{LIGHT['paper']}}}",
            "@media (prefers-color-scheme:dark){",
            *(
                f".{k}{{fill:{DARK[k]}}}"
                for k in ("ocean-900", "ocean-700", "ocean-500", "ocean-300",
                          "point", "ink")
            ),
            f".line{{stroke:{DARK['ink']}}}.halo{{fill:{DARK['paper']}}}",
            f".gap{{stroke:{DARK['paper']}}}",
            "}</style>",
        ]
    )
    return svg("0 0 24 24", style + mark(LIGHT, favicon=True, classes=True))


def png(source: str, width: int, height: int, background: str | None = None
        ) -> bytes:
    data = cairosvg.svg2png(
        bytestring=source.encode(), output_width=width, output_height=height,
        background_color=background,
    )
    return data


def card(wm: Wordmark, tl: Wordmark, c: dict[str, str], width: int,
         height: int, tagline: bool = True) -> str:
    """Lockup (and tagline) centred on a paper background."""
    size = 17.0
    lo, hi = wm.bounds("pyinterp")
    scale = size / wm.upem
    baseline = 12 + (hi + lo) / 2 * scale
    defs, body, wwidth = wm.svg(
        [("py", c["ink-muted"]), ("interp", c["ink"])], 28, baseline, size
    )
    lock_w = 28 + wwidth
    k = width * 0.62 / lock_w
    tx = (width - lock_w * k) / 2
    ty = height * (0.40 if tagline else 0.5) - 12 * k
    parts = [
        f'<rect width="{width}" height="{height}" fill="{c["paper"]}"/>',
        f'<g transform="translate({tx:.1f} {ty:.1f}) scale({k:.4f})">'
        f"{mark(c)}{body}</g>",
    ]
    if tagline:
        tsize = 46
        tdefs, tbody, tw = tl.svg(
            [("Fast interpolation of geo-referenced data", c["ink-muted"])],
            0, 0, tsize, prefix="t",
        )
        defs += tdefs
        parts.append(
            f'<g transform="translate({(width - tw) / 2:.1f} '
            f'{ty + 24 * k + 90:.1f})">{tbody}</g>'
        )
        # Thin ramp rule along the bottom edge, one band per step.
        band = width / 4
        for i, tok in enumerate(("ocean-900", "ocean-700", "ocean-500",
                                 "ocean-300")):
            parts.append(
                f'<rect x="{i * band:g}" y="{height - 12}" width="{band:g}" '
                f'height="12" fill="{c[tok]}"/>'
            )
    return svg(f"0 0 {width} {height}", f"<defs>{defs}</defs>{''.join(parts)}")


def main() -> None:
    wm = Wordmark(pathlib.Path(sys.argv[1]), WORDMARK)
    files = {
        "pyinterp-light.svg": lockup(wm, LIGHT),
        "pyinterp-dark.svg": lockup(wm, DARK),
        "pyinterp-mark.svg": svg("0 0 24 24", mark(LIGHT)),
        "pyinterp-mark-dark.svg": svg("0 0 24 24", mark(DARK)),
        "favicon.svg": favicon(),
    }
    for name, text in files.items():
        STATIC.joinpath(name).write_text(text)
        print(f"{name:24s} {len(text.encode()):6d} bytes")

    # One-colour mark, documented in the brand sheet.
    HERE.joinpath("pyinterp-mark-mono.svg").write_text(
        svg("0 0 24 24", mark(LIGHT, mono=True))
    )
    HERE.joinpath("pyinterp-mark-mono-dark.svg").write_text(
        svg("0 0 24 24", mark(DARK, mono=True))
    )

    favicon_light = svg("0 0 24 24", mark(LIGHT, favicon=True))
    STATIC.joinpath("favicon-32.png").write_bytes(png(favicon_light, 32, 32))
    # The touch icon is opaque, with the mark inside a safe area.
    touch = svg(
        "0 0 180 180",
        f'<rect width="180" height="180" fill="{LIGHT["paper"]}"/>'
        f'<g transform="translate(30 30) scale(5)">{mark(LIGHT)}</g>',
    )
    STATIC.joinpath("apple-touch-icon.png").write_bytes(png(touch, 180, 180))
    STATIC.joinpath("social-card.png").write_bytes(
        png(card(wm, Wordmark(pathlib.Path(sys.argv[1]), TAGLINE), LIGHT,
                 1200, 630), 1200, 630)
    )
    thumb = svg(
        "0 0 400 280",
        f'<rect width="400" height="280" fill="{LIGHT["surface"]}"/>'
        f'<g transform="translate(104 44) scale(8)">{mark(LIGHT)}</g>',
    )
    STATIC.joinpath("gallery-thumb.png").write_bytes(png(thumb, 400, 280))

    # Quantise the PNGs: they only hold a handful of colours.
    for name in ("favicon-32.png", "apple-touch-icon.png", "social-card.png",
                 "gallery-thumb.png"):
        path = STATIC / name
        Image.open(path).save(path, optimize=True)
        print(f"{name:24s} {path.stat().st_size:6d} bytes")


if __name__ == "__main__":
    main()
