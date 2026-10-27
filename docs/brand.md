# pyinterp brand sheet

## Concept

The mark is the textbook picture of bilinear interpolation: a point P inside a
grid cell splits it into four rectangles, and each corner's weight is the area
of the rectangle opposite to it. The four rectangles are four steps of one
ocean ramp, so the darkest tone sits on the largest area and the shades read
as weights.

![Mark](source/_static/pyinterp-mark.svg)

## Files

| File | Use |
|---|---|
| `source/_static/pyinterp-light.svg`, `pyinterp-dark.svg` | Horizontal lockup, documentation header and hero |
| `source/_static/pyinterp-mark.svg`, `pyinterp-mark-dark.svg` | Mark only, square |
| `source/_static/favicon.svg` | Favicon, switches with `prefers-color-scheme` |
| `source/_static/favicon-32.png`, `apple-touch-icon.png` | Raster fallbacks (light) |
| `source/_static/social-card.png` | 1200 × 630, for `og:image` |
| `source/_static/gallery-thumb.png` | sphinx-gallery default thumbnail |
| `source/_static/pyinterp.css` | Tokens and theme mapping |
| `brand/pyinterp-mark-mono.svg`, `pyinterp-mark-mono-dark.svg` | One-colour mark (`ink` tints) |
| `brand/build_assets.py` | Regenerates all of the above |

Regenerate with `python docs/brand/build_assets.py "Inter[opsz,wght].ttf"`
(needs fonttools, uharfbuzz, cairosvg and pillow; Inter comes from
<https://github.com/google/fonts/tree/main/ofl/inter>).

## Geometry

Drawn on a 24-unit grid. The cell spans (4, 4) to (20, 20), and P sits at
(9, 10). The rectangles measure 5 × 6, 11 × 6, 5 × 10 and 11 × 10 (areas 30,
66, 50 and 110). Grid lines are 1.25 units wide and extend 3 units past the
corners. The nodes have a radius of 1.5. P has a radius of 1.75 inside a
background-coloured halo of 0.75. The dividing lines are 1-unit gaps in the
background colour.

The favicon uses a larger cell, (2, 2) to (22, 22) with P at (8, 9), and drops
the overshoots so that it stays legible at 16 px.

## Colour

| Token | Role | Light | Dark |
|---|---|---|---|
| `ink` | text, outlines, nodes | `#0B2540` | `#E6EEF5` |
| `ink-muted` | secondary text, the `py` of the wordmark | `#4F6680` | `#9DB0C3` |
| `ocean-900` | largest rectangle, link hover, inline code | `#0B4F7C` | `#A3D9F7` |
| `ocean-700` | primary: links, buttons, notes | `#0E6BA8` | `#5DB3E8` |
| `ocean-500` | third rectangle | `#2E9CCA` | `#2F82B8` |
| `ocean-300` | smallest rectangle, title rule | `#9ED3EC` | `#1B4B6C` |
| `point` | the point P, warnings, focus ring; never text | `#D9700F` | `#FFA54F` |
| `paper` | page background | `#FFFFFF` | `#0F1720` |
| `surface` | code blocks, cards | `#F3F6F9` | `#16212C` |
| `teal` | tips (derived from the ramp) | `#10746D` | `#5CC8B8` |

Compared with the starting brief, light `point` was darkened from `#F28C28`
(2.45:1) so that it reaches 3:1 on white. The dark ramp was spread out
(`ocean-900` `#7CC4F0` → `#A3D9F7`, `ocean-500` `#3A8FC4` → `#2F82B8`,
`ocean-300` `#1F5A80` → `#1B4B6C`) because its two lightest steps were only
6 L* apart.

### Contrast (WCAG 2.2)

| Pair | Light | Dark | Needed |
|---|---|---|---|
| `ink` on `paper` / `surface` | 15.52 / 14.31 | 15.39 / 13.91 | 4.5 |
| `ink-muted` on `paper` / `surface` | 5.92 / 5.46 | 8.10 / 7.32 | 4.5 |
| `ocean-700` (links) on `paper` / `surface` | 5.70 / 5.25 | 7.80 / 7.05 | 4.5 |
| `paper` on `ocean-700` (filled button) | 5.70 | 7.80 | 4.5 |
| `teal` on `paper` | 5.61 | 8.94 | 4.5 |
| `ink` on the admonition title tints | ≥ 13.3 | ≥ 11.2 | 4.5 |
| `point` on `paper` (graphics) | 3.35 | 9.23 | 3 |
| `point` icon on its warning tint | 3.06 | 7.25 | 3 |

On the rendered pages (home, an API page and the gallery, in both modes),
axe-core 4.10 reports no `color-contrast` violations.

### Ramp legibility

The ramp's order carries the meaning of the mark, so each step stays apart in
lightness (CIE L*), including under simulated colour-vision deficiencies
(Machado 2009, full severity):

| | Normal | Protan | Deutan | Tritan |
|---|---|---|---|---|
| Light: 900 / 700 / 500 / 300 | 32 / 43 / 60 / 82 | 35 / 47 / 64 / 84 | 30 / 42 / 58 / 80 | 34 / 46 / 63 / 82 |
| Dark: 900 / 700 / 500 / 300 | 84 / 70 / 52 / 30 | 86 / 73 / 55 / 32 | 83 / 68 / 50 / 29 | 84 / 71 / 54 / 32 |

The smallest step is 11 L* in light mode and 13 in dark mode. In dark mode the
ramp is inverted, so the largest rectangle is the lightest one and still
stands out.

## Type

- **Wordmark**: `pyinterp`, always lowercase. Inter SemiBold (`wght` 600,
  `opsz` 32) with −0.02 em tracking and the font's kerning, outlined in the
  SVG files. `py` is set in `ink-muted` and `interp` in `ink`.
- **Headings**: Inter (vendored Latin subset, weights 500–700, SIL OFL).
- **Body**: the theme's system font stack.
- **Code**: JetBrains Mono (vendored Latin subset, weights 400–700, SIL OFL),
  with the theme's accessible Pygments styles (`a11y-high-contrast-light` and
  `a11y-high-contrast-dark`).

## Lockup, clear space and minimum sizes

- **Horizontal lockup**: the mark on the left and the wordmark on the right,
  centred on the cell. The gap is the width of the smallest rectangle (5
  units).
- **Clear space**: half the mark's height on every side.
- **Minimum sizes**: the lockup at 24 px high, and the mark at 24 px. Below
  that, down to 16 px, use `favicon.svg`.

## Do

- Use the dark files (or `favicon.svg`) on dark backgrounds.
- Use the one-colour mark when only one ink is available (print, embossing).
- Keep the largest rectangle as the strongest tone.

## Don't

- Recolour the rectangles with unrelated hues, or reorder the ramp.
- Move P to the centre: the four areas must stay different.
- Set the name as "PyInterp" or "PYINTERP", or retype it in another font.
- Add shadows, gradients, 3-D effects or a globe.
- Use `point` for text.

## Social card

`sphinxext-opengraph` is not a dependency yet. If it is accepted, add it to the
documentation requirements and set the following in `conf.py`:

```python
extensions.append("sphinxext.opengraph")
ogp_site_url = "https://cnes.github.io/pangeo-pyinterp/"
ogp_image = "_static/social-card.png"
ogp_description_length = 160
```

## Previews

`brand/previews/` holds the home page, the `regular_grid` API page and the
gallery, rendered before and after, in light and dark, at 1440 px and 390 px.
These are for review only and do not need to be committed.
