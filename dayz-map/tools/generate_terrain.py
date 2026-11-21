#!/usr/bin/env python3
"""
Generate a DayZ terrain draft from a concept image.

Outputs (in ../output):
  heightmap.png        16-bit grayscale heightmap (0..65535 -> MIN_H..MAX_H metres)
  heightmap.asc        ESRI ASCII grid for Terrain Builder import (metres)
  satellite.png        satellite texture (concept image fitted to the square terrain)
  surface_mask.png     surface layer mask (colours listed in layers.cfg)
  layers.cfg           Terrain Builder layer colour -> surface mapping
  preview_height.png   shaded-relief preview for checking the result

Usage:  python3 generate_terrain.py [--size 2048] [--terrain-m 15360]
"""
import argparse
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "source", "concept.png")
OUT = os.path.join(HERE, "..", "output")

MIN_H = -40.0     # deepest sea, metres
MAX_H = 1000.0    # keep peaks below this; see height range printed on run    # highest possible value in 16-bit encoding

# Surface mask colours (Terrain Builder "layers.cfg")
SURFACES = {
    "sea":    ((0, 0, 255),     "dz_sand"),        # under water, sand bottom
    "beach":  ((255, 255, 0),   "dz_sand"),
    "grass":  ((0, 255, 0),     "dz_grass"),
    "field":  ((200, 160, 60),  "dz_field"),
    "forest": ((0, 100, 0),     "dz_forest"),
    "zone":   ((150, 120, 80),  "dz_dry_grass"),   # northern exclusion zone
    "rock":   ((128, 128, 128), "dz_rock"),
    "snow":   ((255, 255, 255), "dz_snow"),
    "urban":  ((255, 0, 0),     "dz_concrete"),
}


def fbm(shape, seed, octaves=6, base=4):
    """Fractal value noise in [-1, 1]."""
    rng = np.random.default_rng(seed)
    h, w = shape
    out = np.zeros(shape)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = base * 2 ** o
        grid = rng.standard_normal((n + 1, int(n * w / h) + 1))
        layer = ndi.zoom(grid, (h / grid.shape[0], w / grid.shape[1]), order=3)[:h, :w]
        out += amp * layer
        total += amp
        amp *= 0.5
    out /= total
    return out / (np.abs(out).max() + 1e-9)


def classify(img):
    """Rough land-cover classification of the concept art."""
    img = ndi.uniform_filter(img.astype(float), size=(3, 3, 1))
    r, g, b = (img[..., i] for i in range(3))
    lum = (r + g + b) / 3
    sat = img.max(-1) - img.min(-1)

    water = (b - r > 14) & (b >= g - 2) & (lum < 90)
    water = ndi.binary_opening(water, iterations=1)
    water = ndi.binary_closing(water, iterations=2)
    lab, n = ndi.label(water)
    sizes = ndi.sum(water, lab, range(1, n + 1))
    water = np.isin(lab, 1 + np.where(sizes > 40)[0])


    snow = (lum > 120) & (sat < 45)
    snow = ndi.binary_opening(snow, iterations=1)

    urban = (sat < 25) & (lum > 70) & ~snow & ~water
    urban = ndi.binary_opening(urban, iterations=1)

    near_snow = ndi.gaussian_filter(snow.astype(float), 20) > 0.04
    hh, ww = lum.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    ne = (xx > 0.52 * ww) & (yy < 0.36 * hh)          # exclusion zone, north-east
    zone = ne & (r > g - 8) & (r > b + 12) & (lum > 45) & ~urban & ~near_snow
    field = (g > b + 12) & (lum >= 48) & ~urban & ~zone
    rock = near_snow & (sat < 25) & (lum > 40) & ~snow & ~urban
    forest = ~(water | snow | urban | zone | field | rock)
    clean = lambda m: ndi.binary_closing(ndi.binary_opening(m, iterations=1), iterations=1)
    return dict(water=water, snow=snow, urban=urban,
                forest=clean(forest), field=clean(field), zone=clean(zone), rock=clean(rock))


def build_heights(m, shape):
    h, w = shape
    sea, water = m["sea"], m["water"]

    k = w / 2048.0                       # distances below are tuned for 2048 px

    # 1. inland rise: ~2 m at the shore -> ~120 m deep inland
    d_sea = ndi.distance_transform_edt(~sea)
    land = 2 + 120 * (1 - np.exp(-d_sea / (300 * k)))

    # 2. rolling hills and ridges everywhere
    hills = fbm(shape, 1, octaves=6, base=3)
    ridged = 1 - np.abs(fbm(shape, 2, octaves=6, base=4))
    land += 70 * (hills + 0.3) + 55 * ridged ** 2 + 5 * fbm(shape, 3, 4, 32)

    # 3. snowy massif in the east, built around the snow / rock areas
    massif = ndi.gaussian_filter((m["snow"] | m["rock"]).astype(float), 70 * k)
    massif = massif / (massif.max() + 1e-9)
    peaks = ndi.gaussian_filter(m["snow"].astype(float), 25 * k)
    peaks /= peaks.max() + 1e-9
    land += 480 * massif ** 1.6 * (0.6 + 0.6 * ridged ** 3) + 140 * peaks

    # 4. flatten towns and fields so buildings sit nicely
    flat = ndi.gaussian_filter((m["urban"] | m["field"]).astype(float), 8 * k)
    smooth = ndi.gaussian_filter(land, 25 * k)
    land = land * (1 - 0.6 * flat) + smooth * 0.6 * flat

    # 5. river valleys and lakes
    inland_water = water & ~sea
    if inland_water.any():
        d_w = ndi.distance_transform_edt(~inland_water)
        valley = np.exp(-d_w / (45 * k))
        river_level = ndi.gaussian_filter(land, 60 * k) * 0.55
        land = land * (1 - valley) + np.minimum(land, river_level) * valley
        land[inland_water] = np.minimum(land[inland_water], river_level[inland_water] - 3)

    # 6. gentle shoreline + sea floor
    coast = 1 - np.exp(-d_sea / (80 * k))
    land = 1.0 + coast * (land - 1.0)
    d_land = ndi.distance_transform_edt(sea)
    sea_floor = -3 - 37 * (1 - np.exp(-d_land / (60 * k))) + 2 * fbm(shape, 4, 4, 8)
    heights = np.where(sea, sea_floor, land)

    return ndi.gaussian_filter(heights, 1.2)


def fit_square(arr, size, top, img_h, fill):
    """Place an image-space array into a square canvas (image fills full width)."""
    out = np.full((size, size) + arr.shape[2:], fill, dtype=arr.dtype)
    out[top:top + img_h] = arr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2048, help="heightmap resolution (px)")
    ap.add_argument("--terrain-m", type=float, default=15360.0, help="terrain size (m)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    src = Image.open(SRC).convert("RGB")
    size = args.size
    img_h = round(size * src.height / src.width)
    top = (size - img_h) // 2
    img = np.asarray(src.resize((size, img_h), Image.LANCZOS))

    print(f"classifying {size}x{img_h} ...")
    masks = classify(img)

    # canvas: north padding = mirrored forested hills, south padding = open sea
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[top:top + img_h] = img
    # north: blurred mirror tinted to forest, so no duplicated towns
    mirror = ndi.gaussian_filter(img[1:top + 1][::-1].astype(float), (6, 6, 0))
    canvas[:top] = (0.35 * mirror + 0.65 * np.array([34, 42, 26])).astype(np.uint8)
    canvas[top + img_h:] = (25, 45, 56)
    for k, v in masks.items():
        masks[k] = fit_square(v, size, top, img_h, k == "water")
    masks["water"][:top] = False
    masks["forest"][:top] = True
    # sea = water connected to the open sea south of the art
    lab, _ = ndi.label(masks["water"])
    south = np.unique(lab[-1]); south = south[south > 0]
    masks["sea"] = np.isin(lab, south)

    print("building heights ...")
    heights = build_heights(masks, (size, size))
    print(f"height range: {heights.min():.1f} .. {heights.max():.1f} m")

    # 16-bit PNG
    norm = np.clip((heights - MIN_H) / (MAX_H - MIN_H), 0, 1)
    Image.fromarray((norm * 65535).astype(np.uint16)).save(os.path.join(OUT, "heightmap.png"))

    # ESRI ASCII grid (Terrain Builder: File > Import > Terrain from .asc)
    cell = args.terrain_m / size
    with open(os.path.join(OUT, "heightmap.asc"), "w") as f:
        f.write(f"ncols {size}\nnrows {size}\nxllcorner 200000\nyllcorner 0\n"
                f"cellsize {cell}\nNODATA_value -9999\n")
        np.savetxt(f, heights, fmt="%.2f")

    # satellite texture
    Image.fromarray(canvas).resize((size * 2, size * 2), Image.LANCZOS).save(
        os.path.join(OUT, "satellite.png"))

    # surface mask (later layers override earlier ones)
    mask = np.zeros((size, size, 3), np.uint8)
    mask[:] = SURFACES["grass"][0]
    for name in ("field", "forest", "zone", "rock", "urban", "snow"):
        mask[masks[name]] = SURFACES[name][0]
    d_sea = ndi.distance_transform_edt(~masks["sea"])
    mask[(d_sea > 0) & (d_sea < 4)] = SURFACES["beach"][0]
    mask[masks["water"]] = SURFACES["sea"][0]
    Image.fromarray(mask).save(os.path.join(OUT, "surface_mask.png"))

    with open(os.path.join(OUT, "layers.cfg"), "w") as f:
        f.write("class Layers\n{\n")
        for name, (_, surf) in SURFACES.items():
            f.write(f"\tclass {name}\n\t{{\n\t\ttexture = \"\";\n"
                    f"\t\tmaterial = \"DZ\\surfaces\\data\\terrain\\{surf}.rvmat\";\n\t}};\n")
        f.write("};\n\nclass Legend\n{\n\tpicture = \"maskLegend.png\";\n\tclass Colors\n\t{\n")
        for name, (c, _) in SURFACES.items():
            f.write(f"\t\t{name}[] = {{{{{c[0]},{c[1]},{c[2]}}}}};\n")
        f.write("\t};\n};\n")

    # shaded relief preview
    gy, gx = np.gradient(heights)
    shade = np.clip(0.6 + (-gx + gy) * 0.08, 0, 1)
    col = np.interp(heights, [-40, 0, 1, 80, 250, 500, 800],
                    [0, 0, 1, 2, 3, 4, 5])
    pal = np.array([[20, 40, 90], [70, 130, 180], [190, 180, 120], [70, 130, 60],
                    [120, 140, 70], [130, 110, 90], [240, 240, 240]], float)
    lo = np.floor(col).astype(int).clip(0, 5)
    t = (col - lo)[..., None]
    rgb = pal[lo] * (1 - t) + pal[(lo + 1).clip(0, 6)] * t
    rgb *= shade[..., None]
    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).resize((1024, 1024)).save(
        os.path.join(OUT, "preview_height.png"))
    print("done ->", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
