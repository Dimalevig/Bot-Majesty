# DayZ map: draft from the concept art

`source/concept.png` is the concept art. `tools/generate_terrain.py` turns it into source files for Terrain Builder (DayZ Tools).

## Generate

```bash
pip install numpy pillow scipy
python3 tools/generate_terrain.py                 # 2048 px, 15.36 km (7.5 m/cell)
python3 tools/generate_terrain.py --size 4096     # more detail (3.75 m/cell)
```

| File in `output/` | Purpose |
|---|---|
| `heightmap.asc` | relief in metres, imported into Terrain Builder |
| `heightmap.png` | the same relief as a 16-bit PNG (0 = -40 m, 65535 = 1000 m) |
| `satellite.png` | satellite texture (the art, 2× resolution) |
| `surface_mask.png` + `layers.cfg` | surface mask: forest, field, rock, snow, sand… |
| `preview_height.png` | shaded preview of the relief |

## Map layout
- Terrain is 15.36 × 15.36 km (same as Chernarus). The art spans the full width.
- North strip: forest (no buildings). South strip: open sea.
- East: snowy massif up to about 820 m. North-east: exclusion zone with the power plant.
- Centre: river with a dam. South: bay with islands and a port. South-west: airfield.

## Importing into Terrain Builder (Windows, DayZ Tools)
1. Create a P: drive in DayZ Tools, folder `P:\mymap\`.
2. Terrain Builder → New project → Mapframe: size 2048×2048, cell 7.5 m, easting 200000.
3. Mapframe properties → Terrain → Import `heightmap.asc`.
4. Satellite/Surface: put `satellite.png` and `surface_mask.png` in the project, copy `layers.cfg`, then Generate Layers.
5. Place objects (buildings, trees) by hand or from a generated object list.
6. Export WRP → `config.cpp` → pack with Addon Builder → test on a local server.

The masks come from automatic colour recognition, so they are rough. Fix them by hand in any image editor.
