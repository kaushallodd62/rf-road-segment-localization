#!/usr/bin/env python3
"""Split local road centerlines and map GPS samples to the resulting segments.

Input: WGS84 GeoJSON LineString/MultiLineString and CSV with latitude/longitude.
The script is independent of street names and retains all input point columns.
"""

from __future__ import annotations

import csv
import json
import math
import argparse
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "output"
INPUT_ROADS = DATA_DIR / "road_centerlines.geojson"
INPUT_POINTS = OUTPUT_DIR / "location_points.csv"
OUTPUT_SEGMENTS = OUTPUT_DIR / "road_segments.geojson"
OUTPUT_POINTS = OUTPUT_DIR / "location_points_segmented.csv"
EARTH_RADIUS_M = 6_371_000.0


def local_xy(lon: float, lat: float, lon0: float, lat0: float) -> tuple[float, float]:
    """Project this small local area to meters using an equirectangular tangent plane."""
    x = math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def split_line(coords, segment_length_m, lon0, lat0):
    """Split one LineString at segment_length_m, retaining all original vertices."""
    if len(coords) < 2:
        return []
    pieces = []
    current = [coords[0]]
    current_length = 0.0
    target = segment_length_m

    for start, end in zip(coords, coords[1:]):
        sx, sy = local_xy(start[0], start[1], lon0, lat0)
        ex, ey = local_xy(end[0], end[1], lon0, lat0)
        edge_len = math.hypot(ex - sx, ey - sy)
        if edge_len == 0:
            continue
        consumed = 0.0
        while edge_len - consumed >= target - current_length - 1e-9:
            take = max(0.0, target - current_length)
            consumed += take
            ratio = min(1.0, consumed / edge_len)
            cut = [start[0] + ratio * (end[0] - start[0]), start[1] + ratio * (end[1] - start[1])]
            current.append(cut)
            pieces.append(current)
            current = [cut]
            current_length = 0.0
            if edge_len - consumed <= 1e-9:
                break
        remaining = edge_len - consumed
        if remaining > 1e-9:
            current.append(end)
            current_length += remaining

    if len(current) >= 2 and current_length > 1e-6:
        pieces.append(current)
    return pieces


def point_to_line(point_xy, coords, lon0, lat0):
    """Return nearest distance and along-line distance for a projected point."""
    best_distance = math.inf
    best_along = 0.0
    along = 0.0
    for start, end in zip(coords, coords[1:]):
        a = local_xy(start[0], start[1], lon0, lat0)
        b = local_xy(end[0], end[1], lon0, lat0)
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        t = max(0.0, min(1.0, ((point_xy[0] - a[0]) * dx + (point_xy[1] - a[1]) * dy) / (length * length)))
        nearest = (a[0] + t * dx, a[1] + t * dy)
        d = distance(point_xy, nearest)
        if d < best_distance:
            best_distance = d
            best_along = along + t * length
        along += length
    return best_distance, best_along


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centerlines", type=Path, default=INPUT_ROADS, help="Local WGS84 road centerline GeoJSON")
    parser.add_argument("--points", type=Path, default=INPUT_POINTS, help="CSV containing latitude and longitude columns")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--segment-length-m", type=float, default=200.0)
    parser.add_argument("--max-match-distance-m", type=float, default=20.0)
    parser.add_argument("--latitude-field", default="latitude")
    parser.add_argument("--longitude-field", default="longitude")
    args = parser.parse_args()
    if args.segment_length_m <= 0 or args.max_match_distance_m < 0:
        parser.error("segment length must be positive and match distance nonnegative")

    roads_data = json.loads(args.centerlines.read_text(encoding="utf-8"))
    with args.points.open(newline="", encoding="utf-8") as stream:
        points = list(csv.DictReader(stream))
    if not points:
        raise SystemExit("No point rows found; regenerate the map data first.")

    for row in points:
        row["latitude"] = float(row[args.latitude_field])
        row["longitude"] = float(row[args.longitude_field])
    lat0 = sum(float(p["latitude"]) for p in points) / len(points)
    lon0 = sum(float(p["longitude"]) for p in points) / len(points)

    segments = []
    for feature in roads_data.get("features", []):
        geometry = feature.get("geometry") or {}
        parts = geometry.get("coordinates", [])
        if geometry.get("type") == "LineString":
            parts = [parts]
        elif geometry.get("type") != "MultiLineString":
            continue
        props = feature.get("properties", {})
        road_id = str(next((props[k] for k in ("road_id", "ROADSEG_ID", "roadseg_id", "OBJECTID", "objectid", "GlobalID", "globalid", "id") if props.get(k) is not None), f"feature-{len(segments)}"))
        base_name = str(props.get("street_name") or props.get("FULLNAME") or props.get("ST_NAME") or "")
        for part_index, coords in enumerate(parts):
            for sub_index, piece in enumerate(split_line(coords, args.segment_length_m, lon0, lat0)):
                length = sum(distance(local_xy(a[0], a[1], lon0, lat0), local_xy(b[0], b[1], lon0, lat0)) for a, b in zip(piece, piece[1:]))
                segment_id = f"{road_id}:p{part_index}:s{sub_index:02d}"
                segments.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": piece},
                    "properties": {
                        "segment_id": segment_id,
                        "road_id": road_id,
                        "segment_index": sub_index,
                        "source_part_index": part_index,
                        "segment_color_index": len(segments),
                        "street_name": base_name,
                        "segment_length_m": round(length, 2),
                        "oneway": props.get("oneway", props.get("ONEWAY")),
                        "match_sample_count": 0,
                    },
                })

    if not segments:
        raise SystemExit("No centerline geometries found in the downloaded GeoJSON.")

    segment_counts = {s["properties"]["segment_id"]: 0 for s in segments}
    for row in points:
        lat, lon = float(row["latitude"]), float(row["longitude"])
        pxy = local_xy(lon, lat, lon0, lat0)
        best = None
        for segment in segments:
            d, along = point_to_line(pxy, segment["geometry"]["coordinates"], lon0, lat0)
            if best is None or d < best[0]:
                best = (d, along, segment)
        d, along, segment = best
        props = segment["properties"]
        if d <= args.max_match_distance_m:
            row["matched_segment_id"] = props["segment_id"]
            row["matched_road_id"] = props["road_id"]
            row["centerline_offset_m"] = round(d, 2)
            row["along_segment_m"] = round(along, 2)
            row["match_status"] = "matched"
            segment_counts[props["segment_id"]] += 1
        else:
            row["matched_segment_id"] = ""
            row["matched_road_id"] = ""
            row["centerline_offset_m"] = round(d, 2)
            row["along_segment_m"] = ""
            row["match_status"] = f"unmatched_over_{args.max_match_distance_m:g}m"

    for segment in segments:
        segment["properties"]["match_sample_count"] = segment_counts[segment["properties"]["segment_id"]]

    output_segments = args.output_dir / "road_segments.geojson"
    output_points = args.output_dir / "location_points_segmented.csv"
    output_segments.parent.mkdir(parents=True, exist_ok=True)
    collection = {"type": "FeatureCollection", "features": segments}
    output_segments.write_text(json.dumps(collection, separators=(",", ":")), encoding="utf-8")
    fieldnames = list(points[0])
    with output_points.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)

    matched = sum(r["match_status"] == "matched" for r in points)
    active = sum(v > 0 for v in segment_counts.values())
    offsets = sorted(float(r["centerline_offset_m"]) for r in points)
    print(f"Road pieces: {len(segments)} at up to {args.segment_length_m:.0f} m, split within source road features.")
    print(f"Matched points: {matched:,}/{len(points):,}; unmatched: {len(points)-matched:,} (limit {args.max_match_distance_m:.0f} m).")
    print(f"Segments with samples: {active}; median offset: {offsets[len(offsets)//2]:.2f} m; max offset: {offsets[-1]:.2f} m.")
    print(f"Wrote {output_segments}")
    print(f"Wrote {output_points}")


if __name__ == "__main__":
    main()
