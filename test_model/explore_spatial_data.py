#!/usr/bin/env python3
"""Create an initial interactive map and CSV from the CV2X export.

Reads channel_summary records from field_test_files and writes a point table plus
an HTML map into test_model/output. Road matching and road segmentation are
deliberately deferred until road centerline geometry is available.
"""

from __future__ import annotations

import csv
import argparse
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "field_test_files" / "segments" / "structured"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def records_from_file(path: Path):
    """Yield JSON values from a pretty-printed JSON-lines export file."""
    decoder = json.JSONDecoder()
    text = path.read_text(encoding="utf-8")
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos == len(text):
            break
        record, pos = decoder.raw_decode(text, pos)
        yield record


def collect_points():
    points = []
    for path in sorted(INPUT_DIR.glob("*.ndjson")):
        for record in records_from_file(path):
            if record.get("record_type") != "channel_summary":
                continue
            payload = record.get("payload", {})
            timestamp = payload.get("timestamp_unix_ms", record.get("timestamp_unix_ms"))
            tracks = payload.get("ue_locations") or payload.get("ue_tracks") or []
            for ue in tracks:
                lat, lon = ue.get("lat"), ue.get("lng")
                if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                    continue
                points.append({
                    "timestamp_unix_ms": timestamp,
                    "timestamp_utc": "" if timestamp is None else __import__("datetime").datetime.fromtimestamp(
                        timestamp / 1000, tz=__import__("datetime").timezone.utc
                    ).isoformat(),
                    "latitude": lat,
                    "longitude": lon,
                    "rsu_latitude": (payload.get("monitor_location") or {}).get("lat", ""),
                    "rsu_longitude": (payload.get("monitor_location") or {}).get("lng", ""),
                    "rsu_coverage_radius_m": (payload.get("monitor_location") or {}).get("coverage_radius_m", ""),
                    "track_id": ue.get("track_id", ""),
                    "display_id": ue.get("display_id", ""),
                    "reported_rssi_dbm": ue.get("rssi_dbm", payload.get("rssi_dbm")),
                    "reported_rsrp_dbm": ue.get("rsrp_dbm", payload.get("rsrp_dbm")),
                    "reported_heading_deg": ue.get("heading_deg", ""),
                    "freshness_ms": ue.get("freshness_ms", payload.get("message_freshness_ms")),
                    "estimated_cbr_percent": payload.get("estimated_cbr_percent"),
                    "sci_crc_failure_percent": payload.get("sci_crc_failure_percent"),
                    "tb_crc_failure_percent": payload.get("tb_crc_failure_percent"),
                    "observed_burst_count": payload.get("observed_burst_count"),
                    "bsm_message_count": payload.get("bsm_message_count"),
                })
    points.sort(key=lambda p: (p["timestamp_unix_ms"] or 0, p["track_id"]))
    segmented_path = OUTPUT_DIR / "location_points_segmented.csv"
    if segmented_path.exists():
        assignments = {}
        with segmented_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (int(row["timestamp_unix_ms"]), round(float(row["latitude"]), 8),
                       round(float(row["longitude"]), 8), row.get("track_id", ""))
                assignments[key] = row
        for point in points:
            key = (int(point["timestamp_unix_ms"]), round(point["latitude"], 8),
                   round(point["longitude"], 8), point["track_id"])
            assignment = assignments.get(key, {})
            for field in ("matched_segment_id", "matched_road_id", "centerline_offset_m", "along_segment_m", "match_status"):
                point[field] = assignment.get(field, "")
    return points


def write_csv(points):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "location_points.csv"
    if not points:
        raise SystemExit(f"No UE location points found under {INPUT_DIR}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    return path


def write_map(points):
    # Embed data as JSON; escape HTML-sensitive characters before inserting it.
    point_json = json.dumps(points, separators=(",", ":"), ensure_ascii=False)
    point_json = point_json.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    data_dir = Path(__file__).resolve().parent / "data"
    roads_path = OUTPUT_DIR / "road_segments.geojson"
    if not roads_path.exists():
        roads_path = data_dir / "road_centerlines.geojson"
    road_features = []
    if roads_path.exists():
        road_features = json.loads(roads_path.read_text(encoding="utf-8")).get("features", [])
    road_json = json.dumps(road_features, separators=(",", ":"), ensure_ascii=False)
    road_json = road_json.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    safe_title = html.escape("CV2X reported UE locations")
    output = OUTPUT_DIR / "location_map.html"
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title>
<style>
html,body{{height:100%;margin:0;font:14px system-ui,sans-serif;color:#18212b}} .layout{{height:100%;display:grid;grid-template-rows:auto 1fr}}
header{{padding:10px 16px;background:#fff;box-shadow:0 1px 5px #0002;z-index:500}} h1{{font-size:18px;margin:0 0 3px}} .note{{color:#52606d;font-size:12px}}
#map{{height:100%;min-height:400px}} .legend{{background:white;padding:8px 10px;border-radius:4px;box-shadow:0 1px 5px #0005;line-height:1.6}}
.swatch{{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px;vertical-align:-2px}}
@media(max-width:600px){{header{{padding:8px 10px}}}}
</style></head><body><div class="layout"><header><h1>{safe_title}</h1>
<div class="note" id="summary">Loading points…</div><div class="note">Color shows time from start (blue) to end (red). BSM locations are treated as ground truth for this experiment. The RSU marker is a working assumption at their arithmetic center; the 600 m circle is a reference radius. Google basemap provides visual context only.</div></header>
<div id="map"></div></div><script>
const points={point_json};
const roads={road_json};
const valid=points.filter(p=>Number.isFinite(p.latitude)&&Number.isFinite(p.longitude));
const start=Math.min(...valid.map(p=>p.timestamp_unix_ms||0));
const end=Math.max(...valid.map(p=>p.timestamp_unix_ms||0));
function color(t){{const x=end===start?0:(t-start)/(end-start);return `hsl(${{220-220*x}},78%,48%)`;}}
let map;
function initMap(){{
 const center=valid.length?{{lat:valid[0].latitude,lng:valid[0].longitude}}:{{lat:44.974,lng:-93.228}};
 map=new google.maps.Map(document.getElementById('map'),{{center,zoom:14,mapTypeControl:true,streetViewControl:false}});
 const bounds=new google.maps.LatLngBounds(); const coords=[]; const info=new google.maps.InfoWindow();
 valid.forEach((p,i)=>{{const t=p.timestamp_unix_ms||start;const pos={{lat:p.latitude,lng:p.longitude}};bounds.extend(pos);coords.push(pos);
  const detail=[['Time',p.timestamp_utc],['Track',p.display_id||p.track_id],['Mapped road segment',p.matched_segment_id],['Offset from centerline (m)',p.centerline_offset_m],['Match status',p.match_status],['Reported RSSI (dBm)',p.reported_rssi_dbm],['Reported RSRP (dBm)',p.reported_rsrp_dbm],['Heading (deg)',p.reported_heading_deg],['Freshness (ms)',p.freshness_ms],['CBR (%)',p.estimated_cbr_percent],['SCI CRC failure (%)',p.sci_crc_failure_percent],['TB CRC failure (%)',p.tb_crc_failure_percent],['Observed bursts',p.observed_burst_count],['BSM count',p.bsm_message_count]];
  const content=detail.filter(v=>v[1]!==null&&v[1]!==undefined&&v[1]!=='').map(v=>`<div><b>${{v[0]}}</b>: ${{v[1]}}</div>`).join('');
  const marker=new google.maps.Marker({{position:pos,map,clickable:true,title:`${{p.display_id||p.track_id}} · ${{p.timestamp_utc}}`,zIndex:10+i,icon:{{path:google.maps.SymbolPath.CIRCLE,scale:4,fillColor:color(t),fillOpacity:.9,strokeColor:'#fff',strokeWeight:1}}}});
  marker.addListener('click',()=>{{info.setContent(content);info.open({{map,anchor:marker}});}});
 }});
 const rsu=valid.length?{{lat:valid.reduce((a,p)=>a+p.latitude,0)/valid.length,lng:valid.reduce((a,p)=>a+p.longitude,0)/valid.length}}:null;
 const radius=Number(valid.find(p=>Number(p.rsu_coverage_radius_m))?.rsu_coverage_radius_m)||600;
 if(rsu){{new google.maps.Marker({{position:rsu,map,title:'Assumed RSU at reported-location centroid',label:{{text:'RSU*',color:'#fff',fontWeight:'700'}},icon:{{path:google.maps.SymbolPath.CIRCLE,scale:10,fillColor:'#6b21a8',fillOpacity:1,strokeColor:'#fff',strokeWeight:2}},zIndex:10000}});new google.maps.Circle({{map,center:rsu,radius,fillColor:'#7c3aed',fillOpacity:.12,strokeColor:'#6b21a8',strokeOpacity:.85,strokeWeight:2}});}}
 roads.forEach(feature=>{{
  const geom=feature.geometry; if(!geom)return;
  const parts=geom.type==='MultiLineString'?geom.coordinates:[geom.coordinates];
  const props=feature.properties||{{}};
  const label=props.street_name||[props.ST_PRE_DIR,props.ST_NAME,props.ST_POS_TYP,props.ST_POS_DIR].filter(Boolean).join(' ');
  parts.forEach(coords=>{{
   const colorIndex=Number(props.segment_color_index||0);const lineColor=`hsl(${{(colorIndex*47)%360}},68%,42%)`;
   const line=new google.maps.Polyline({{path:coords.map(c=>({{lat:c[1],lng:c[0]}})),geodesic:true,strokeColor:lineColor,strokeOpacity:1,strokeWeight:7,map,zIndex:4}});
   line.addListener('click',event=>{{const details=[['Street',label||'Road centerline'],['Subsegment ID',props.segment_id],['Source road ID',props.road_id||props.ROADSEG_ID],['Length (m)',props.segment_length_m],['Matched GPS samples',props.match_sample_count],['One-way',props.oneway||props.ONEWAY],['Lanes',props.NUM_LANES]].filter(x=>x[1]!==null&&x[1]!==undefined&&x[1]!=='').map(x=>`<div><b>${{x[0]}}</b>: ${{x[1]}}</div>`).join('');info.setContent(details);info.setPosition(event.latLng);info.open({{map}});}});
  }});
 }});
 if(valid.length>1){{new google.maps.Polyline({{path:coords,geodesic:true,strokeColor:'#263238',strokeOpacity:.55,strokeWeight:2,map,zIndex:2}});}}
 map.fitBounds(bounds,36);
 const legend=document.createElement('div');legend.className='legend';legend.innerHTML='<div><span class="swatch" style="background:hsl(220,78%,48%)"></span>Earlier GPS points</div><div><span class="swatch" style="background:hsl(0,78%,48%)"></span>Later GPS points</div><div><span class="swatch" style="background:#263238"></span>Time-ordered GPS path</div><div><span class="swatch" style="background:#087ea4"></span>Road subsegments (distinct colors)</div><div><span class="swatch" style="background:#6b21a8"></span>Assumed RSU centroid and reference radius</div>';map.controls[google.maps.ControlPosition.RIGHT_BOTTOM].push(legend);
}}
 const matched=valid.filter(p=>p.match_status==='matched');const assigned=new Set(matched.map(p=>p.matched_segment_id));
 document.getElementById('summary').textContent=`${{valid.length.toLocaleString()}} reported GPS samples · ${{new Set(valid.map(p=>p.track_id)).size}} track(s) · ${{matched.length.toLocaleString()}} matched to ${{assigned.size}} road subsegment(s) · ${{valid.length?new Date(start).toLocaleString():''}} to ${{valid.length?new Date(end).toLocaleString():''}}`;
window.initMap=initMap;
</script></body></html>'''
    output.write_text(page, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description="Create a map from location data with latitude and longitude columns.")
    parser.add_argument("--points-csv", type=Path, help="Optional normalized location CSV; defaults to the current CV2X export.")
    args = parser.parse_args()
    if args.points_csv:
        with args.points_csv.open(newline="", encoding="utf-8") as stream:
            points = list(csv.DictReader(stream))
        for point in points:
            point["latitude"] = float(point["latitude"])
            point["longitude"] = float(point["longitude"])
        csv_path = args.points_csv
    else:
        points = collect_points()
        csv_path = write_csv(points)
    map_path = write_map(points)
    print(f"Mapped {len(points):,} location samples.")
    print(f"Wrote {csv_path}")
    print(f"Wrote {map_path}")
    if points:
        print(f"Latitude range: {min(p['latitude'] for p in points):.7f} to {max(p['latitude'] for p in points):.7f}")
        print(f"Longitude range: {min(p['longitude'] for p in points):.7f} to {max(p['longitude'] for p in points):.7f}")
        print("Road matching fields are included when location_points_segmented.csv exists.")


if __name__ == "__main__":
    main()
