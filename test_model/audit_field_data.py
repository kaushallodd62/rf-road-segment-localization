#!/usr/bin/env python3
"""Build a self-contained exploratory audit page from the field-test export."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STRUCTURED = ROOT / "field_test_files" / "segments" / "structured"
DIAGNOSTICS = ROOT / "field_test_files" / "segments" / "diagnostics"
POINTS = HERE / "output" / "location_points_segmented.csv"
SEGMENTS = HERE / "output" / "road_segments.geojson"
CONFIG = ROOT / "field_test_files" / "configs" / "effective" / "cfg-fa7120bd36bf8370d24fe74e90011423d77cb854a87095aac7c7719c89d8ebfc.json"
OUT = HERE / "output" / "data_audit.html"
RADIUS = 6_371_000.0

FEATURES = [
    ("rsrp_dbm", "RSRP", "dBm"),
    ("rssi_dbm", "RSSI", "dBm"),
    ("burst_power_dbm", "Burst power", "dBm"),
    ("estimated_cbr_percent", "Estimated CBR", "%"),
    ("frame_busy_ratio_percent", "Frame busy ratio", "%"),
    ("message_freshness_ms", "Message freshness", "ms"),
    ("rx_gain_db", "Receiver gain", "dB"),
    ("sci_crc_failure_percent", "SCI CRC failure", "%"),
    ("tb_crc_failure_percent", "TB CRC failure", "%"),
    ("bsm_rate_per_sec", "BSM rate", "messages/s"),
]


def records(path):
    decoder = json.JSONDecoder()
    source = path.read_text(encoding="utf-8")
    pos = 0
    while pos < len(source):
        while pos < len(source) and source[pos].isspace():
            pos += 1
        if pos >= len(source):
            break
        value, pos = decoder.raw_decode(source, pos)
        yield value


def finite(value):
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def quantile(values, p):
    values = sorted(values)
    return values[round((len(values) - 1) * p)] if values else None


def distance_m(lat1, lon1, lat2, lon2):
    a1, a2 = math.radians(lat1), math.radians(lat2)
    da, do = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = math.sin(da / 2) ** 2 + math.cos(a1) * math.cos(a2) * math.sin(do / 2) ** 2
    return 2 * RADIUS * math.asin(math.sqrt(h))


def pearson(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    mx = statistics.mean(x for x, _ in pairs)
    my = statistics.mean(y for _, y in pairs)
    dx = [x - mx for x, _ in pairs]
    dy = [y - my for _, y in pairs]
    den = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / den if den else 0.0


def make_data():
    summaries, bursts = [], []
    for path in sorted(STRUCTURED.glob("*.ndjson")):
        for row in records(path):
            if row.get("record_type") == "channel_summary":
                summaries.append(row)
            elif row.get("record_type") == "burst":
                bursts.append(row)
    summaries.sort(key=lambda x: x.get("timestamp_unix_ms", 0))
    bursts.sort(key=lambda x: x.get("timestamp_unix_ms", 0))

    with POINTS.open(newline="", encoding="utf-8") as stream:
        points = list(csv.DictReader(stream))
    points_by_time = {int(p["timestamp_unix_ms"]): p for p in points}
    config = json.loads(CONFIG.read_text(encoding="utf-8"))["content"]["effective_config"]
    valid_locations = [(finite(p.get("latitude")), finite(p.get("longitude"))) for p in points]
    valid_locations = [(lat, lon) for lat, lon in valid_locations if lat is not None and lon is not None]
    assumed_rsu = {
        "lat": statistics.mean(lat for lat, _ in valid_locations),
        "lon": statistics.mean(lon for _, lon in valid_locations),
        "radius": config["monitor_location"]["coverage_radius_m"],
    }

    features = {}
    for key, label, unit in FEATURES:
        values = [finite(row["payload"].get(key)) for row in summaries]
        nums = [x for x in values if x is not None]
        features[key] = {
            "label": label, "unit": unit, "values": values, "n": len(nums),
            "missing": len(values) - len(nums), "min": min(nums) if nums else None,
            "max": max(nums) if nums else None, "mean": statistics.mean(nums) if nums else None,
            "sd": statistics.pstdev(nums) if len(nums) > 1 else 0,
            "p05": quantile(nums, .05), "q25": quantile(nums, .25),
            "median": quantile(nums, .5), "q75": quantile(nums, .75),
            "p95": quantile(nums, .95),
        }

    road_features = json.loads(SEGMENTS.read_text(encoding="utf-8"))["features"]
    centers = {}
    for f in road_features:
        coords = f["geometry"]["coordinates"]
        centers[f["properties"]["segment_id"]] = (
            sum(c[1] for c in coords) / len(coords),
            sum(c[0] for c in coords) / len(coords),
        )
    ordered_ids = sorted(centers, key=lambda sid: (-centers[sid][0], centers[sid][1]))
    names = {sid: "S%02d" % (i + 1) for i, sid in enumerate(ordered_ids)}
    segment_rows = []
    for f in road_features:
        p = f["properties"]
        sid = p["segment_id"]
        segment_rows.append({
            "label": names[sid], "id": sid, "length": p.get("segment_length_m"),
            "count": int(p.get("match_sample_count", 0) or 0),
            "lat": centers[sid][0], "lon": centers[sid][1],
        })
    segment_rows.sort(key=lambda x: x["label"])

    series, distances = [], []
    for row in summaries:
        ts = row.get("timestamp_unix_ms", 0)
        p = row["payload"]
        locs = p.get("ue_locations") or p.get("ue_tracks") or []
        loc = locs[0] if locs else {}
        lat, lon = finite(loc.get("lat")), finite(loc.get("lng"))
        dist = distance_m(assumed_rsu["lat"], assumed_rsu["lon"], lat, lon) if lat is not None and lon is not None else None
        if dist is not None:
            distances.append(dist)
        point = points_by_time.get(ts, {})
        sid = point.get("matched_segment_id", "")
        s = {
            "t": (ts - summaries[0]["timestamp_unix_ms"]) / 60000,
            "utc": datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%H:%M:%S"),
            "segment": names.get(sid, "unmatched"), "lat": lat, "lon": lon,
            "distance": dist, "offset": finite(point.get("centerline_offset_m")),
            "heading": finite(point.get("reported_heading_deg")),
        }
        for key, _, _ in FEATURES:
            s[key] = finite(p.get(key))
        series.append(s)

    # Estimate each end-to-end traversal from BSM heading reversals.
    passes = []
    for row in series:
        heading = row["heading"]
        if heading is None:
            continue
        heading %= 360
        direction = "N" if min(heading, 360 - heading) < 90 else "S"
        if not passes or passes[-1]["direction"] != direction:
            passes.append({"direction": direction, "start": row, "end": row, "count": 0, "segment_counts": {}})
        passes[-1]["end"] = row
        passes[-1]["count"] += 1
        passes[-1]["segment_counts"][row["segment"]] = passes[-1]["segment_counts"].get(row["segment"], 0) + 1
    for traversal in passes:
        start, end = traversal["start"], traversal["end"]
        duration_s = max(0.0, (end["t"] - start["t"]) * 60)
        endpoint_distance_m = abs(end["lat"] - start["lat"]) * 111_000 if start["lat"] is not None and end["lat"] is not None else None
        traversal.update({
            "start_utc": start["utc"], "end_utc": end["utc"], "duration_s": duration_s,
            "distance_m": endpoint_distance_m,
            "avg_speed_kmh": endpoint_distance_m / duration_s * 3.6 if endpoint_distance_m is not None and duration_s > 0 else None,
        })
    pass_segment_rows = []
    for segment in segment_rows:
        row = {"label": segment["label"], "length": segment["length"], "total": segment["count"]}
        for direction in ("N", "S"):
            per_pass = [p["segment_counts"].get(segment["label"], 0) for p in passes if p["direction"] == direction]
            visited = [n for n in per_pass if n > 0]
            row[direction.lower() + "_visits"] = len(visited)
            row[direction.lower() + "_mean_s"] = statistics.mean(visited) if visited else None
            row[direction.lower() + "_speed_kmh"] = segment["length"] / row[direction.lower() + "_mean_s"] * 3.6 if visited and segment["length"] else None
        pass_segment_rows.append(row)

    rng = random.Random(31415)
    point_samples = []
    for p in rng.sample(points, min(12, len(points))):
        ts = int(p["timestamp_unix_ms"])
        point_samples.append({
            "utc": datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%H:%M:%S"),
            "lat": finite(p.get("latitude")), "lon": finite(p.get("longitude")),
            "segment": names.get(p.get("matched_segment_id", ""), "unmatched"),
            "offset": finite(p.get("centerline_offset_m")),
            "rsrp": finite(p.get("reported_rsrp_dbm")), "rssi": finite(p.get("reported_rssi_dbm")),
            "cbr": finite(p.get("estimated_cbr_percent")), "bsm": finite(p.get("bsm_message_count")),
        })
    burst_samples = []
    for row in rng.sample(bursts, min(12, len(bursts))):
        b = row["payload"].get("burst", {})
        phy, sci = b.get("phy", {}), b.get("sci", {})
        doppler = phy.get("doppler", {})
        burst_samples.append({
            "utc": datetime.fromtimestamp(row.get("timestamp_unix_ms", 0) / 1000, timezone.utc).strftime("%H:%M:%S.%f")[:-3],
            "power": finite(phy.get("burst_power_dbm")), "rsrp": finite(phy.get("rsrp_dbm")),
            "rssi": finite(phy.get("rssi_dbm")), "doppler": doppler.get("available"),
            "doppler_hz": finite(doppler.get("magnitude_hz")), "sci": b.get("sci_crc_state"),
            "tb": b.get("tb_crc_state"), "mcs": sci.get("mcs"), "attribution": b.get("ue_attribution_status"),
        })

    correlation_keys = ["rsrp_dbm", "rssi_dbm", "burst_power_dbm", "estimated_cbr_percent",
                        "frame_busy_ratio_percent", "rx_gain_db", "message_freshness_ms", "bsm_rate_per_sec"]
    correlation = {}
    for a in correlation_keys:
        correlation[a] = {}
        for b in correlation_keys:
            correlation[a][b] = pearson(features[a]["values"], features[b]["values"])

    state_counts = {}
    for key in ("sci_crc_state", "tb_crc_state", "ue_attribution_status", "upper_layer_outcome"):
        counts = {}
        for row in bursts:
            value = row["payload"]["burst"].get(key, "(missing)")
            counts[value] = counts.get(value, 0) + 1
        state_counts[key] = counts
    burst_availability = {
        "Burst power": sum(row["payload"]["burst"].get("phy", {}).get("burst_power_dbm") is not None for row in bursts),
        "RSRP": sum(row["payload"]["burst"].get("phy", {}).get("rsrp_dbm") is not None for row in bursts),
        "RSSI": sum(row["payload"]["burst"].get("phy", {}).get("rssi_dbm") is not None for row in bursts),
        "Doppler": sum(bool(row["payload"]["burst"].get("phy", {}).get("doppler", {}).get("available")) for row in bursts),
        "Channel response": sum(bool(row["payload"]["burst"].get("phy", {}).get("channel_response_available")) for row in bursts),
    }

    manifest = json.loads((ROOT / "field_test_files" / "export_manifest.json").read_text(encoding="utf-8"))
    intervals = [(b["timestamp_unix_ms"] - a["timestamp_unix_ms"]) / 1000 for a, b in zip(summaries, summaries[1:])]
    return {
        "features": features, "feature_order": [x[0] for x in FEATURES],
        "series": series, "segments": segment_rows, "passes": passes,
        "pass_segment_rows": pass_segment_rows, "samples": point_samples,
        "burst_samples": burst_samples, "correlation": correlation, "correlation_keys": correlation_keys,
        "counts": {
            "summaries": len(summaries), "bursts": len(bursts),
            "diagnostics": sum(1 for path in DIAGNOSTICS.glob("*.ndjson") for _ in records(path)),
            "tracks": len({t.get("track_id") for r in summaries for t in r["payload"].get("ue_locations", [])}),
            "runs": len(manifest.get("decoder_run_ids", [])),
            "active_segments": sum(s["count"] > 0 for s in segment_rows),
        },
        "duration": (summaries[-1]["timestamp_unix_ms"] - summaries[0]["timestamp_unix_ms"]) / 60000,
        "start": datetime.fromtimestamp(summaries[0]["timestamp_unix_ms"] / 1000, timezone.utc).isoformat(),
        "end": datetime.fromtimestamp(summaries[-1]["timestamp_unix_ms"] / 1000, timezone.utc).isoformat(),
        "median_interval": statistics.median(intervals), "gaps_over_2s": sum(x > 2 for x in intervals),
        "rsu": assumed_rsu,
        "rsu_distances": distances,
        "within_radius": sum(x <= assumed_rsu["radius"] for x in distances),
        "distance_rsrp_correlation": pearson([s["distance"] for s in series], [s["rsrp_dbm"] for s in series]),
        "offsets": [finite(p.get("centerline_offset_m")) for p in points],
        "states": state_counts,
        "burst_availability": burst_availability,
        "doppler_available": sum(bool(r["payload"]["burst"].get("phy", {}).get("doppler", {}).get("available")) for r in bursts),
        "channel_available": sum(bool(r["payload"]["burst"].get("phy", {}).get("channel_response_available")) for r in bursts),
        "dropped": sum(int(r["payload"].get("analyzer_dropped_event_count", 0) or 0) for r in summaries),
        "late": sum(int(r["payload"].get("analyzer_late_event_count", 0) or 0) for r in summaries),
        "artifact_policy": config["storage"].get("artifacts", {}),
    }


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Field Data Audit</title><style>
:root{color-scheme:light dark;--bg:#f7f8fa;--fg:#18212b;--muted:#5b6673;--line:#d6dbe1;--grid:#e7eaee;--blue:#1875c8;--orange:#d97706;--green:#16845b;--purple:#7c5bc7;--red:#c74242;--surface:#fff;--warn:#fff1cc;--warnfg:#653b00}
@media(prefers-color-scheme:dark){:root{--bg:#15191e;--fg:#edf1f5;--muted:#aab4c0;--line:#414a55;--grid:#2b333d;--blue:#62b1ff;--orange:#ffb454;--green:#5dd3a0;--purple:#b8a0ff;--red:#ff8585;--surface:#1d232a;--warn:#493610;--warnfg:#ffe3a1}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,sans-serif}main{max-width:1180px;margin:auto;padding:24px 20px 60px}h1{font-size:24px;margin:0 0 4px;font-weight:650}h2{font-size:18px;margin:30px 0 6px;font-weight:600}h3{font-size:14px;margin:16px 0 4px;font-weight:600}.sub,.note,.small{color:var(--muted)}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:20px 0}.stat{background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:12px}.stat b{display:block;font-size:20px;font-variant-numeric:tabular-nums}.stat span,.small{font-size:12px}.warning{padding:12px 14px;border-radius:8px;background:var(--warn);color:var(--warnfg);margin:12px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:18px 28px}.chart{min-width:0}.chart svg,.full svg{width:100%;height:auto}.axis{stroke:var(--line);stroke-width:1}.gridline{stroke:var(--grid);stroke-width:1}.tick{fill:var(--muted);font-size:11px}.label{fill:var(--fg);font-size:12px}.table-wrap{overflow-x:auto;margin:10px 0 18px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{color:var(--muted);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}.mono{font:11px ui-monospace,SFMono-Regular,monospace}.heat td{text-align:center;padding:5px 3px;min-width:45px}.foot{border-top:1px solid var(--line);padding-top:14px;margin-top:25px;color:var(--muted);font-size:12px}
@media(max-width:560px){main{padding:16px 12px 40px}.grid{display:block}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body><main>
<h1>Field data audit</h1><div class="sub">Descriptive exploration of the current C‑V2X capture; not a model evaluation.</div>
<div id="stats" class="stats"></div><div id="warnings"></div>
<h2>1 · Feature availability and ranges</h2><p class="note">One-second channel-summary values. Missing values are excluded from the distribution charts.</p><div id="feature-table" class="table-wrap"></div><div id="burst-table" class="table-wrap"></div><div id="state-table" class="table-wrap"></div>
<h2>2 · Input distributions</h2><div id="hists" class="grid"></div>
<h2>3 · Changes through time</h2><div id="times" class="grid"></div>
<h2>4 · Candidate output: claimed road segment</h2><p class="note">S-labels are shortened centerline-piece IDs, ordered approximately north to south. The label comes from the BSM-derived location.</p><div id="counts" class="full"></div><div id="segment-table" class="table-wrap"></div><div id="timeline" class="full"></div>
<h2>5 · Back-and-forth pass sanity check</h2><p class="note">Passes are estimated from BSM heading flips (near 0° = north, near 180° = south). Whole-pass speed is rough and based on endpoint displacement. The table below uses samples per visited segment per pass as an estimate of time spent there; implied speed is segment length divided by that time.</p><div id="pass-chart" class="full"></div><div id="pass-table" class="table-wrap"></div><h3>Time spent in each segment by direction</h3><div id="pass-segment-table" class="table-wrap"></div>
<h2>6 · Input variation by claimed segment</h2><div id="box" class="full"></div><div id="correlations" class="table-wrap"></div>
<h2>7 · Location and matching checks</h2><p class="note">Working assumption: the RSU is at the arithmetic center of the reported BSM coordinates. The 600 m circle is retained from the capture configuration for scale; its center is assumed here, not measured.</p><div id="locations" class="grid"></div>
<h2>8 · Random row inspection</h2><h3>Summary rows: inputs and assigned segment</h3><div id="samples" class="table-wrap"></div><h3>Radio bursts: raw signal and decode fields</h3><div id="burst-samples" class="table-wrap"></div>
<div class="foot">The capture is one run and one UE track. Adjacent one-second rows are related; a random row split would overstate independent test performance.</div>
</main><script>
const D=__DATA__;
const css=getComputedStyle(document.documentElement);
const C={blue:css.getPropertyValue('--blue').trim(),orange:css.getPropertyValue('--orange').trim(),green:css.getPropertyValue('--green').trim(),purple:css.getPropertyValue('--purple').trim(),red:css.getPropertyValue('--red').trim(),fg:css.getPropertyValue('--fg').trim(),muted:css.getPropertyValue('--muted').trim(),line:css.getPropertyValue('--line').trim(),grid:css.getPropertyValue('--grid').trim()};
function f(x,d){return x==null?'—':Number(x).toLocaleString(undefined,{maximumFractionDigits:d==null?2:d});}
function stat(label,value){return '<div class="stat"><b>'+value+'</b><span>'+label+'</span></div>';}
document.getElementById('stats').innerHTML=[
 stat('Summary rows',f(D.counts.summaries,0)+' · one-second'),
 stat('Radio bursts',f(D.counts.bursts,0)),
 stat('Capture span',f(D.duration,1)+' min'),
 stat('UE tracks / runs',D.counts.tracks+' / '+D.counts.runs),
 stat('Assigned pieces',D.counts.active_segments+' / '+D.segments.length),
 stat('Median sample spacing',f(D.median_interval,2)+' s')
].join('');
document.getElementById('warnings').innerHTML=
 '<div class="warning"><b>Feature gap:</b> Doppler is available for '+f(D.doppler_available,0)+' of '+f(D.counts.bursts,0)+' bursts; channel response for '+f(D.channel_available,0)+'. The saved capture configuration disabled both.</div>'+
 '<div class="warning"><b>RSU assumption:</b> using the coordinate centroid ('+D.rsu.lat.toFixed(6)+', '+D.rsu.lon.toFixed(6)+'). '+f(D.within_radius,0)+' of '+f(D.series.length,0)+' points fall within the assumed '+f(D.rsu.radius,0)+' m radius; distances span '+f(Math.min.apply(null,D.rsu_distances),0)+'–'+f(Math.max.apply(null,D.rsu_distances),0)+' m.</div>'+
 '<div class="warning"><b>Target interpretation:</b> for this experiment, BSM locations are treated as ground truth; the segment target is assigned from those coordinates by nearest-centerline matching. This export has no spoofed/benign outcome label.</div>';
function svg(w,h,title,desc){return '<svg viewBox="0 0 '+w+' '+h+'" role="img" aria-label="'+title+'"><title>'+title+'</title><desc>'+desc+'</desc>';}
function place(id,title,markup){document.getElementById(id).innerHTML+='<div class="chart"><h3>'+title+'</h3>'+markup+'</div>';}
function histogram(key){const x=D.features[key],vals=x.values.filter(v=>v!=null);if(!vals.length)return;const W=520,H=238,L=50,R=12,T=12,B=40,iw=W-L-R,ih=H-T-B,n=16,mn=Math.min.apply(null,vals),mx=Math.max.apply(null,vals),span=mx-mn||1,counts=Array(n).fill(0);vals.forEach(v=>counts[Math.min(n-1,Math.floor((v-mn)/span*n))]++);const ym=Math.max.apply(null,counts);let s=svg(W,H,x.label+' distribution','Histogram of '+vals.length+' one-second rows; values in '+x.unit+'.');for(let i=0;i<=4;i++){let y=T+ih*i/4;s+='<line class="gridline" x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'"/><text class="tick" x="'+(L-7)+'" y="'+(y+4)+'" text-anchor="end">'+f(ym*(1-i/4),0)+'</text>';}counts.forEach((v,i)=>{let bw=iw/n-2,bh=ih*v/ym;s+='<rect x="'+(L+i*iw/n+1)+'" y="'+(T+ih-bh)+'" width="'+bw+'" height="'+bh+'" fill="'+C.blue+'" opacity=".85"><title>'+f(mn+span*i/n)+' to '+f(mn+span*(i+1)/n)+' '+x.unit+': '+v+' rows</title></rect>';});s+='<line class="axis" x1="'+L+'" x2="'+(W-R)+'" y1="'+(T+ih)+'" y2="'+(T+ih)+'"/><text class="tick" x="'+L+'" y="'+(H-19)+'">'+f(mn)+'</text><text class="tick" x="'+(W-R)+'" y="'+(H-19)+'" text-anchor="end">'+f(mx)+'</text><text class="label" x="'+(L+iw/2)+'" y="'+(H-2)+'" text-anchor="middle">'+x.label+' ('+x.unit+')</text></svg>';return s;}
const featureRows=D.feature_order.map(k=>{const x=D.features[k];return ['<b>'+x.label+'</b> ('+x.unit+')',f(x.n,0),f(x.missing,0),f(x.min),f(x.p05),f(x.q25),f(x.median),f(x.q75),f(x.p95),f(x.max),f(x.mean),f(x.sd)];});
document.getElementById('feature-table').innerHTML='<table><thead><tr><th>Feature</th><th>Present</th><th>Missing</th><th>Min</th><th>P05</th><th>Q1</th><th>Median</th><th>Q3</th><th>P95</th><th>Max</th><th>Mean</th><th>SD</th></tr></thead><tbody>'+featureRows.map(r=>'<tr>'+r.map((v,i)=>'<td'+(i?' class="num"':'')+'>'+v+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
document.getElementById('burst-table').innerHTML='<h3>Individual-burst feature availability</h3>'+table(['Burst feature','Present','Missing','Total'],Object.keys(D.burst_availability).map(k=>[k,f(D.burst_availability[k],0),f(D.counts.bursts-D.burst_availability[k],0),f(D.counts.bursts,0)]));
const stateRows=[];Object.keys(D.states).forEach(k=>Object.keys(D.states[k]).sort((a,b)=>D.states[k][b]-D.states[k][a]).forEach(v=>stateRows.push([k.replaceAll('_',' '),v,f(D.states[k][v],0)])));
document.getElementById('state-table').innerHTML='<h3>Burst decode and attribution outcomes</h3>'+table(['Field','Observed value','Burst count'],stateRows);
document.getElementById('hists').innerHTML=D.feature_order.map(k=>'<div class="chart"><h3>'+D.features[k].label+' · '+D.features[k].unit+'</h3>'+histogram(k)+'</div>').join('');
function lineChart(title,series,ylabel){const W=820,H=280,L=58,R=14,T=14,B=42,iw=W-L-R,ih=H-T-B,all=series.flatMap(s=>s.values.filter(v=>v!=null)),mn=Math.min.apply(null,all),mx=Math.max.apply(null,all),span=mx-mn||1;let s=svg(W,H,title,title+' across '+f(D.duration,1)+' minutes.');for(let i=0;i<=4;i++){let y=T+ih*i/4;s+='<line class="gridline" x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'"/><text class="tick" x="'+(L-7)+'" y="'+(y+4)+'" text-anchor="end">'+f(mx-span*i/4,1)+'</text>';}for(let i=0;i<=4;i++){let x=L+iw*i/4;s+='<line class="gridline" x1="'+x+'" x2="'+x+'" y1="'+T+'" y2="'+(T+ih)+'"/><text class="tick" x="'+x+'" y="'+(T+ih+17)+'" text-anchor="middle">'+f(D.duration*i/4,0)+'</text>';}series.forEach((ser,j)=>{let d='',on=false;ser.values.forEach((v,i)=>{if(v==null){on=false;return;}let x=L+iw*i/(ser.values.length-1),y=T+ih*(mx-v)/span;d+=(on?'L':'M')+x.toFixed(1)+','+y.toFixed(1)+' ';on=true;});s+='<path d="'+d+'" fill="none" stroke="'+ser.color+'" stroke-width="1.6"/><text class="label" x="'+(L+j*155)+'" y="'+(H-2)+'" fill="'+ser.color+'">● '+ser.name+'</text>';});s+='<text class="label" transform="translate(13 '+(T+ih/2)+') rotate(-90)" text-anchor="middle">'+ylabel+'</text><text class="tick" x="'+(W-R)+'" y="'+(T+ih+17)+'" text-anchor="end">minutes from capture start</text></svg>';return s;}
place('times','Signal power through time',lineChart('RSRP and RSSI',[{name:'RSRP',values:D.series.map(p=>p.rsrp_dbm),color:C.blue},{name:'RSSI',values:D.series.map(p=>p.rssi_dbm),color:C.orange}],'dBm'));
place('times','Channel activity through time',lineChart('Estimated CBR and frame busy ratio',[{name:'Estimated CBR',values:D.series.map(p=>p.estimated_cbr_percent),color:C.green},{name:'Frame busy ratio',values:D.series.map(p=>p.frame_busy_ratio_percent),color:C.purple}],'percent'));
function table(headers,rows){return '<table><thead><tr>'+headers.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map((v,i)=>'<td'+(i?' class="num"':'')+'>'+(v==null?'—':v)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';}
function segmentBars(){const rows=D.segments,W=900,H=24+rows.length*22,L=72,R=35,T=10,B=27,iw=W-L-R,ym=Math.max.apply(null,rows.map(x=>x.count));let s=svg(W,H,'Samples by centerline piece','Bar chart of count of mapped summary rows for each of 25 pieces.');rows.forEach((r,i)=>{let y=T+i*22,bw=iw*r.count/(ym||1);s+='<text class="tick" x="'+(L-8)+'" y="'+(y+14)+'" text-anchor="end">'+r.label+'</text><rect x="'+L+'" y="'+(y+2)+'" width="'+bw+'" height="14" fill="'+(r.count?C.blue:C.line)+'"><title>'+r.label+' · '+r.id+' · '+r.count+' samples · '+f(r.length)+' m</title></rect><text class="tick" x="'+(L+bw+4)+'" y="'+(y+14)+'">'+r.count+'</text>';});s+='<text class="label" x="'+(L+iw/2)+'" y="'+(H-3)+'" text-anchor="middle">One-second samples assigned</text></svg>';return s;}
place('counts','Sample counts by road piece',segmentBars());
document.getElementById('segment-table').innerHTML=table(['Label','Segment ID','Length (m)','Sample count'],D.segments.map(x=>[x.label,'<span class="mono">'+x.id+'</span>',f(x.length),f(x.count,0)]));
function passChart(){const P=D.passes,W=900,H=40+P.length*24,L=75,R=45,T=10,iw=W-L-R,mx=Math.max.apply(null,P.map(x=>x.avg_speed_kmh||0));let s=svg(W,H,'Estimated average speed per traversal','Bars show rough north-south endpoint speed; N/S indicates BSM heading direction.');P.forEach((p,i)=>{const y=T+i*24,w=iw*(p.avg_speed_kmh||0)/(mx||1),color=p.direction==='N'?C.blue:C.green;s+='<text class="tick" x="'+(L-8)+'" y="'+(y+14)+'" text-anchor="end">'+(i+1)+' '+p.direction+'</text><rect x="'+L+'" y="'+(y+2)+'" width="'+w+'" height="15" fill="'+color+'"><title>'+p.start_utc+'–'+p.end_utc+' UTC · '+f(p.duration_s,0)+' s · '+f(p.avg_speed_kmh,1)+' km/h</title></rect><text class="tick" x="'+(L+w+5)+'" y="'+(y+14)+'">'+f(p.avg_speed_kmh,1)+' km/h</text>';});return s+'<text class="label" x="'+(L+iw/2)+'" y="'+(H-3)+'" text-anchor="middle">Approximate average speed per pass</text></svg>';}
place('pass-chart','Pass duration varies substantially',passChart());
document.getElementById('pass-table').innerHTML=table(['Pass','Direction','Start (UTC)','End (UTC)','Samples','Duration (s)','Endpoint distance (m)','Approx. avg speed (km/h)'],D.passes.map((p,i)=>[i+1,p.direction,p.start_utc,p.end_utc,p.count,f(p.duration_s,0),f(p.distance_m,0),f(p.avg_speed_kmh,1)]));
document.getElementById('pass-segment-table').innerHTML=table(['Segment','Length (m)','N visits','N mean samples / visit','N implied speed (km/h)','S visits','S mean samples / visit','S implied speed (km/h)'],D.pass_segment_rows.filter(x=>x.total>0).map(x=>[x.label,f(x.length),x.n_visits,f(x.n_mean_s,1),f(x.n_speed_kmh,1),x.s_visits,f(x.s_mean_s,1),f(x.s_speed_kmh,1)]));
function timeline(){const W=900,H=420,L=65,R=15,T=10,B=38,iw=W-L-R,ih=H-T-B,n=D.segments.length,lookup={};D.segments.forEach((s,i)=>lookup[s.label]=i);let s=svg(W,H,'Assigned segment through time','One dot per summary row; vertical position is the assigned centerline piece.');for(let i=0;i<n;i++){let y=T+ih*(i+.5)/n;s+='<text class="tick" x="'+(L-7)+'" y="'+(y+3)+'" text-anchor="end">'+D.segments[i].label+'</text>';}for(let j=0;j<=4;j++){let x=L+iw*j/4;s+='<line class="gridline" x1="'+x+'" x2="'+x+'" y1="'+T+'" y2="'+(T+ih)+'"/><text class="tick" x="'+x+'" y="'+(H-19)+'" text-anchor="middle">'+f(D.duration*j/4,0)+'</text>';}D.series.forEach(p=>{let i=lookup[p.segment];if(i==null)return;let x=L+iw*p.t/D.duration,y=T+ih*(i+.5)/n;s+='<circle cx="'+x.toFixed(1)+'" cy="'+y.toFixed(1)+'" r="1.8" fill="'+C.blue+'" opacity=".72"><title>'+p.utc+' UTC · '+p.segment+' · RSRP '+f(p.rsrp_dbm)+' dBm</title></circle>';});s+='<text class="label" transform="translate(13 '+(T+ih/2)+') rotate(-90)" text-anchor="middle">Assigned piece (north-to-south order)</text><text class="label" x="'+(L+iw/2)+'" y="'+(H-1)+'" text-anchor="middle">Time from capture start (minutes)</text></svg>';return s;}
place('timeline','Label sequence during capture',timeline());
function boxplot(){const rows=D.segments.filter(x=>x.count>0),all=D.series.map(x=>x.rsrp_dbm).filter(x=>x!=null),mn=Math.min.apply(null,all),mx=Math.max.apply(null,all),span=mx-mn||1,W=900,H=35+rows.length*29,L=80,R=30,T=10,B=31,iw=W-L-R;let s=svg(W,H,'RSRP by assigned segment','Median and interquartile range of RSRP for each segment with samples.');for(let j=0;j<=4;j++){let x=L+iw*j/4;s+='<line class="gridline" x1="'+x+'" x2="'+x+'" y1="'+T+'" y2="'+(H-B)+'"/><text class="tick" x="'+x+'" y="'+(H-12)+'" text-anchor="middle">'+f(mn+span*j/4,0)+'</text>';}rows.forEach((r,i)=>{let vals=D.series.filter(p=>p.segment===r.label&&p.rsrp_dbm!=null).map(p=>p.rsrp_dbm).sort((a,b)=>a-b),q=p=>vals[Math.round((vals.length-1)*p)],x=v=>L+iw*(v-mn)/span,y=T+i*29+12;s+='<text class="tick" x="'+(L-8)+'" y="'+(y+4)+'" text-anchor="end">'+r.label+' ('+vals.length+')</text><line x1="'+x(q(.25))+'" x2="'+x(q(.75))+'" y1="'+y+'" y2="'+y+'" stroke="'+C.blue+'" stroke-width="8"><title>'+r.label+': Q1 '+f(q(.25))+' dBm, median '+f(q(.5))+' dBm, Q3 '+f(q(.75))+' dBm</title></line><line x1="'+x(q(.5))+'" x2="'+x(q(.5))+'" y1="'+(y-6)+'" y2="'+(y+6)+'" stroke="'+C.fg+'" stroke-width="2"/>';});s+='<text class="label" x="'+(L+iw/2)+'" y="'+(H-1)+'" text-anchor="middle">RSRP (dBm) · thick mark Q1–Q3, tick median</text></svg>';return s;}
place('box','RSRP spread per claimed segment',boxplot());
function correlation(){const keys=D.correlation_keys,head='<thead><tr><th>Feature</th>'+keys.map(k=>'<th>'+k.replaceAll('_',' ')+'</th>').join('')+'</tr></thead>';const rows=keys.map(a=>'<tr><th>'+a.replaceAll('_',' ')+'</th>'+keys.map(b=>{let v=D.correlation[a][b],alpha=v==null?0:Math.min(.6,Math.abs(v)*.6),color=v==null?'transparent':v>=0?'rgba(24,117,200,'+alpha+')':'rgba(199,66,66,'+alpha+')';return '<td style="background:'+color+'">'+(v==null?'—':v.toFixed(2))+'</td>';}).join('')+'</tr>').join('');return '<h3>Pairwise Pearson correlations · summary rows</h3><p class="small">Descriptive only: shared time trends and repeated samples can inflate correlation.</p><table class="heat">'+head+'<tbody>'+rows+'</tbody></table>';}
document.getElementById('correlations').innerHTML=correlation();
function scatter(){const pts=D.series.filter(p=>p.distance!=null&&p.rsrp_dbm!=null),W=700,H=285,L=62,R=16,T=15,B=42,iw=W-L-R,ih=H-T-B,xmin=Math.min.apply(null,pts.map(p=>p.distance)),xmax=Math.max.apply(null,pts.map(p=>p.distance)),ymin=Math.min.apply(null,pts.map(p=>p.rsrp_dbm)),ymax=Math.max.apply(null,pts.map(p=>p.rsrp_dbm)),xs=xmax-xmin||1,ys=ymax-ymin||1;let s=svg(W,H,'Distance from assumed central RSU and RSRP','Scatter plot of BSM-point distance from the assumed coordinate centroid and RSRP.');for(let j=0;j<=4;j++){let x=L+iw*j/4,y=T+ih*j/4;s+='<line class="gridline" x1="'+x+'" x2="'+x+'" y1="'+T+'" y2="'+(T+ih)+'"/><line class="gridline" x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'"/><text class="tick" x="'+x+'" y="'+(H-19)+'" text-anchor="middle">'+f(xmin+xs*j/4,0)+'</text><text class="tick" x="'+(L-7)+'" y="'+(y+4)+'" text-anchor="end">'+f(ymax-ys*j/4,0)+'</text>';}pts.forEach(p=>{let x=L+iw*(p.distance-xmin)/xs,y=T+ih*(ymax-p.rsrp_dbm)/ys;s+='<circle cx="'+x+'" cy="'+y+'" r="2" fill="'+C.green+'" opacity=".48"><title>'+p.utc+' UTC · '+f(p.distance,0)+' m from assumed center · '+f(p.rsrp_dbm)+' dBm · '+p.segment+'</title></circle>';});s+='<text class="label" x="'+(L+iw/2)+'" y="'+(H-1)+'" text-anchor="middle">Distance from assumed RSU center (m)</text><text class="label" transform="translate(13 '+(T+ih/2)+') rotate(-90)" text-anchor="middle">RSRP (dBm)</text></svg><p class="small">Pearson r (distance, RSRP): '+f(D.distance_rsrp_correlation,3)+' · negative values mean RSRP tends to be lower farther from the assumed center.</p>';return s;}
place('locations','Signal level against distance from assumed RSU center',scatter());
function offsets(){const vals=D.offsets.filter(x=>x!=null),mx=Math.max.apply(null,vals),bins=Array(16).fill(0);vals.forEach(v=>bins[Math.min(15,Math.floor(v/(mx||1)*16))]++);const W=700,H=235,L=55,R=15,T=12,B=40,iw=W-L-R,ih=H-T-B,ym=Math.max.apply(null,bins);let s=svg(W,H,'Point-to-centerline offset distribution','Histogram of mapped GPS point distances to nearest centerline in meters.');for(let i=0;i<=4;i++){let y=T+ih*i/4;s+='<line class="gridline" x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'"/><text class="tick" x="'+(L-7)+'" y="'+(y+4)+'" text-anchor="end">'+f(ym*(1-i/4),0)+'</text>';}bins.forEach((v,i)=>{let bw=iw/16-2,bh=ih*v/ym;s+='<rect x="'+(L+i*iw/16+1)+'" y="'+(T+ih-bh)+'" width="'+bw+'" height="'+bh+'" fill="'+C.purple+'"><title>'+f(mx*i/16)+'–'+f(mx*(i+1)/16)+' m: '+v+' points</title></rect>';});s+='<line class="axis" x1="'+L+'" x2="'+(W-R)+'" y1="'+(T+ih)+'" y2="'+(T+ih)+'"/><text class="tick" x="'+L+'" y="'+(H-18)+'">0</text><text class="tick" x="'+(W-R)+'" y="'+(H-18)+'" text-anchor="end">'+f(mx)+' m</text><text class="label" x="'+(L+iw/2)+'" y="'+(H-1)+'" text-anchor="middle">Nearest centerline offset</text></svg>';const sorted=vals.slice().sort((a,b)=>a-b);return s+'<p class="small">Median '+f(sorted[Math.floor(sorted.length/2)])+' m · maximum '+f(mx)+' m · matching cutoff 20 m.</p>';}
place('locations','Point-to-centerline offsets',offsets());
const sampleRows=D.samples.map(x=>[x.utc,x.lat.toFixed(6),x.lon.toFixed(6),x.segment,f(x.offset),f(x.rsrp),f(x.rssi),f(x.cbr),f(x.bsm,0)]);
document.getElementById('samples').innerHTML=table(['UTC','Latitude','Longitude','Segment','Offset m','RSRP dBm','RSSI dBm','CBR %','BSMs/s'],sampleRows);
const burstRows=D.burst_samples.map(x=>[x.utc,f(x.power),f(x.rsrp),f(x.rssi),String(x.doppler),f(x.doppler_hz),x.sci||'—',x.tb||'—',x.mcs==null?'—':x.mcs,x.attribution||'—']);
document.getElementById('burst-samples').innerHTML=table(['UTC','Burst power dBm','RSRP dBm','RSSI dBm','Doppler available','Doppler Hz','SCI CRC','TB CRC','MCS','UE attribution'],burstRows);
</script></body></html>'''


def main():
    data = make_data()
    source = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    source = source.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(HTML.replace("__DATA__", source), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Summary rows: {data['counts']['summaries']:,}; bursts: {data['counts']['bursts']:,}; diagnostics: {data['counts']['diagnostics']:,}")
    print(f"Segments with samples: {data['counts']['active_segments']}/{len(data['segments'])}")
    print(f"Doppler available: {data['doppler_available']:,}; channel response available: {data['channel_available']:,}")


if __name__ == "__main__":
    main()
