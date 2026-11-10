#!/usr/bin/env python3
"""Inspect saved training-only CV predictions; write a standalone HTML report.

No fitting or test-set scoring. Geography is point-to-predicted-polyline distance,
not network distance or a predicted point-location error.
"""
import csv
import html
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from segment_road import local_xy, point_to_line

OUT = Path(__file__).resolve().parent / 'output'


def read_csv(name):
    with (OUT / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    predictions = read_csv('weighted_knn_cv_predictions.csv')
    train = {r['timestamp_unix_ms']: r for r in read_csv('modeling_dataset.csv') if r['split'] == 'train'}
    points = {r['timestamp_unix_ms']: r for r in read_csv('location_points_segmented.csv')}
    assert predictions and len(predictions) == len(train)
    assert len({r['timestamp_unix_ms'] for r in predictions}) == len(predictions)
    assert {r['timestamp_unix_ms'] for r in predictions} == set(train), 'Predictions must cover training rows exactly'
    features = json.loads((OUT / 'road_segments.geojson').read_text())['features']
    roads = {f['properties']['segment_id']: f['geometry']['coordinates'] for f in features}
    # Same display ordering as audit_field_data.make_data; IDs are not adjacency.
    ordered = sorted(roads, key=lambda s: (-statistics.mean(c[1] for c in roads[s]), statistics.mean(c[0] for c in roads[s])))
    names = {s: f'S{i+1:02}' for i, s in enumerate(ordered)}
    counts = Counter()
    wrong_distances = []
    lon0 = statistics.mean(float(points[t]['longitude']) for t in train)
    lat0 = statistics.mean(float(points[t]['latitude']) for t in train)
    for row in predictions:
        t = row['timestamp_unix_ms']
        for key in ('segment_id', 'cv_fold', 'pass_index', 'direction'):
            assert row[key] == train[t][key], (t, key)
        actual, predicted = row['segment_id'], row['predicted_segment_id']
        assert actual == points[t]['matched_segment_id']
        assert actual in roads and predicted in roads
        counts[actual, predicted] += 1
        if actual != predicted:
            p = points[t]
            xy = local_xy(float(p['longitude']), float(p['latitude']), lon0, lat0)
            d, _ = point_to_line(xy, roads[predicted], lon0, lat0)
            assert math.isfinite(d) and d >= 0
            wrong_distances.append(d)
    labels = [s for s in ordered if any(a == s or b == s for a, b in counts)]
    support = {a: sum(counts[a, b] for b in labels) for a in labels}
    recall = {a: counts[a, a] / support[a] if support[a] else None for a in labels}
    accuracy = sum(counts[a, a] for a in labels) / len(predictions)
    macro = statistics.mean(v for v in recall.values() if v is not None)
    report = json.loads((OUT / 'weighted_knn_cv_report.json').read_text())['overall']
    assert report['held_out_rows'] == len(predictions)
    assert math.isclose(accuracy, report['weighted_knn_accuracy'], abs_tol=1e-12)
    assert math.isclose(macro, report['weighted_knn_macro_recall'], abs_tol=1e-12)
    assert sum(counts.values()) == len(predictions)
    esc = html.escape
    page = ['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Weighted k-NN error inspection</title><style>
body{font:15px/1.5 system-ui;color:#18212b;background:#f6f8fb;margin:0}main{max-width:1150px;margin:auto;padding:28px}h1{margin-bottom:6px}h2{margin-top:32px}p{max-width:900px}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:13px}th,td{padding:7px;border:1px solid #dbe2eb;text-align:right}th:first-child,td:first-child{text-align:left}.matrix td{text-align:center;min-width:44px}.bar{height:16px;background:#2472b8;border-radius:2px}small{color:#526174}svg{width:100%;max-width:900px;background:white}code{font-size:12px}
</style><main><h1>Weighted k-NN: where are the errors?</h1>
<p>Saved out-of-fold predictions from five grouped folds within the training set. No retraining, validation scoring, or test scoring.</p>''']
    page.append(f'<p><b>{len(predictions):,} rows · accuracy {accuracy:.1%} · macro recall {macro:.1%} · {len(wrong_distances):,} incorrect predictions</b></p>')
    page.append('<h2>1. Confusion matrix</h2><p>Rows are true segments; columns are predicted segments. Each cell shows its row percentage; hover for the count. Darker cells indicate more predictions. Correct predictions lie on the diagonal. S-labels match the audit.</p><div class="scroll"><table class="matrix"><tr><th>True ↓ / Predicted →</th>')
    page.extend(f'<th>{names[b]}</th>' for b in labels)
    page.append('</tr>')
    for a in labels:
        page.append(f'<tr><th>{names[a]} ({support[a]})</th>')
        for b in labels:
            pct = counts[a, b] / support[a] if support[a] else 0
            page.append(f'<td title="{counts[a,b]} / {support[a]}" style="background:rgba(36,114,184,{pct:.3f});color:{"white" if pct>.55 else "#18212b"}">{pct:.0%}</td>')
        page.append('</tr>')
    page.append('</table></div><h2>2. Recall by segment</h2><p>Recall is the fraction of a segment’s rows classified correctly. Support is the number of held-out rows for that segment, pooled across folds.</p><table><tr><th>Segment</th><th>Support</th><th>Correct</th><th>Recall</th><th style="width:35%">0–100%</th></tr>')
    for a in labels:
        v = recall[a] or 0
        page.append(f'<tr><td title="{esc(a)}">{names[a]}</td><td>{support[a]}</td><td>{counts[a,a]}</td><td>{v:.1%}</td><td><div class="bar" style="width:{v*100:.2f}%"></div></td></tr>')
    page.append('</table><h2>3. Geographic distance for incorrect predictions</h2><p>Shortest distance from the reported BSM coordinate to the predicted segment’s centerline, in a local metric projection. Only incorrect segment predictions are included. This is not road-network distance, segment adjacency, or distance to a predicted vehicle position. A wrong neighboring segment may still be very close to the coordinate.</p>')
    if wrong_distances:
        ds = sorted(wrong_distances)
        p90 = ds[math.ceil(.9*len(ds))-1]
        page.append(f'<p>Median <b>{statistics.median(ds):.1f} m</b> · 90th percentile <b>{p90:.1f} m</b> · maximum <b>{max(ds):.1f} m</b>. P90 uses the nearest-rank definition.</p>')
        xmax = max(ds) or 1
        svg = ['<svg viewBox="0 0 900 300" role="img" aria-label="Cumulative fraction of incorrect predictions versus distance to predicted segment"><title>Geographic error cumulative distribution</title>']
        for i in range(5):
            y = 250-220*i/4
            svg.append(f'<line x1="70" x2="860" y1="{y}" y2="{y}" stroke="#dbe2eb"/><text x="60" y="{y+5}" text-anchor="end">{i*25}%</text>')
            svg.append(f'<text x="{70+790*i/4}" y="273" text-anchor="middle">{xmax*i/4:.0f}</text>')
        path = 'M70,250 ' + ' '.join(f'H{70+790*d/xmax:.2f} V{250-220*(i+1)/len(ds):.2f}' for i,d in enumerate(ds))
        svg.append(f'<path d="{path}" fill="none" stroke="#2472b8" stroke-width="2"/><text x="465" y="296" text-anchor="middle">Distance to predicted centerline (m)</text></svg>')
        page.extend(svg)
        page.append('<p><small>At each distance, the curve shows the fraction of incorrect predictions within that distance. No spoof-acceptance threshold is selected here.</small></p>')
    page.append('<h2>4. Most frequent confusions</h2><table><tr><th>True → predicted</th><th>Count</th><th>Fraction of true segment</th></tr>')
    for (a,b), n in sorted(((pair,n) for pair,n in counts.items() if pair[0]!=pair[1]), key=lambda x:(-x[1],x[0]))[:10]:
        page.append(f'<tr><td>{names[a]} → {names[b]}</td><td>{n}</td><td>{n/support[a]:.1%}</td></tr>')
    page.append('</table><p><small>Sources: weighted_knn_cv_predictions.csv, modeling_dataset.csv, location_points_segmented.csv, road_segments.geojson, weighted_knn_cv_report.json. All results concern one capture and one vehicle. Segment codes identify geometries; their numerical differences do not measure separation.</small></p></main></html>')
    target = OUT / 'knn_error_analysis.html'
    target.write_text(''.join(page))
    print(f'Wrote {target}\nAccuracy {accuracy:.6f}; macro recall {macro:.6f}; incorrect {len(wrong_distances)}')
    if wrong_distances:
        print(f'Incorrect predictions: median distance {statistics.median(ds):.1f} m; P90 {p90:.1f} m')
    print('Checks passed: unique training-only coverage, metadata and labels, finite distances, and agreement with baseline report.')


if __name__ == '__main__':
    # Analytic checks for perpendicular projection and clamping at an endpoint.
    line = [[0,0],[.001,0]]
    assert math.isclose(point_to_line(local_xy(.0005,.0001,0,0),line,0,0)[0],local_xy(0,.0001,0,0)[1])
    assert math.isclose(point_to_line(local_xy(.002,0,0,0),line,0,0)[0],local_xy(.001,0,0,0)[0])
    main()
