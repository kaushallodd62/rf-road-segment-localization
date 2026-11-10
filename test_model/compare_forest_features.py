#!/usr/bin/env python3
"""Compare fixed forest feature groups on the existing training-only grouped folds."""
import csv
import hashlib
import html
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import recall_score
from threadpoolctl import threadpool_limits

from compare_models import FEATURES, OUT, RF, SEEDS, model, read, scores
from segment_road import local_xy, point_to_line

GROUPS = {
    'RSRP': ['rsrp_dbm'],
    'Power': ['rsrp_dbm', 'rssi_dbm', 'burst_power_dbm'],
    'Power + gain': ['rsrp_dbm', 'rssi_dbm', 'burst_power_dbm', 'rx_gain_db'],
    'All ten': FEATURES,
}
METRICS = ['accuracy', 'macro_recall', 'macro_f1']
DISTANCES = ['all_distance_median', 'all_distance_p90', 'wrong_distance_median', 'wrong_distance_p90']


def run():
    rows = [r for r in read('modeling_dataset.csv') if r['split'] == 'train']
    baseline = json.loads((OUT / 'model_comparison.json').read_text())
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    assert digest == baseline['training_rows_sha256']
    assert FEATURES == baseline['features'] and RF == baseline['settings']['RF']
    assert SEEDS == baseline['seeds'] and sklearn.__version__ == baseline['versions']['sklearn']
    ids = [r['timestamp_unix_ms'] for r in rows]
    assert len(ids) == len(set(ids))
    y = np.array([r['segment_id'] for r in rows]); labels = np.unique(y)
    folds = np.array([int(r['cv_fold']) for r in rows])
    passes = np.array([r['pass_index'] for r in rows])
    assert set(folds) == set(range(1, 6))
    previous = {(int(r['seed']), r['timestamp_unix_ms']): r['predicted_segment_id']
                for r in read('model_comparison_predictions.csv') if r['model'] == 'RF'}
    assert set(previous) == {(seed, t) for seed in SEEDS for t in ids}
    points = {r['timestamp_unix_ms']: r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in set(ids)}
    roads = {f['properties']['segment_id']: f['geometry']['coordinates']
             for f in json.loads((OUT / 'road_segments.geojson').read_text())['features']}
    lon0 = np.mean([float(points[t]['longitude']) for t in ids])
    lat0 = np.mean([float(points[t]['latitude']) for t in ids])
    geo = np.empty((len(rows), len(labels)))
    for i, t in enumerate(ids):
        assert points[t]['matched_segment_id'] == y[i]
        xy = local_xy(float(points[t]['longitude']), float(points[t]['latitude']), lon0, lat0)
        for j, label in enumerate(labels):
            geo[i, j] = point_to_line(xy, roads[label], lon0, lat0)[0]
    assert np.isfinite(geo).all()
    runs = []; predictions = []
    for group, features in GROUPS.items():
        X = np.array([[float(r[f]) if r[f] else np.nan for f in features] for r in rows])
        assert not np.isinf(X).any()
        for seed in SEEDS:
            pred = np.empty_like(y); seen = np.zeros(len(rows), int); details = []
            for fold in sorted(set(folds)):
                fit = folds != fold; held = ~fit
                assert not set(passes[fit]) & set(passes[held])
                assert set(y[fit]) == set(labels) and not np.isnan(X[fit]).all(axis=0).any()
                pipe = model('RF', seed)
                pipe.fit(X[fit], y[fit])
                np.testing.assert_allclose(pipe[0].statistics_, np.nanmedian(X[fit], axis=0))
                pred[held] = pipe.predict(X[held]); seen[held] += 1
                details.append(dict(fold=int(fold), fit_rows=int(fit.sum()), held_rows=int(held.sum()),
                                    held=scores(y[held], pred[held], labels)))
            assert np.all(seen == 1)
            if group == 'All ten':
                assert all(pred[i] == previous[seed, t] for i, t in enumerate(ids)), 'Baseline predictions changed'
            m = scores(y, pred, labels)
            distance = geo[np.arange(len(rows)), np.searchsorted(labels, pred)]
            wrong = pred != y
            m.update(all_distance_median=float(np.median(distance)),
                     all_distance_p90=float(np.quantile(distance, .9, method='inverted_cdf')),
                     wrong_distance_median=float(np.median(distance[wrong])) if wrong.any() else None,
                     wrong_distance_p90=float(np.quantile(distance[wrong], .9, method='inverted_cdf')) if wrong.any() else None,
                     wrong_count=int(wrong.sum()),
                     per_segment_recall=recall_score(y, pred, labels=labels, average=None, zero_division=0).tolist())
            runs.append(dict(group=group, seed=seed, metrics=m, folds=details))
            for i, t in enumerate(ids):
                predictions.append(dict(group=group, seed=seed, timestamp_unix_ms=t, cv_fold=int(folds[i]),
                                        pass_index=passes[i], segment_id=y[i], predicted_segment_id=pred[i], distance_m=float(distance[i])))
            print(f'{group}, seed {seed}: accuracy {m["accuracy"]:.4f}, macro recall {m["macro_recall"]:.4f}', flush=True)
    summary = []
    for group in GROUPS:
        rr = [r for r in runs if r['group'] == group]
        summary.append(dict(group=group, **{k: float(np.mean([r['metrics'][k] for r in rr])) for k in METRICS + DISTANCES}))
    report = dict(protocol='Training-only grouped CV; same five folds and three seeds for every group. No validation/test scoring or final fit.',
                  feature_groups=GROUPS, forest_settings=RF, seeds=SEEDS,
                  sqrt_candidates={g: max(1, int(np.sqrt(len(f)))) for g, f in GROUPS.items()},
                  labels=labels.tolist(), display_names=baseline['display_names'],
                  support=[int(sum(y == label)) for label in labels], runs=runs, summary=summary,
                  training_rows_sha256=digest, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_sha256={name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in ['location_points_segmented.csv', 'road_segments.geojson']},
                  versions={'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__},
                  checks='Disjoint passes; fitting-fold medians; each training row predicted once per seed; all-ten predictions exactly match saved RF baseline for each seed.',
                  limitations=['One vehicle and capture; seeds reuse rows and do not estimate independent-drive uncertainty.',
                               'Fixed sqrt changes candidate count with feature count; feature effects and candidate sampling are not isolated.',
                               'Distance is shortest authentic point to predicted centerline, not road-network distance; wrong-only subsets differ across models.',
                               'Coordinates are used only for labels and evaluation, never as inputs. Spoof detection is not evaluated.'])
    (OUT / 'forest_feature_groups.json').write_text(json.dumps(report, indent=2) + '\n')
    with (OUT / 'forest_feature_groups_predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    render(report)


def render(r):
    esc = html.escape
    parts = ['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Forest feature groups</title><style>body{font:15px/1.5 system-ui;background:#f6f8fb;color:#18212b}main{max-width:1250px;margin:auto;padding:24px}table{border-collapse:collapse;background:white;width:100%;margin:14px 0}th,td{padding:8px;border:1px solid #dbe2eb;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}code{font-size:.9em}</style><main><h1>Forest feature-group comparison</h1><p>1,613 training rows, five pass-grouped folds, seeds 17, 42, 123. Each cell compares the same held-out rows. Validation and test remain unscored. All-ten predictions reproduce the previous forest baseline exactly.</p><p>Each model uses 300 trees, minimum leaf size 2, and fitting-fold median imputation. With <code>max_features="sqrt"</code>, the nominal candidate counts are 1, 1, 2 and 3 for 1, 3, 4 and 10 inputs respectively; split search may inspect more features if needed to find a valid partition. The experiment changes both the available inputs and this candidate count.</p><h2>Overall pooled CV results</h2><p>Each entry averages three seed-specific scores; seeds reuse observations and are not independent drives. Distance is shortest actual point to predicted centerline, in meters. Wrong-only values use different rows when errors change; all-row values include correct predictions. P90 uses nearest rank.</p><div class="scroll"><table><tr><th>Group</th><th>Accuracy</th><th>Macro recall</th><th>Macro F1</th><th>All median m</th><th>All P90 m</th><th>Wrong median m</th><th>Wrong P90 m</th></tr>''']
    for s in r['summary']:
        parts.append('<tr><td>'+esc(s['group'])+'</td>'+''.join(f'<td>{s[k]:.1%}</td>' for k in METRICS)+''.join(f'<td>{s[k]:.1f}</td>' for k in DISTANCES)+'</tr>')
    parts.append('</table></div><h2>Feature definitions</h2><ul>')
    for g, fs in r['feature_groups'].items():
        parts.append('<li>'+esc(g)+': <code>'+esc(', '.join(fs))+'</code></li>')
    parts.append('</ul><h2>Paired seed results</h2><p>Each group shares the seed and all held-out rows. Values are accuracy / macro recall / macro F1; Δ columns are percentage-point differences from All ten with the same seed.</p><div class="scroll"><table><tr><th>Group / seed</th><th>Accuracy</th><th>Macro recall</th><th>Macro F1</th><th>Δ accuracy pp</th><th>Δ recall pp</th><th>Δ F1 pp</th></tr>')
    for run in r['runs']:
        m = run['metrics']; base = next(x['metrics'] for x in r['runs'] if x['group'] == 'All ten' and x['seed'] == run['seed'])
        parts.append(f'<tr><td>{esc(run["group"])} / {run["seed"]}</td>'+''.join(f'<td>{m[k]:.1%}</td>' for k in METRICS)+''.join(f'<td>{(m[k]-base[k])*100:+.1f}</td>' for k in METRICS)+'</tr>')
    parts.append('</table></div><h2>Paired folds and seeds</h2><p>Each row is the same seed and held-out fold; cells show accuracy / macro recall / macro F1. Fold macro scores use the complete training label list, including zero for a segment absent from that held-out fold, matching the baseline protocol.</p><div class="scroll"><table><tr><th>Seed / fold (rows)</th>'+''.join('<th>'+esc(g)+'</th>' for g in GROUPS)+'</tr>')
    for seed in SEEDS:
        for fold in range(1, 6):
            details = [next(f for f in next(x for x in r['runs'] if x['group'] == g and x['seed'] == seed)['folds'] if f['fold'] == fold) for g in GROUPS]
            parts.append(f'<tr><td>{seed} / {fold} ({details[0]["held_rows"]})</td>'+''.join('<td>'+' / '.join(f'{d["held"][k]:.1%}' for k in METRICS)+'</td>' for d in details)+'</tr>')
    parts.append('</table></div><h2>Per-segment recall</h2><p>Support counts training rows once; recall is the mean of the three seed-specific pooled predictions. S-labels are display names only, not distance measurements.</p><div class="scroll"><table><tr><th>Segment</th><th>Support</th>'+''.join('<th>'+esc(g)+'</th>' for g in GROUPS)+'</tr>')
    for j, label in sorted(enumerate(r['labels']), key=lambda x: r['display_names'][x[1]]):
        vals = [np.mean([x['metrics']['per_segment_recall'][j] for x in r['runs'] if x['group'] == g]) for g in GROUPS]
        parts.append(f'<tr><td title="{esc(label, quote=True)}">{esc(r["display_names"][label])}</td><td>{r["support"][j]}</td>'+''.join(f'<td>{v:.1%}</td>' for v in vals)+'</tr>')
    parts.append('</table></div><h2>Protocol and limits</h2><p>'+esc(r['checks'])+'</p><ul>'+''.join('<li>'+esc(x)+'</li>' for x in r['limitations'])+'</ul><p>Full metrics and row-level predictions: <a href="forest_feature_groups.json">JSON</a> and <a href="forest_feature_groups_predictions.csv">CSV</a>. Reproduce from the project root with <code>test_model/.venv/bin/python test_model/compare_forest_features.py</code>.</p></main></html>')
    (OUT / 'forest_feature_groups.html').write_text(''.join(parts))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
