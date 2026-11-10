#!/usr/bin/env python3
"""Fixed score blend and coverage/geography evaluation; no fitting or detector."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_models import FEATURES, OUT, SEEDS, read
from compare_window_models import KEYS, NEW
from compare_window_weights import NAMES
from inspect_forest_leaf_scores import score_metrics
from optimize_models import evaluate, geography

BASELINE = 'Forest unweighted'
BLEND = 'Forest-MLP equal blend'
SIZES = [1, 3, 5, 8, 16]
DISTANCES = [200, 500, 1000]
# Each axis is kept separate; no weighted aggregate objective is constructed.
AXES = {'accuracy':1, 'macro_recall':1, 'macro_top3_coverage':1,
        'log_loss':-1, 'multiclass_brier':-1, 'all_distance_p90':-1}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_set_metrics(y, p, labels, geo, size):
    yi = np.searchsorted(labels, y)
    selected = p.argmax(axis=1)[:, None] if size == 1 else np.argsort(p, axis=1)[:, -size:]
    hit = (selected == yi[:, None]).any(axis=1)
    true = p[np.arange(len(y)), yi]
    greater = np.sum(p > true[:, None], axis=1); equal = np.sum(p == true[:, None], axis=1)
    distances = np.take_along_axis(geo, selected, axis=1)
    near = distances.min(axis=1); far = distances.max(axis=1)
    recall = [float(hit[y == v].mean()) if (y == v).any() else 0. for v in labels]
    return dict(size=size, coverage=float(hit.mean()), macro_coverage=float(np.mean(recall)), per_segment_coverage=recall,
                guaranteed_coverage=float((greater+equal <= size).mean()), possible_coverage=float((greater < size).mean()),
                closest_centerline_median=float(np.median(near)), closest_centerline_p90=float(np.quantile(near, .9, method='inverted_cdf')),
                farthest_centerline_median=float(np.median(far)), farthest_centerline_p90=float(np.quantile(far, .9, method='inverted_cdf')))


def measure(y, p, labels, geo):
    m, d = evaluate(y, p, labels, geo)
    sets = [candidate_set_metrics(y, p, labels, geo, k) for k in SIZES]
    m['macro_top3_coverage'] = sets[1]['macro_coverage']
    for threshold in DISTANCES:
        bad = d > threshold
        m[f'distance_over_{threshold}m'] = float(bad.mean())
        m[f'macro_distance_over_{threshold}m'] = float(np.mean([bad[y == v].mean() if (y == v).any() else 0. for v in labels]))
    bins = score_metrics(y, p, labels)[1]
    return m, sets, bins


def research_screen(summary):
    base = next(s for s in summary if s['model'] == BASELINE)
    result = []
    for s in summary:
        failed = [k for k, sign in AXES.items() if sign*(s[k]-base[k]) < -1e-12]
        result.append(dict(model=s['model'], passes_baseline_screen=not failed, failed_axes=failed))
    frontier = []
    for s in summary:
        dominated_by = []
        for other in summary:
            if other['model'] == s['model']: continue
            changes = [sign*(other[k]-s[k]) for k, sign in AXES.items()]
            if all(v >= -1e-12 for v in changes) and any(v > 1e-12 for v in changes): dominated_by.append(other['model'])
        frontier.append(dict(model=s['model'], dominated_by=dominated_by, nondominated=not dominated_by))
    return result, frontier


def run():
    target = OUT/'balanced_localization.json'; predictions_path = OUT/'balanced_localization_blend_predictions.csv'
    if target.exists() or predictions_path.exists(): raise RuntimeError('Balanced evaluation outputs exist; refusing to overwrite')
    base = json.loads((OUT/'window_model_comparison.json').read_text())
    weighted = json.loads((OUT/'window_moderate_weights.json').read_text())
    manifest = json.loads((OUT/'window_dataset_v1.json').read_text())
    assert FEATURES == base['features'] == weighted['features'] == manifest['classifier_features']
    assert SEEDS == base['seeds'] == weighted['seeds']
    assert sha(OUT/'training_windows_v1.csv') == base['window_csv_sha256'] == weighted['window_csv_sha256']
    assert sha(OUT/'window_dataset_v1.json') == base['manifest_sha256'] == weighted['manifest_sha256']
    rows = read('training_windows_v1.csv'); ids = [r['timestamp_unix_ms'] for r in rows]; needed = set(ids)
    assert len(rows) == len(needed) == 1393 and all(r['split'] == 'train' and 1 <= int(r['pass_index']) <= 10 for r in rows)
    y = np.array([r['segment_id'] for r in rows]); labels = np.unique(y); folds = np.array([int(r['cv_fold']) for r in rows])
    assert labels.tolist() == base['labels'] == weighted['labels']
    points = {r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in needed}
    assert set(points) == needed
    roads = {f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    geo = geography(rows, labels, ids, roads, points)
    source = {'unweighted':read('window_model_predictions.csv'), 'weighted':read('window_moderate_weights_predictions.csv')}
    sources = [(BASELINE, 'unweighted', NEW['RF']), ('MLP unweighted', 'unweighted', NEW['MLP']),
               ('Forest moderate weights', 'weighted', NAMES['RF']), ('MLP moderate weights', 'weighted', NAMES['MLP'])]
    probabilities = {}; runs = []; output = []
    for name, data, oldname in sources:
        saved = {(int(r['seed']), r['timestamp_unix_ms']):r for r in source[data] if r['model'] == oldname}
        assert set(saved) == {(s, t) for s in SEEDS for t in ids}
        for seed in SEEDS:
            old = [saved[seed, t] for t in ids]
            for a, b in zip(rows, old):
                for k in ['segment_id', 'cv_fold', 'pass_index']: assert a[k] == b[k]
            p = np.array([json.loads(r['probabilities']) for r in old])
            assert np.array_equal(labels[p.argmax(axis=1)], [r['predicted_segment_id'] for r in old])
            probabilities[name, seed] = p
    names = [name for name, _, _ in sources]+[BLEND]
    for seed in SEEDS:
        probabilities[BLEND, seed] = .5*(probabilities['Forest moderate weights', seed]+probabilities['MLP unweighted', seed])
    for name in names:
        for seed in SEEDS:
            p = probabilities[name, seed]
            assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1)
            metrics, sets, bins = measure(y, p, labels, geo)
            by_fold = [dict(fold=f, support=int((folds == f).sum()), metrics=measure(y[folds == f], p[folds == f], labels, geo[folds == f])[0]) for f in range(1, 6)]
            runs.append(dict(model=name, seed=seed, metrics=metrics, candidate_sets=sets, confidence_bins=bins, folds=by_fold))
            if name == BLEND:
                d = geo[np.arange(len(y)), p.argmax(axis=1)]
                for i, row in enumerate(rows):
                    output.append(dict(seed=seed, timestamp_unix_ms=ids[i], cv_fold=int(row['cv_fold']), pass_index=int(row['pass_index']),
                                       segment_id=y[i], predicted_segment_id=labels[p[i].argmax()], probabilities=json.dumps(p[i].tolist()), distance_m=float(d[i])))
    keys = KEYS+['macro_top3_coverage']+[f'{prefix}distance_over_{d}m' for prefix in ['', 'macro_'] for d in DISTANCES]
    summary = [dict(model=name, **{k:float(np.mean([r['metrics'][k] for r in runs if r['model'] == name])) for k in keys}) for name in names]
    screens, frontier = research_screen(summary)
    paired = []
    for name in names:
        if name == BASELINE: continue
        for seed in SEEDS:
            for f in [0, 1, 2, 3, 4, 5]:
                ix = np.ones(len(y), bool) if f == 0 else folds == f
                a = probabilities[BASELINE, seed][ix]; b = probabilities[name, seed][ix]
                am = measure(y[ix], a, labels, geo[ix])[0]; bm = measure(y[ix], b, labels, geo[ix])[0]
                ac = labels[a.argmax(axis=1)] == y[ix]; bc = labels[b.argmax(axis=1)] == y[ix]
                paired.append(dict(model=name, seed=seed, fold=f, support=int(ix.sum()), delta={k:bm[k]-am[k] for k in keys},
                                   baseline_only_correct=int((ac & ~bc).sum()), candidate_only_correct=int((~ac & bc).sum())))
    files = ['training_windows_v1.csv', 'window_dataset_v1.json', 'window_model_comparison.json', 'window_model_predictions.csv',
             'window_moderate_weights.json', 'window_moderate_weights_predictions.csv', 'road_segments.geojson']
    report = dict(protocol='Fixed equal score blend and multiobjective training-window evaluation; no model fit or detector.',
                  features=FEATURES, labels=labels.tolist(), support=base['support'], display_names=base['display_names'], seeds=SEEDS,
                  blend=dict(components=['Forest moderate weights', 'MLP unweighted'], weights=[.5, .5]),
                  objective=dict(axes=AXES, reference=BASELINE, tolerance=1e-12, operational_limits='Undecided; research screen is baseline-relative only.'),
                  summary=summary, runs=runs, paired=paired, baseline_screen=screens, pareto_frontier=frontier,
                  eligible_by_macro_recall=[s['model'] for s in sorted(summary, key=lambda s:s['macro_recall'], reverse=True)
                                            if next(v for v in screens if v['model'] == s['model'])['passes_baseline_screen']],
                  code_sha256=sha(Path(__file__)), source_sha256={f:sha(OUT/f) for f in files},
                  dependency_sha256={f:sha(Path(__file__).parent/f) for f in ['compare_models.py', 'compare_window_models.py', 'compare_window_weights.py',
                                                                          'optimize_models.py', 'inspect_forest_leaf_scores.py', 'segment_road.py']},
                  limitations=['Reused same-capture observations; seeds and nearby windows are dependent.',
                               'Baseline-relative screens are not operational limits, statistical tests or final model selection.',
                               'Candidate sets are uncalibrated empirical inclusion diagnostics; no false-alarm or spoof-detection rate is measured.',
                               'Candidate-set closest/farthest distances are actual-point-to-selected-centerline footprints, not set diameter or road distance.',
                               'Broader sets trivially raise inclusion; specificity and geographic footprint must be considered.',
                               'Live feature availability and physical label timing remain unresolved.'])
    target.write_text(json.dumps(report, indent=2)+'\n')
    with predictions_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    print(json.dumps(dict(summary=summary, baseline_screen=screens, eligible=report['eligible_by_macro_recall']), indent=2))


if __name__ == '__main__': run()
