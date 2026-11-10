#!/usr/bin/env python3
"""Fixed training-window audit of segment separation and pass repeatability."""
import csv
import hashlib
import html
import itertools
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from scipy.spatial.distance import cdist
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from audit_training_diagnostics import segment_geometry
from compare_models import FEATURES, OUT, read

PRIMARY = [f for f in FEATURES if f not in ['message_freshness_ms', 'bsm_rate_per_sec']]
COMPACT = ['rsrp_dbm', 'estimated_cbr_percent', 'rx_gain_db',
           'sci_crc_failure_percent', 'tb_crc_failure_percent']
VIEWS = {'eight': PRIMARY, 'compact': COMPACT, 'rsrp': ['rsrp_dbm'], 'ten': FEATURES}
SETTINGS = dict(C=1.0, class_weight='balanced', solver='lbfgs', tol=1e-6, max_iter=2000)
STEM = 'segment_fingerprints_v1'
OUTPUTS = [STEM + s for s in ['.json', '_neighbors.csv', '_pair_predictions.csv', '.html']]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(rows, features):
    x = np.array([[float(r[f]) if r[f] else np.nan for f in features] for r in rows])
    if np.isinf(x).any():
        raise ValueError('Infinite feature value')
    return x


def transform(x, fit, held):
    if not fit.any() or np.isnan(x[fit]).all(axis=0).any():
        raise ValueError('Fitting portion has an entirely missing feature')
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    a = scaler.fit_transform(imputer.fit_transform(x[fit]))
    b = scaler.transform(imputer.transform(x[held]))
    return a, b, dict(medians=imputer.statistics_.tolist(), mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist())


def nearest_distances(a, b, fit_labels, held_labels):
    d = cdist(b, a)
    same = held_labels[:, None] == fit_labels[None, :]
    ds = np.where(same, d, np.inf).min(axis=1)
    do = np.where(~same, d, np.inf).min(axis=1)
    valid = np.isfinite(ds) & np.isfinite(do)
    other = fit_labels[np.where(~same, d, np.inf).argmin(axis=1)]
    return ds, do, valid, other


def neighbor_summary(records):
    valid = [r for r in records if r['eligible']]
    def rate(part):
        return float(np.mean([r['other_closer'] for r in part])) if part else None
    return dict(queries=len(records), eligible=len(valid), unsupported=len(records)-len(valid),
                other_closer_rate=rate(valid), ties=sum(r['tie'] for r in valid),
                macro_other_closer_rate=float(np.mean([rate([r for r in valid if r['segment_id'] == s])
                                                      for s in sorted({r['segment_id'] for r in valid})])) if valid else None,
                by_fold={str(f): dict(support=sum(r['cv_fold'] == f for r in valid),
                                      other_closer_rate=rate([r for r in valid if r['cv_fold'] == f])) for f in range(1, 6)},
                by_segment={s: dict(support=sum(r['segment_id'] == s for r in valid),
                                    other_closer_rate=rate([r for r in valid if r['segment_id'] == s]))
                            for s in sorted({r['segment_id'] for r in records})})


def neighbors(rows, x, y, folds, view, scope, eligible):
    records = []; preprocessing = []
    directions = np.array([r['direction'] for r in rows])
    for f in range(1, 6):
        for direction in (['N', 'S'] if scope == 'same_direction' else ['all']):
            population = eligible & ((directions == direction) if direction != 'all' else True)
            fit = (folds != f) & population; held = (folds == f) & population
            if not held.any(): continue
            a, b, detail = transform(x, fit, held)
            ds, do, valid, other = nearest_distances(a, b, y[fit], y[held])
            preprocessing.append(dict(fold=f, direction=direction, fit_rows=int(fit.sum()), **detail))
            for j, i in enumerate(np.flatnonzero(held)):
                records.append(dict(view=view, scope=scope, timestamp_unix_ms=rows[i]['timestamp_unix_ms'],
                                    pass_index=int(rows[i]['pass_index']), cv_fold=f, segment_id=y[i],
                                    eligible=int(valid[j]), same_distance=float(ds[j]) if valid[j] else None,
                                    other_distance=float(do[j]) if valid[j] else None,
                                    other_segment=other[j] if valid[j] else '',
                                    other_closer=int(valid[j] and do[j] < ds[j]),
                                    tie=int(valid[j] and do[j] == ds[j])))
    return dict(view=view, scope=scope, **neighbor_summary(records), preprocessing=preprocessing), records


def binary_metrics(y, probability):
    pred = probability >= .5
    return dict(balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                auc=float(roc_auc_score(y, probability)),
                recall=recall_score(y, pred, labels=[0, 1], average=None).tolist(),
                support=[int((y == k).sum()) for k in [0, 1]])


def pair_probe(rows, x, y, folds, pair, view, direction):
    a, b = pair; directions = np.array([r['direction'] for r in rows])
    population = np.isin(y, pair) & ((directions == direction) if direction != 'all' else True)
    parts = []; predictions = []
    for f in range(1, 6):
        fit = population & (folds != f); held = population & (folds == f)
        support_fit = [int((fit & (y == s)).sum()) for s in pair]
        support_held = [int((held & (y == s)).sum()) for s in pair]
        minimum = 5 if direction != 'all' else 1
        base = dict(fold=f, fit_support=support_fit, held_support=support_held)
        if min(support_fit + support_held) < minimum:
            parts.append(dict(base, eligible=False)); continue
        fit_passes = sorted({int(rows[i]['pass_index']) for i in np.flatnonzero(fit)})
        held_passes = sorted({int(rows[i]['pass_index']) for i in np.flatnonzero(held)})
        assert not set(fit_passes) & set(held_passes)
        xx, hh, preprocessing = transform(x, fit, held)
        model = LogisticRegression(**SETTINGS)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(xx, (y[fit] == b).astype(int))
        probability = model.predict_proba(hh)[:, 1]
        metrics = binary_metrics((y[held] == b).astype(int), probability)
        parts.append(dict(base, eligible=True, metrics=metrics, preprocessing=preprocessing,
                          fit_passes=fit_passes, held_passes=held_passes,
                          coefficients=model.coef_.tolist(), intercept=model.intercept_.tolist()))
        for i, p in zip(np.flatnonzero(held), probability):
            predictions.append(dict(view=view, direction=direction, a=a, b=b, cv_fold=f,
                                    timestamp_unix_ms=rows[i]['timestamp_unix_ms'], segment_id=y[i], probability_b=float(p)))
    supported = [p for p in parts if p['eligible']]
    result = dict(view=view, direction=direction, a=a, b=b, folds=parts, eligible_folds=len(supported))
    if predictions:
        result['pooled'] = binary_metrics(np.array([int(r['segment_id'] == b) for r in predictions]),
                                          np.array([r['probability_b'] for r in predictions]))
        scores = [p['metrics']['balanced_accuracy'] for p in supported]
        result.update(fold_mean_balanced_accuracy=float(np.mean(scores)),
                      fold_min_balanced_accuracy=min(scores), fold_max_balanced_accuracy=max(scores))
    return result, predictions


def signature_diagnostics(rows, x, y, along, features):
    all_rows = np.ones(len(rows), bool)
    z, _, preprocessing = transform(x, all_rows, all_rows)
    passes = np.array([int(r['pass_index']) for r in rows])
    directions = {int(r['pass_index']): r['direction'] for r in rows}
    cells = {}
    for s in sorted(set(y)):
        for p in range(1, 11):
            mask = (y == s) & (passes == p)
            if mask.sum() >= 5:
                cells[s, p] = dict(signature=np.median(z[mask], axis=0), support=int(mask.sum()),
                                   along_median=float(np.median(along[mask])),
                                   radius=float(np.median(np.linalg.norm(z[mask]-np.median(z[mask], axis=0), axis=1))))
    contrasts = []
    for a, b in itertools.combinations(sorted(set(y)), 2):
        for direction in ['N', 'S']:
            supported = [p for p in range(1, 11) if directions[p] == direction and (a, p) in cells and (b, p) in cells]
            between = [dict(pass_index=p, support=[cells[a, p]['support'], cells[b, p]['support']],
                            distance=float(np.linalg.norm(cells[a, p]['signature']-cells[b, p]['signature']))) for p in supported]
            within = []
            for s in [a, b]:
                for p, q in itertools.combinations(supported, 2):
                    within.append(dict(segment=s, pass_a=p, pass_b=q,
                                       distance=float(np.linalg.norm(cells[s, p]['signature']-cells[s, q]['signature'])),
                                       along_medians=[cells[s, p]['along_median'], cells[s, q]['along_median']]))
            eligible = len(supported) >= 2
            bd = float(np.median([r['distance'] for r in between])) if between else None
            wd = float(np.median([r['distance'] for r in within])) if within else None
            contrasts.append(dict(a=a, b=b, direction=direction, eligible=eligible, supported_passes=supported,
                                  between=between, within=within, between_median=bd, within_median=wd,
                                  separation_to_drift=bd/wd if eligible and wd > 0 else None))
    bins = np.floor(along/50).astype(int); matched = []
    for s in sorted(set(y)):
        for direction in ['N', 'S']:
            for bin_id in sorted(set(bins[y == s])):
                masks = {p: (y == s) & (passes == p) & (bins == bin_id) for p in range(1, 11) if directions[p] == direction}
                supported = [p for p, mask in masks.items() if mask.sum() >= 5]
                for p, q in itertools.combinations(supported, 2):
                    ma, mb = masks[p], masks[q]
                    ca, cb = np.median(z[ma], axis=0), np.median(z[mb], axis=0)
                    observed = {}
                    for j, feature in enumerate(features):
                        va = x[ma, j]; vb = x[mb, j]; va = va[np.isfinite(va)]; vb = vb[np.isfinite(vb)]
                        observed[feature] = dict(support=[len(va), len(vb)],
                                                 median_delta=float(np.median(vb)-np.median(va)) if min(len(va), len(vb)) >= 5 else None)
                    matched.append(dict(segment=s, direction=direction, bin_start_m=int(bin_id*50), pass_a=p, pass_b=q,
                                        support=[int(ma.sum()), int(mb.sum())],
                                        along_medians=[float(np.median(along[ma])), float(np.median(along[mb]))],
                                        signature_drift=float(np.linalg.norm(ca-cb)),
                                        within_cell_radii=[float(np.median(np.linalg.norm(z[ma]-ca, axis=1))),
                                                           float(np.median(np.linalg.norm(z[mb]-cb, axis=1)))], features=observed))
    return dict(preprocessing=preprocessing, contrasts=contrasts, matched_position=matched,
                pass_signatures=[dict(segment=s, pass_index=p, direction=directions[p],
                                      **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()})
                                 for (s, p), data in cells.items()])


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def render(report):
    names = report['display_names']; labels = sorted(report['labels'], key=names.get)
    sections = []
    for view in ['eight', 'compact']:
        pairs = {(r['a'], r['b']): r for r in report['pairs'] if r['view'] == view and r['direction'] == 'all'}
        cells = ['<tr><th></th>' + ''.join(f'<th>{names[s]}</th>' for s in labels) + '</tr>']
        for a in labels:
            cells.append('<tr><th>'+names[a]+'</th>')
            for b in labels:
                if a == b: cells.append('<td>—</td>'); continue
                r = pairs.get((a, b), pairs.get((b, a))); value = r['pooled']['balanced_accuracy']
                title = f"{names[a]} / {names[b]}: pooled balanced accuracy {value:.3f}; fold range {r['fold_min_balanced_accuracy']:.3f}–{r['fold_max_balanced_accuracy']:.3f}; supports {r['pooled']['support']}"
                cells.append(f'<td title="{html.escape(title)}" style="background:hsl({max(0,min(120,240*(value-.5)))},55%,87%)">{100*value:.0f}</td>')
            cells.append('</tr>')
        sections.append(f'<h2>{view}: pairwise held-pass balanced accuracy (%)</h2><div class="scroll"><table>'+''.join(cells)+'</table></div>')
    neighbor_rows = ''.join(f"<tr><td>{r['view']}</td><td>{r['scope']}</td><td>{r['eligible']}</td><td>{100*r['other_closer_rate']:.1f}%</td><td>{100*r['macro_other_closer_rate']:.1f}%</td><td>{r['ties']}</td></tr>" for r in report['neighbors'])
    limits = ''.join('<li>'+html.escape(v)+'</li>' for v in report['limitations'])
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Segment fingerprints</title>
<style>body{font:16px/1.55 system-ui;color:#17283a;background:#f8fafc;margin:32px auto;max-width:1200px;padding:0 20px}table{border-collapse:collapse;background:white}td,th{border:1px solid #ccd5df;padding:7px;text-align:center}h1,h2{line-height:1.2}.scroll{overflow:auto}p{max-width:900px}small{color:#425466}</style>
<h1>Segment fingerprint distinguishability</h1><p>1,393 distinct training windows · 16 segments · 10 inferred passes · five held-pass folds.</p>
<p>Question: are segment differences stable across passes relative to variation inside a segment? Pairwise linear probes and distance diagnostics answer restricted versions of this question. Hover over heatmap cells for supports and fold ranges.</p>
<p>The eight-input view excludes freshness and BSM rate. The compact view also removes RSSI, burst power and frame busy ratio to reduce repeated weighting of correlated descriptors. All preprocessing for held-pass checks is fitted on fitting passes only.</p>'''+''.join(sections)+'''<p>Each cell evaluates only the two named segments. 50% is the equal-recall reference for a binary task; it is not the 16-segment chance rate. Strong pairwise discrimination does not establish unique multiclass localization. Low scores do not prove information absence.</p>
<h2>Nearest RF match in a different segment</h2><p>Strictly closer other-segment match versus the closest same-segment match in fitting passes. Lower is better under this distance measure. Unequal reference support and feature scaling influence results.</p><div class="scroll"><table><tr><th>View</th><th>Population</th><th>Eligible queries</th><th>Other closer</th><th>Segment-balanced</th><th>Ties</th></tr>'''+neighbor_rows+'''</table></div><p>Same-direction checks condition on reported heading. Gap and boundary filters restrict fitting and query rows separately from the other checks; thresholds are diagnostic choices.</p><h2>Limits</h2><ul>'''+limits+'''</ul><p>Full support, preprocessing, per-pass signatures, matched-position drift and pairwise fold metrics are in <a href="segment_fingerprints_v1.json">JSON</a>. Saved <a href="segment_fingerprints_v1_neighbors.csv">neighbor distances</a> and <a href="segment_fingerprints_v1_pair_predictions.csv">pair predictions</a> allow reproduction.</p></html>'''


def run():
    if any((OUT/name).exists() for name in OUTPUTS):
        raise RuntimeError('Fingerprint artifacts exist; refusing to overwrite')
    manifest = json.loads((OUT/'window_dataset_v1.json').read_text())
    for name, expected in manifest['artifact_sha256'].items():
        assert digest(OUT/name) == expected
    rows = read('training_windows_v1.csv')
    assert len(rows) == 1393 and manifest['classifier_features'] == FEATURES
    assert all(r['split'] == 'train' and 1 <= int(r['pass_index']) <= 10 for r in rows)
    assert len({tuple(r[k] for k in manifest['observation_key']) for r in rows}) == len(rows)
    ids = [r['timestamp_unix_ms'] for r in rows]; assert len(set(ids)) == len(rows)
    y = np.array([r['segment_id'] for r in rows]); labels = sorted(set(y))
    folds = np.array([int(r['cv_fold']) for r in rows])
    assert all(int(r['cv_fold']) == (int(r['pass_index'])+1)//2 for r in rows)
    assert all(len({r['direction'] for r in rows if int(r['pass_index']) == p}) == 1 for p in range(1, 11))
    points = {r['timestamp_unix_ms']: r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in set(ids)}
    assert set(points) == set(ids) and all(points[t]['matched_segment_id'] == s for t, s in zip(ids, y))
    roads = {f['properties']['segment_id']: f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    geometry, _, _ = segment_geometry([points[t] for t in ids], roads, labels)
    along = np.array([r['along_m'] for r in geometry])
    all_rows = np.ones(len(rows), bool)
    populations = {'all': all_rows, 'same_direction': all_rows,
                   'gap_at_most_1000ms': np.array([int(r['summary_to_ue_last_seen_ms']) <= 1000 for r in rows]),
                   'interior_over_25m': np.array([r['boundary_distance_m'] is None or r['boundary_distance_m'] > 25 for r in geometry])}
    results = []; neighbor_records = []; pairs = []; pair_records = []; signatures = {}
    for view, features in VIEWS.items():
        x = matrix(rows, features)
        for scope in (populations if view == 'eight' else ['all']):
            result, records = neighbors(rows, x, y, folds, view, scope, populations[scope])
            results.append(result); neighbor_records.extend(records)
        if view in ['eight', 'compact']:
            for direction in (['all', 'N', 'S'] if view == 'eight' else ['all']):
                for pair in itertools.combinations(labels, 2):
                    result, records = pair_probe(rows, x, y, folds, pair, view, direction)
                    pairs.append(result); pair_records.extend(records)
                print(f'Completed {view} / {direction}: 120 pairs', flush=True)
            signatures[view] = signature_diagnostics(rows, x, y, along, features)
    ordered = sorted(roads, key=lambda s: (-np.mean([c[1] for c in roads[s]]), np.mean([c[0] for c in roads[s]])))
    report = dict(protocol='Fixed training-window distinguishability audit; validation and test excluded.',
                  features=VIEWS, primary='eight', labels=labels, display_names={s:f'S{i+1:02}' for i,s in enumerate(ordered)},
                  support=dict(Counter(y)), logistic_settings=SETTINGS, min_direction_class_support=5,
                  neighbors=results, pairs=pairs, signatures=signatures,
                  geometry_context=[dict(g, summary_to_ue_last_seen_ms=int(r['summary_to_ue_last_seen_ms'])) for g, r in zip(geometry, rows)],
                  source_sha256={name:digest(OUT/name) for name in ['training_windows_v1.csv', 'window_dataset_v1.json', 'road_segments.geojson']},
                  training_points_sha256=hashlib.sha256(json.dumps([points[t] for t in ids], sort_keys=True).encode()).hexdigest(),
                  code_sha256=digest(Path(__file__)),
                  dependency_sha256={name:digest(Path(__file__).parent/name) for name in ['audit_training_diagnostics.py', 'compare_models.py', 'segment_road.py']},
                  versions=dict(numpy=np.__version__, sklearn=sklearn.__version__),
                  limitations=['Same capture and repeatedly inspected development folds; correlated windows and overlapping comparisons.',
                               'Pairwise linear discrimination and Euclidean distance cannot prove unique localization or an information ceiling.',
                               'Unequal reference counts affect neighbors; compact features and direction conditioning are fixed sensitivities.',
                               'Descriptive signature scaling uses all training windows; those distances are not held-out performance.',
                               'Whole-segment signatures mix positions; 50 m bins only approximately match position.',
                               'Gap/boundary thresholds are diagnostics, not verified RF age or GPS uncertainty limits.',
                               'Reported direction is analysis metadata, not an additional deployed classifier input.',
                               'Actual RSU position, RF/label physical-time alignment and live feature availability remain unverified.',
                               'No calibration, spoof-detection threshold, new feature collection, sequence modeling or deployment selection.'])
    (OUT/OUTPUTS[0]).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_csv(OUT/OUTPUTS[1], neighbor_records); write_csv(OUT/OUTPUTS[2], pair_records)
    (OUT/OUTPUTS[3]).write_text(render(report))
    print(json.dumps([{k:r[k] for k in ['view', 'scope', 'eligible', 'other_closer_rate', 'macro_other_closer_rate']} for r in results], indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1): run()
