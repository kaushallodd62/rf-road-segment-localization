#!/usr/bin/env python3
"""One fixed training-only RF/MLP comparison on distinct analyzer windows."""
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from compare_models import FEATURES, MLP, OUT, RF, SEEDS, model, read
from group_stopped_mlp import MAX_EPOCHS, MIN_DELTA, PATIENCE, fit_inner, refit_outer
from optimize_models import evaluate, feature_matrix, geography

KEYS = ['accuracy', 'macro_recall', 'macro_f1', 'top3_accuracy', 'log_loss', 'multiclass_brier',
        'all_distance_median', 'all_distance_p90', 'wrong_distance_median', 'wrong_distance_p90']
NEW = {'RF': 'Forest window fit', 'MLP': 'MLP window fit'}
OLD = {'RF': 'Forest published-row fit', 'MLP': 'MLP published-row fit'}


def fit_one(kind, seed, X, y, labels, folds, passes, outer):
    fit = folds != outer; held = ~fit; inner = outer % 5 + 1
    assert set(y[fit]) == set(labels) and not set(passes[fit]) & set(passes[held])
    detail = dict(outer_fold=outer, outer_fit_rows=int(fit.sum()), outer_held_rows=int(held.sum()),
                  outer_fit_passes=sorted(set(passes[fit].tolist())), outer_held_passes=sorted(set(passes[held].tolist())),
                  outer_fit_support=[int((y[fit] == v).sum()) for v in labels],
                  outer_held_support=[int((y[held] == v).sum()) for v in labels])
    if kind == 'RF':
        pipe = model('RF', seed); pipe.fit(X[fit], y[fit])
    else:
        inner_fit = fit & (folds != inner); stop = folds == inner
        assert set(y[inner_fit]) == set(labels) and set(y[stop]) == set(labels)
        for a, b in [(inner_fit, stop), (inner_fit, held), (stop, held)]:
            assert not set(passes[a]) & set(passes[b])
        epoch, history = fit_inner(X, y, inner_fit, stop, labels, seed)
        pipe = refit_outer(X, y, fit, labels, seed, epoch)
        med = np.nanmedian(X[inner_fit], axis=0)
        detail.update(inner_fold=inner, inner_fit_rows=int(inner_fit.sum()), inner_stop_rows=int(stop.sum()),
                      inner_fit_passes=sorted(set(passes[inner_fit].tolist())), inner_stop_passes=sorted(set(passes[stop].tolist())),
                      inner_fit_support=[int((y[inner_fit] == v).sum()) for v in labels],
                      inner_stop_support=[int((y[stop] == v).sum()) for v in labels],
                      selected_epoch=epoch, stopped_epoch=len(history), hit_epoch_cap=len(history) == MAX_EPOCHS,
                      inner_log_loss=history, inner_expected_median=med.tolist(),
                      inner_expected_scale_mean=np.mean(np.where(np.isnan(X[inner_fit]), med, X[inner_fit]), axis=0).tolist(),
                      outer_scale_mean=pipe[1].mean_.tolist(), outer_scale=pipe[1].scale_.tolist())
    assert np.array_equal(pipe.classes_, labels)
    np.testing.assert_allclose(pipe[0].statistics_, np.nanmedian(X[fit], axis=0))
    detail['outer_median'] = pipe[0].statistics_.tolist()
    return pipe, detail


def run():
    targets = [OUT/n for n in ['window_model_comparison.json', 'window_model_predictions.csv']]
    if any(p.exists() for p in targets): raise RuntimeError('Comparison outputs exist; refusing to overwrite')
    manifest = json.loads((OUT/'window_dataset_v1.json').read_text())
    for f, digest in manifest['artifact_sha256'].items():
        assert hashlib.sha256((OUT/f).read_bytes()).hexdigest() == digest
    rows = read('training_windows_v1.csv')
    assert len(rows) == 1393 and all(r['split'] == 'train' and 1 <= int(r['pass_index']) <= 10 for r in rows)
    assert manifest['classifier_features'] == FEATURES
    ids = [r['timestamp_unix_ms'] for r in rows]; assert len(set(ids)) == len(ids)
    assert len({(r['node_id'], r['session_id'], r['analyzer_window_id']) for r in rows}) == len(rows)
    X = feature_matrix(rows); y = np.array([r['segment_id'] for r in rows]); labels = np.unique(y)
    folds = np.array([int(r['cv_fold']) for r in rows]); passes = np.array([int(r['pass_index']) for r in rows])
    base = json.loads((OUT/'group_stopped_mlp.json').read_text())
    assert labels.tolist() == base['labels'] and SEEDS == base['seeds']
    assert json.loads(json.dumps(MLP)) == base['mlp_settings']
    assert manifest['training_rows_sha256'] == base['training_rows_sha256']
    assert sklearn.__version__ == base['versions']['sklearn']
    needed = set(ids)
    points = {r['timestamp_unix_ms']: r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in needed}
    assert set(points) == needed
    roads = {f['properties']['segment_id']: f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    geo = geography(rows, labels, ids, roads, points)
    prior = {'RF': {(int(r['seed']), r['timestamp_unix_ms']): r for r in read('forest_leaf_scores_predictions.csv') if int(r['leaf']) == 2 and r['timestamp_unix_ms'] in needed},
             'MLP': {(int(r['seed']), r['timestamp_unix_ms']): r for r in read('group_stopped_mlp_predictions.csv') if r['timestamp_unix_ms'] in needed}}
    runs = []; predictions = []; probabilities = {}
    for kind in ['RF', 'MLP']:
        assert set(prior[kind]) == {(s, t) for s in SEEDS for t in ids}
        for seed in SEEDS:
            p = np.empty((len(y), len(labels))); seen = np.zeros(len(y), int); details = []
            for outer in range(1, 6):
                pipe, detail = fit_one(kind, seed, X, y, labels, folds, passes, outer)
                held = folds == outer; p[held] = pipe.predict_proba(X[held]); seen[held] += 1; details.append(detail)
                print(f'{kind} seed={seed} fold={outer}' + (f' epoch={detail["selected_epoch"]}' if kind == 'MLP' else ''), flush=True)
            assert np.all(seen == 1)
            old_rows = [prior[kind][seed, t] for t in ids]
            for row, old in zip(rows, old_rows):
                for k in ['segment_id', 'cv_fold', 'pass_index']: assert row[k] == old[k]
            oldp = np.array([json.loads(r['probabilities']) for r in old_rows])
            assert np.array_equal(labels[oldp.argmax(axis=1)], [r['predicted_segment_id'] for r in old_rows])
            for name, pp, training in [(NEW[kind], p, details), (OLD[kind], oldp, [])]:
                assert np.isfinite(pp).all() and (pp >= 0).all() and np.allclose(pp.sum(axis=1), 1)
                m, d = evaluate(y, pp, labels, geo)
                by_fold = [dict(fold=f, support=int((folds == f).sum()), metrics=evaluate(y[folds == f], pp[folds == f], labels, geo[folds == f])[0]) for f in range(1, 6)]
                runs.append(dict(model=name, seed=seed, metrics=m, folds=by_fold, training=training))
                probabilities[name, seed] = pp
                if name == NEW[kind]:
                    for i, row in enumerate(rows):
                        predictions.append(dict(model=name, seed=seed, timestamp_unix_ms=ids[i], pass_index=int(passes[i]), cv_fold=int(folds[i]),
                                                segment_id=y[i], predicted_segment_id=labels[pp[i].argmax()], probabilities=json.dumps(pp[i].tolist()), distance_m=float(d[i])))
            print(f'{NEW[kind]} seed={seed}: {runs[-2]["metrics"]["accuracy"]:.3f} accuracy, {runs[-2]["metrics"]["macro_recall"]:.3f} recall', flush=True)
    paired = []
    for a, b in [(NEW['RF'], NEW['MLP']), (OLD['RF'], NEW['RF']), (OLD['MLP'], NEW['MLP'])]:
        for seed in SEEDS:
            ap, bp = probabilities[a, seed], probabilities[b, seed]
            for f in [0, 1, 2, 3, 4, 5]:
                mask = np.ones(len(y), bool) if f == 0 else folds == f
                am = evaluate(y[mask], ap[mask], labels, geo[mask])[0]; bm = evaluate(y[mask], bp[mask], labels, geo[mask])[0]
                ac = labels[ap[mask].argmax(axis=1)] == y[mask]; bc = labels[bp[mask].argmax(axis=1)] == y[mask]
                paired.append(dict(reference=a, candidate=b, seed=seed, fold=f, support=int(mask.sum()), delta={k: bm[k]-am[k] for k in KEYS},
                                   reference_only_correct=int((ac & ~bc).sum()), candidate_only_correct=int((~ac & bc).sum()),
                                   both_correct=int((ac & bc).sum()), both_wrong=int((~ac & ~bc).sum())))
    names = [NEW['RF'], NEW['MLP'], OLD['RF'], OLD['MLP']]
    summary = [dict(model=name, **{k: float(np.mean([r['metrics'][k] for r in runs if r['model'] == name])) for k in KEYS}) for name in names]
    report = dict(protocol='Fixed paired grouped CV on 1393 training windows; validation/test excluded.',
                  features=FEATURES, labels=labels.tolist(), display_names=base['display_names'], support=[int((y == v).sum()) for v in labels],
                  seeds=SEEDS, rf_settings=RF, mlp_settings=MLP,
                  stopping=dict(max_epochs=MAX_EPOCHS, min_delta=MIN_DELTA, patience=PATIENCE, inner_fold='outer % 5 + 1', objective='unweighted log loss'),
                  window_csv_sha256=hashlib.sha256((OUT/'training_windows_v1.csv').read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256((OUT/'window_dataset_v1.json').read_bytes()).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dependency_sha256={f: hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in ['compare_models.py', 'group_stopped_mlp.py', 'optimize_models.py', 'inspect_forest_leaf_scores.py', 'segment_road.py']},
                  reference_sha256={f: hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in ['group_stopped_mlp.json', 'forest_leaf_scores_predictions.csv', 'group_stopped_mlp_predictions.csv', 'road_segments.geojson']},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
                  runs=runs, summary=summary, paired=paired,
                  limitations=['Reused training folds within one capture; seeds and windows are not independent datasets.',
                               'Label/window physical-time alignment and live feature availability remain unverified.',
                               'All four configurations are scored on identical windows; historical row-weighted pooled scores are not comparable.',
                               'Refitting changes effective weights, preprocessing and optimization paths; causal effects are not isolated.',
                               'Wrong-only geographic subsets differ; distances use actual point to predicted centerline.',
                               'No calibration, authenticity threshold, spoof detection or final model selection.'])
    targets[0].write_text(json.dumps(report, indent=2)+'\n')
    with targets[1].open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1): run()
