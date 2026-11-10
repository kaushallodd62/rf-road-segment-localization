#!/usr/bin/env python3
"""One moderate training-weight rule; macro recall is the primary outcome."""
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from compare_models import FEATURES, MLP, OUT, RF, SEEDS, read
from compare_window_models import KEYS, NEW
from group_stopped_mlp import MAX_EPOCHS
from moderate_forest_weights import fit_forest
from optimize_models import evaluate, feature_matrix, geography
from sqrt_group_mlp import train_weighted

NAMES = {'RF': 'Forest moderate weights', 'MLP': 'MLP moderate weights'}
REFERENCES = {'RF': 'Forest unweighted', 'MLP': 'MLP unweighted'}


def fit_weighted_one(kind, seed, X, y, labels, folds, passes, outer):
    fit = folds != outer; held = ~fit; inner = outer % 5 + 1
    assert not set(passes[fit]) & set(passes[held]) and set(y[fit]) == set(labels)
    detail = dict(outer_fold=outer, outer_fit_passes=sorted(set(passes[fit].tolist())),
                  outer_held_passes=sorted(set(passes[held].tolist())), outer_fit_rows=int(fit.sum()), outer_held_rows=int(held.sum()))
    if kind == 'RF':
        pipe, meta = fit_forest(2, 'moderate', seed, X[fit], y[fit], labels)
        meta['median'] = pipe[0].statistics_.tolist()
        detail['outer_preprocessing_and_weights'] = meta
    else:
        inner_fit = fit & (folds != inner); stop = folds == inner
        assert set(y[inner_fit]) == set(labels) and set(y[stop]) == set(labels)
        for a, b in [(inner_fit, stop), (inner_fit, held), (stop, held)]:
            assert not set(passes[a]) & set(passes[b])
        _, imeta, epoch, history = train_weighted(X, y, inner_fit, labels, seed, held=stop)
        pipe, ometa, _, _ = train_weighted(X, y, fit, labels, seed, epochs=epoch)
        detail.update(inner_fold=inner, inner_fit_passes=sorted(set(passes[inner_fit].tolist())),
                      inner_stop_passes=sorted(set(passes[stop].tolist())), inner_fit_rows=int(inner_fit.sum()), inner_stop_rows=int(stop.sum()),
                      inner_preprocessing_and_weights=imeta, outer_preprocessing_and_weights=ometa,
                      selected_epoch=epoch, stopped_epoch=len(history), hit_epoch_cap=len(history) == MAX_EPOCHS, inner_log_loss=history,
                      inner_stop_support=[int((y[stop] == v).sum()) for v in labels])
    assert np.array_equal(pipe.classes_, labels) and pipe.n_features_in_ == 10
    return pipe, detail


def run():
    targets = [OUT/n for n in ['window_moderate_weights.json', 'window_moderate_weights_predictions.csv']]
    if any(p.exists() for p in targets): raise RuntimeError('Weighting outputs exist; refusing to overwrite')
    base = json.loads((OUT/'window_model_comparison.json').read_text())
    manifest = json.loads((OUT/'window_dataset_v1.json').read_text())
    for f, digest in manifest['artifact_sha256'].items():
        assert hashlib.sha256((OUT/f).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((OUT/'training_windows_v1.csv').read_bytes()).hexdigest() == base['window_csv_sha256']
    assert hashlib.sha256((OUT/'window_dataset_v1.json').read_bytes()).hexdigest() == base['manifest_sha256']
    assert FEATURES == manifest['classifier_features'] == base['features'] and SEEDS == base['seeds']
    assert RF == base['rf_settings'] and json.loads(json.dumps(MLP)) == base['mlp_settings']
    assert sklearn.__version__ == base['versions']['sklearn'] and np.__version__ == base['versions']['numpy']
    for f, digest in base['dependency_sha256'].items():
        assert hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() == digest
    rows = read('training_windows_v1.csv')
    assert len(rows) == 1393 and all(r['split'] == 'train' and 1 <= int(r['pass_index']) <= 10 for r in rows)
    ids = [r['timestamp_unix_ms'] for r in rows]; needed = set(ids); assert len(needed) == len(rows)
    X = feature_matrix(rows); y = np.array([r['segment_id'] for r in rows]); labels = np.unique(y)
    assert labels.tolist() == base['labels']
    folds = np.array([int(r['cv_fold']) for r in rows]); passes = np.array([int(r['pass_index']) for r in rows])
    points = {r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in needed}
    assert set(points) == needed
    roads = {f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    geo = geography(rows, labels, ids, roads, points)
    saved = {(r['model'], int(r['seed']), r['timestamp_unix_ms']):r for r in read('window_model_predictions.csv')}
    assert set(saved) == {(NEW[k], s, t) for k in ['RF', 'MLP'] for s in SEEDS for t in ids}
    runs = []; predictions = []; paired = []
    for kind in ['RF', 'MLP']:
        for seed in SEEDS:
            p = np.empty((len(y), len(labels))); seen = np.zeros(len(y), int); details = []
            for outer in range(1, 6):
                pipe, detail = fit_weighted_one(kind, seed, X, y, labels, folds, passes, outer)
                held = folds == outer; p[held] = pipe.predict_proba(X[held]); seen[held] += 1; details.append(detail)
                print(f'{kind} moderate seed={seed} fold={outer}' + (f' epoch={detail["selected_epoch"]}' if kind == 'MLP' else ''), flush=True)
            assert np.all(seen == 1)
            oldrows = [saved[NEW[kind], seed, t] for t in ids]
            for a, b in zip(rows, oldrows):
                for k in ['segment_id', 'cv_fold', 'pass_index']: assert a[k] == b[k]
            oldp = np.array([json.loads(r['probabilities']) for r in oldrows])
            oldrun = next(r for r in base['runs'] if r['model'] == NEW[kind] and r['seed'] == seed)
            for name, pp, training in [(REFERENCES[kind], oldp, []), (NAMES[kind], p, details)]:
                assert np.isfinite(pp).all() and (pp >= 0).all() and np.allclose(pp.sum(axis=1), 1)
                m, d = evaluate(y, pp, labels, geo)
                if not training:
                    for k in KEYS: np.testing.assert_allclose(m[k], oldrun['metrics'][k])
                by_fold = [dict(fold=f, support=int((folds == f).sum()), metrics=evaluate(y[folds == f], pp[folds == f], labels, geo[folds == f])[0]) for f in range(1, 6)]
                runs.append(dict(model=name, kind=kind, seed=seed, metrics=m, folds=by_fold, training=training))
                if training:
                    for i, row in enumerate(rows):
                        predictions.append(dict(model=name, seed=seed, timestamp_unix_ms=ids[i], cv_fold=int(folds[i]), pass_index=int(passes[i]),
                                                segment_id=y[i], predicted_segment_id=labels[pp[i].argmax()], probabilities=json.dumps(pp[i].tolist()), distance_m=float(d[i])))
            for f in [0, 1, 2, 3, 4, 5]:
                ix = np.ones(len(y), bool) if f == 0 else folds == f
                om = evaluate(y[ix], oldp[ix], labels, geo[ix])[0]; nm = evaluate(y[ix], p[ix], labels, geo[ix])[0]
                a = labels[oldp[ix].argmax(axis=1)] == y[ix]; b = labels[p[ix].argmax(axis=1)] == y[ix]
                paired.append(dict(kind=kind, seed=seed, fold=f, support=int(ix.sum()), delta={k:nm[k]-om[k] for k in KEYS},
                                   unweighted_only_correct=int((a & ~b).sum()), weighted_only_correct=int((~a & b).sum()),
                                   both_correct=int((a & b).sum()), both_wrong=int((~a & ~b).sum())))
    names = [REFERENCES['RF'], NAMES['RF'], REFERENCES['MLP'], NAMES['MLP']]
    summary = [dict(model=name, **{k:float(np.mean([r['metrics'][k] for r in runs if r['model'] == name])) for k in KEYS}) for name in names]
    report = dict(protocol='One moderate fitting-class-weight rule on fixed ten-input window models; macro recall primary.',
                  features=FEATURES, labels=labels.tolist(), display_names=base['display_names'], support=base['support'], seeds=SEEDS,
                  primary_objective='macro_recall', rf_settings=RF, mlp_settings=MLP, stopping=base['stopping'],
                  weight_rule='1/sqrt(fitting class count), normalized to mean one across fitting rows; separate inner/outer weights.',
                  window_csv_sha256=base['window_csv_sha256'], manifest_sha256=base['manifest_sha256'],
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dependency_sha256={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in ['moderate_forest_weights.py', 'sqrt_group_mlp.py', 'compare_window_models.py', 'compare_models.py', 'group_stopped_mlp.py', 'optimize_models.py', 'inspect_forest_leaf_scores.py', 'segment_road.py']},
                  reference_sha256={f:hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in ['window_model_comparison.json', 'window_model_predictions.csv', 'road_segments.geojson']},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
                  runs=runs, summary=summary, paired=paired,
                  limitations=['Reused same-capture groups; repeated seeds and adjacent windows are dependent.',
                               'Class weighting changes objective, tree structure and MLP minibatch/regularization behavior.',
                               'MLP stopping loss stays unweighted; refit recomputes weights on larger fitting portions.',
                               'Label timing and feature live availability remain unresolved; no unseen-drive or calibration claim.',
                               'Wrong-only distance populations differ; no spoof detection, threshold or revised final model.'])
    targets[0].write_text(json.dumps(report, indent=2)+'\n')
    with targets[1].open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(predictions[0])); w.writeheader(); w.writerows(predictions)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1): run()
