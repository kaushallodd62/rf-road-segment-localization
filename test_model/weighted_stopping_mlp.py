#!/usr/bin/env python3
"""One training-CV experiment: fitting-class-weighted MLP stopping loss."""
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from compare_models import FEATURES, MLP, OUT, SEEDS, read
from group_stopped_mlp import MAX_EPOCHS, MIN_DELTA, PATIENCE, choose_epoch, make_mlp
from optimize_models import evaluate, feature_matrix, geography, load_split
from weighted_group_mlp import fitting_weights, train_weighted

NAME = 'MLP weighted stopping'
BASELINE = 'MLP weighted'
KEYS = ['accuracy', 'macro_recall', 'macro_f1', 'top3_accuracy', 'log_loss',
        'multiclass_brier', 'all_distance_median', 'all_distance_p90',
        'wrong_distance_median', 'wrong_distance_p90']


def counts(y, labels):
    return [int((y == label).sum()) for label in labels]


def stop_weights(y, labels, class_weights):
    """Apply inner-fitting weights; held labels never estimate class weights."""
    assert set(y) <= set(labels)
    return class_weights[np.searchsorted(labels, y)]


def fit_inner(X, y, fit, held, labels, seed):
    assert not np.any(fit & held) and fit.any() and held.any()
    pipe = make_mlp(seed)
    a = pipe[:-1].fit_transform(X[fit])
    b = pipe[:-1].transform(X[held])
    w, n, cw = fitting_weights(y[fit], labels)
    sw = stop_weights(y[held], labels, cw)
    np.testing.assert_allclose(pipe[0].statistics_, np.nanmedian(X[fit], axis=0))
    np.testing.assert_allclose(pipe[1].mean_, np.mean(pipe[0].transform(X[fit]), axis=0))
    weighted = []; ordinary = []; best = float('inf'); selected = 0; stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        pipe[-1].partial_fit(a, y[fit], sample_weight=w, classes=labels)
        p = pipe[-1].predict_proba(b)
        loss = float(log_loss(y[held], p, labels=labels, sample_weight=sw))
        ordinary.append(float(log_loss(y[held], p, labels=labels)))
        weighted.append(loss)
        assert np.isfinite(loss) and np.isfinite(ordinary[-1])
        if loss < best - MIN_DELTA:
            best = loss; selected = epoch; stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    assert choose_epoch(weighted) == (selected, len(weighted))
    return selected, dict(selected_epoch=selected, stopped_epoch=len(weighted),
                         hit_epoch_cap=len(weighted) == MAX_EPOCHS,
                         weighted_log_loss=weighted, unweighted_log_loss=ordinary,
                         counts=n.tolist(), class_weights=cw.tolist(),
                         median=pipe[0].statistics_.tolist(), scale_mean=pipe[1].mean_.tolist())


def run():
    targets = [OUT / f'weighted_stopping_mlp.{ext}' for ext in ('json', 'csv')]
    if any(p.exists() for p in targets):
        raise RuntimeError('Experiment outputs exist; refusing to overwrite')
    rows = load_split('train')
    assert len(rows) == 1613
    base = json.loads((OUT / 'weighted_group_mlp.json').read_text())
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    assert digest == base['training_rows_sha256']
    assert FEATURES == base['features'] and SEEDS == base['seeds']
    assert json.loads(json.dumps(MLP)) == base['mlp_settings']
    assert sklearn.__version__ == base['versions']['sklearn']
    assert hashlib.sha256((Path(__file__).parent / 'weighted_group_mlp.py').read_bytes()).hexdigest() == base['code_sha256']
    assert hashlib.sha256((OUT / 'group_stopped_mlp.json').read_bytes()).hexdigest() == base['source_hashes']['group_stopped_mlp.json']
    ids = [r['timestamp_unix_ms'] for r in rows]
    assert len(set(ids)) == len(ids)
    X = feature_matrix(rows); y = np.array([r['segment_id'] for r in rows]); labels = np.unique(y)
    assert labels.tolist() == base['labels']
    folds = np.array([int(r['cv_fold']) for r in rows])
    passes = np.array([int(r['pass_index']) for r in rows])
    assert set(passes) == set(range(1, 11)) and set(folds) == set(range(1, 6))
    needed = set(ids)
    points = {r['timestamp_unix_ms']: r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in needed}
    roads = {f['properties']['segment_id']: f['geometry']['coordinates'] for f in json.loads((OUT / 'road_segments.geojson').read_text())['features']}
    geo = geography(rows, labels, ids, roads, points)
    saved = {(int(r['seed']), r['timestamp_unix_ms']): r for r in read('weighted_group_mlp_predictions.csv')}
    assert set(saved) == {(seed, t) for seed in SEEDS for t in ids}
    runs = []; output = []; paired = []
    for seed in SEEDS:
        p = np.empty((len(y), len(labels))); seen = np.zeros(len(y), int); details = []
        for outer in range(1, 6):
            inner = outer % 5 + 1
            fit = (folds != outer) & (folds != inner); stop = folds == inner
            outer_fit = folds != outer; held = ~outer_fit
            for a, b in ((fit, stop), (fit, held), (stop, held)):
                assert not set(passes[a]) & set(passes[b])
            epoch, detail = fit_inner(X, y, fit, stop, labels, seed)
            pipe, meta, _, _ = train_weighted(X, y, outer_fit, labels, seed, epochs=epoch)
            assert np.array_equal(pipe.classes_, labels)
            p[held] = pipe.predict_proba(X[held]); seen[held] += 1
            detail.update(fold=outer, inner_fold=inner, outer_preprocessing_and_weights=meta,
                          class_support={k: counts(y[m], labels) for k, m in [('inner_fit', fit), ('inner_stop', stop), ('outer_fit', outer_fit), ('outer_held', held)]},
                          pass_groups={k: sorted(set(passes[m].tolist())) for k, m in [('inner_fit', fit), ('inner_stop', stop), ('outer_fit', outer_fit), ('outer_held', held)]})
            details.append(detail)
            print(f'seed={seed} fold={outer} selected={epoch} stopped={detail["stopped_epoch"]}', flush=True)
        assert np.all(seen == 1) and np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1)
        prior = [saved[seed, t] for t in ids]
        for r, old in zip(rows, prior):
            for k in ('segment_id', 'cv_fold', 'pass_index'):
                assert r[k] == old[k]
        oldp = np.array([json.loads(r['probabilities']) for r in prior])
        old_run = next(r for r in base['runs'] if r['model'] == BASELINE and r['seed'] == seed)
        for k in KEYS:
            np.testing.assert_allclose(evaluate(y, oldp, labels, geo)[0][k], old_run['metrics'][k])
        metrics, d = evaluate(y, p, labels, geo)
        fold_scores = [dict(fold=f, support=int((folds == f).sum()), metrics=evaluate(y[folds == f], p[folds == f], labels, geo[folds == f])[0]) for f in range(1, 6)]
        runs.append(dict(model=NAME, seed=seed, metrics=metrics, folds=fold_scores, stopping=details))
        for f in (0, 1, 2, 3, 4, 5):
            m = np.ones(len(y), bool) if f == 0 else folds == f
            bm = evaluate(y[m], oldp[m], labels, geo[m])[0]; nm = evaluate(y[m], p[m], labels, geo[m])[0]
            a = labels[oldp[m].argmax(axis=1)] == y[m]; b = labels[p[m].argmax(axis=1)] == y[m]
            paired.append(dict(seed=seed, fold=f, support=int(m.sum()), delta={k: nm[k] - bm[k] for k in KEYS},
                               baseline_only_correct=int((a & ~b).sum()), candidate_only_correct=int((~a & b).sum()),
                               both_correct=int((a & b).sum()), both_wrong=int((~a & ~b).sum())))
        for i, r in enumerate(rows):
            output.append(dict(seed=seed, timestamp_unix_ms=ids[i], cv_fold=int(folds[i]), pass_index=int(passes[i]),
                               segment_id=y[i], predicted_segment_id=labels[p[i].argmax()], probabilities=json.dumps(p[i].tolist()), distance_m=float(d[i])))
    references = [dict(model=r['model'], seed=r['seed'], metrics=r['metrics']) for r in base['runs']]
    all_runs = references + runs
    names = list(dict.fromkeys(r['model'] for r in all_runs))
    summary = [dict(model=n, **{k: float(np.mean([r['metrics'][k] for r in all_runs if r['model'] == n])) for k in KEYS}) for n in names]
    source_names = ['weighted_group_mlp.json', 'weighted_group_mlp_predictions.csv', 'road_segments.geojson']
    report = dict(protocol='Training-only grouped CV; change stopping objective only; validation/test excluded.',
                  features=FEATURES, labels=labels.tolist(), support=counts(y, labels), display_names=base['display_names'],
                  seeds=SEEDS, mlp_settings=MLP, training_rows_sha256=digest,
                  stopping=dict(patience=PATIENCE, min_delta=MIN_DELTA, max_epochs=MAX_EPOCHS, inner_fold='outer % 5 + 1',
                                objective='Held log loss weighted by inner-fitting N/(K*n_class); normalized by held total weight.'),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dependency_hashes={n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest() for n in ['group_stopped_mlp.py', 'weighted_group_mlp.py', 'optimize_models.py', 'compare_models.py', 'inspect_forest_leaf_scores.py', 'segment_road.py']},
                  source_hashes={n: hashlib.sha256((OUT / n).read_bytes()).hexdigest() for n in source_names},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
                  runs=runs, references=references, summary=summary, paired=paired,
                  limitations=['Same-capture inferred passes and repeated seeds; reused folds risk selection bias.',
                               'Fitting-class-weighted stopping is a surrogate, not macro recall or held-class-balanced loss.',
                               'Missing stopping labels cannot guide selection; small supports make stopping noisy.',
                               'Inner-to-outer refit changes class counts and minibatch updates per epoch in both methods.',
                               'Wrong-only distance subsets differ; distances are point to predicted centerline.',
                               'Freshness and BSM-rate provenance and decision-time availability remain unresolved.',
                               'No score calibration, spoof detection or final model selection.'])
    targets[0].write_text(json.dumps(report, indent=2) + '\n')
    with targets[1].open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
