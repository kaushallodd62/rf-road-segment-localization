#!/usr/bin/env python3
"""Bounded gradient-free output-head edits, nested within training pass groups."""
import copy
import csv
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.special import logsumexp, softmax
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from compare_models import FEATURES, MLP, OUT, SEEDS, read
from compare_window_models import KEYS
from optimize_models import evaluate, feature_matrix, geography
from sqrt_group_mlp import train_weighted

STEPS = 128
START_STEP = .2
END_STEP = .02
MAX_LOGIT_RMS = 1.
NAMES = ['MLP matched checkpoint', 'MLP bias edits', 'MLP head edits']


def hidden_design(pipe, X):
    """Frozen ReLU representation; last row of head parameters is its bias."""
    assert pipe[-1].activation == 'relu' and pipe[-1].out_activation_ == 'softmax'
    h = pipe[:-1].transform(X)
    for w, b in zip(pipe[-1].coefs_[:-1], pipe[-1].intercepts_[:-1]):
        h = np.maximum(h @ w + b, 0)
    return np.column_stack([h, np.ones(len(h))])


def objective(logits, yi):
    counts = np.bincount(yi, minlength=logits.shape[1])
    assert (counts > 0).all() and np.isfinite(logits).all()
    weights = 1 / counts[yi]
    recall = float(np.sum(weights * (logits.argmax(axis=1) == yi)) / len(counts))
    loss = float(np.average(logsumexp(logits, axis=1) - logits[np.arange(len(yi)), yi], weights=weights))
    return recall, -loss


def edit_head(design, yi, head, seed, bias_only):
    """Paired Gaussian hill climbing. No optimizer, gradients, or fitting calls."""
    rng = np.random.default_rng(seed)
    base_logits = design @ head
    delta = np.zeros_like(head)
    best = objective(base_logits, yi)
    trace = [dict(step=0, macro_recall=best[0], balanced_log_loss=-best[1], accepted_sign=0)]
    for step, size in enumerate(np.geomspace(START_STEP, END_STEP, STEPS), 1):
        direction = np.zeros_like(head)
        if bias_only:
            direction[-1] = rng.normal(size=head.shape[1])
        else:
            direction[:] = rng.normal(size=head.shape)
        direction -= direction.mean(axis=1, keepdims=True)
        direction *= size / max(float(np.sqrt(np.mean((design @ direction)**2))), 1e-12)
        chosen = delta
        sign = 0
        # Both proposals originate at the same incumbent.
        for s in [1, -1]:
            proposal = delta + s * direction
            rms = float(np.sqrt(np.mean((design @ proposal)**2)))
            proposal *= min(1., MAX_LOGIT_RMS / max(rms, 1e-12))
            score = objective(base_logits + design @ proposal, yi)
            if score > best:
                best = score; chosen = proposal; sign = s
        delta = chosen
        trace.append(dict(step=step, macro_recall=best[0], balanced_log_loss=-best[1], accepted_sign=sign))
    return delta, trace


def apply_delta(pipe, delta):
    edited = copy.deepcopy(pipe)
    edited[-1].coefs_[-1] += delta[:-1]
    edited[-1].intercepts_[-1] += delta[-1]
    return edited


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    targets = [OUT / ('direct_weight_edits' + ext) for ext in ['.json', '_predictions.csv', '.joblib']]
    if any(p.exists() for p in targets):
        raise RuntimeError('Direct-edit outputs exist; refusing to overwrite')
    reference = json.loads((OUT / 'window_moderate_weights.json').read_text())
    manifest = json.loads((OUT / 'window_dataset_v1.json').read_text())
    assert digest(OUT / 'training_windows_v1.csv') == reference['window_csv_sha256']
    assert digest(OUT / 'window_dataset_v1.json') == reference['manifest_sha256']
    assert FEATURES == reference['features'] == manifest['classifier_features']
    assert SEEDS == reference['seeds'] and json.loads(json.dumps(MLP)) == reference['mlp_settings']
    assert sklearn.__version__ == reference['versions']['sklearn'] and np.__version__ == reference['versions']['numpy']
    for name, value in reference['dependency_sha256'].items():
        assert digest(Path(__file__).parent / name) == value
    rows = read('training_windows_v1.csv')
    assert len(rows) == 1393 and all(r['split'] == 'train' for r in rows)
    ids = [r['timestamp_unix_ms'] for r in rows]
    assert len(set(ids)) == len(ids)
    X = feature_matrix(rows); y = np.array([r['segment_id'] for r in rows]); labels = np.unique(y)
    assert labels.tolist() == reference['labels']
    yi = np.searchsorted(labels, y)
    folds = np.array([int(r['cv_fold']) for r in rows]); passes = np.array([int(r['pass_index']) for r in rows])
    assert set(passes) == set(range(1, 11)) and set(folds) == set(range(1, 6))
    # Restrict coordinate lookup to training-window IDs, for scoring only.
    needed = set(ids)
    with (OUT / 'location_points_segmented.csv').open(newline='') as f:
        points = {r['timestamp_unix_ms']: r for r in csv.DictReader(f) if r['timestamp_unix_ms'] in needed}
    assert set(points) == needed
    roads = {f['properties']['segment_id']: f['geometry']['coordinates'] for f in json.loads((OUT / 'road_segments.geojson').read_text())['features']}
    geo = geography(rows, labels, ids, roads, points)
    runs = []; predictions = []; models = {}; paired = []
    for seed in SEEDS:
        saved = next(r for r in reference['runs'] if r['model'] == 'MLP moderate weights' and r['seed'] == seed)
        probs = {name: np.empty((len(y), len(labels))) for name in NAMES}
        details = []; seen = np.zeros(len(y), int)
        for outer in range(1, 6):
            inner = outer % 5 + 1
            fit = (folds != outer) & (folds != inner); adapt = folds == inner; held = folds == outer
            for a, b in [(fit, adapt), (fit, held), (adapt, held)]:
                assert not set(passes[a]) & set(passes[b])
            assert all(set(y[m]) == set(labels) for m in [fit, adapt, held])
            old = next(d for d in saved['training'] if d['outer_fold'] == outer)
            # Recreate the selected inner checkpoint, not the later stopped state.
            pipe, meta, _, _ = train_weighted(X, y, fit, labels, seed, epochs=old['selected_epoch'])
            design = hidden_design(pipe, X[adapt])
            head = np.vstack([pipe[-1].coefs_[-1], pipe[-1].intercepts_[-1]])
            np.testing.assert_allclose(softmax(design @ head, axis=1), pipe.predict_proba(X[adapt]), atol=1e-12)
            models[seed, outer, NAMES[0]] = pipe
            probs[NAMES[0]][held] = pipe.predict_proba(X[held])
            detail = dict(outer_fold=outer, inner_fold=inner, selected_epoch=old['selected_epoch'],
                          preprocessing_and_weights=meta,
                          pass_groups={k: sorted(set(passes[m].tolist())) for k, m in [('fit', fit), ('adapt', adapt), ('held', held)]},
                          support={k: np.bincount(yi[m], minlength=len(labels)).tolist() for k, m in [('fit', fit), ('adapt', adapt), ('held', held)]}, edits={})
            for name, bias_only in [(NAMES[1], True), (NAMES[2], False)]:
                delta, trace = edit_head(design, yi[adapt], head, seed * 100 + outer, bias_only)
                edited = apply_delta(pipe, delta)
                models[seed, outer, name] = edited
                probs[name][held] = edited.predict_proba(X[held])
                detail['edits'][name] = dict(delta=delta.tolist(), trace=trace,
                                           logit_rms=float(np.sqrt(np.mean((design @ delta)**2))),
                                           changed_parameters=int(np.count_nonzero(delta)))
            details.append(detail); seen[held] += 1
            print(f'direct edits seed={seed} fold={outer} epoch={old["selected_epoch"]}', flush=True)
        assert np.all(seen == 1)
        for name, p in probs.items():
            assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1)
            metrics, distances = evaluate(y, p, labels, geo)
            metrics['confusion_matrix'] = confusion_matrix(y, labels[p.argmax(axis=1)], labels=labels).tolist()
            by_fold = [dict(fold=f, metrics=evaluate(y[folds == f], p[folds == f], labels, geo[folds == f])[0]) for f in range(1, 6)]
            for f in by_fold:
                m = folds == f['fold']
                f['metrics']['confusion_matrix'] = confusion_matrix(y[m], labels[p[m].argmax(axis=1)], labels=labels).tolist()
            runs.append(dict(model=name, seed=seed, metrics=metrics, folds=by_fold, training=details if name == NAMES[0] else []))
            for i, r in enumerate(rows):
                predictions.append(dict(model=name, seed=seed, timestamp_unix_ms=ids[i], cv_fold=int(folds[i]), pass_index=int(passes[i]),
                                        segment_id=y[i], predicted_segment_id=labels[p[i].argmax()], probabilities=json.dumps(p[i].tolist()), distance_m=float(distances[i])))
            if name != NAMES[0]:
                for f in range(6):
                    m = np.ones(len(y), bool) if f == 0 else folds == f
                    bm = evaluate(y[m], probs[NAMES[0]][m], labels, geo[m])[0]
                    nm = evaluate(y[m], p[m], labels, geo[m])[0]
                    a = probs[NAMES[0]][m].argmax(axis=1) == yi[m]; b = p[m].argmax(axis=1) == yi[m]
                    paired.append(dict(model=name, seed=seed, fold=f, support=int(m.sum()), delta={k: nm[k] - bm[k] for k in KEYS},
                                       baseline_only_correct=int((a & ~b).sum()), edited_only_correct=int((~a & b).sum())))
    summary = [dict(model=n, **{k: float(np.mean([r['metrics'][k] for r in runs if r['model'] == n])) for k in KEYS}) for n in NAMES]
    deps = ['compare_models.py', 'compare_window_models.py', 'sqrt_group_mlp.py', 'group_stopped_mlp.py', 'optimize_models.py', 'inspect_forest_leaf_scores.py', 'segment_road.py', 'evaluate_weighted_knn.py']
    report = dict(protocol='Training-window nested grouped CV; same inner checkpoint for each paired edit; no outer refit, validation or test.',
                  primary_objective='macro_recall', features=FEATURES, labels=labels.tolist(), display_names=reference['display_names'], support=reference['support'], seeds=SEEDS,
                  search=dict(steps=STEPS, proposals_per_step=2, start_step=START_STEP, end_step=END_STEP, max_inner_logit_rms=MAX_LOGIT_RMS,
                              objective='Inner macro recall; ties minimize class-balanced log loss; no edit included.', method='Paired Gaussian hill climbing; bias-only control and full output-head edit.'),
                  versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
                  code_sha256=digest(Path(__file__)), dependency_sha256={f: digest(Path(__file__).parent / f) for f in deps},
                  source_sha256={f: digest(OUT / f) for f in ['training_windows_v1.csv', 'window_dataset_v1.json', 'window_moderate_weights.json', 'road_segments.geojson', 'location_points_segmented.csv']},
                  runs=runs, paired=paired, summary=summary, contextual_references=reference['summary'],
                  limitations=['Baseline initialization uses Adam; only the subsequent edits are gradient-free.',
                               'Inner adaptation labels also choose the existing epoch; their gains are optimization scores.',
                               'Matched models train on six passes, adapt on two and evaluate on two; full-refit references train on eight.',
                               'Reused same-capture folds and repeated seeds do not establish independent-drive generalization.',
                               'No LLM proposes numeric weights here; this tests the derivative-free mechanism, not a learned self-editor.',
                               'Head-only search does not test hidden-layer or from-scratch gradient-free training.',
                               'Timing, label alignment and freshness/BSM-rate availability remain unresolved; no calibration or spoof detection.'])
    joblib.dump(dict(features=FEATURES, labels=labels, models=models), targets[2], compress=3)
    report['model_bundle_sha256'] = digest(targets[2])
    targets[0].write_text(json.dumps(report, indent=2) + '\n')
    with targets[1].open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
