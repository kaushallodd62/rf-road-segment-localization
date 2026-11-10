#!/usr/bin/env python3
"""One eight-versus-ten-input RF/MLP comparison on training windows only."""
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from compare_models import FEATURES, MLP, OUT, RF, SEEDS, read
from compare_window_models import KEYS, NEW, fit_one
from optimize_models import evaluate, geography

REMOVED = ['message_freshness_ms', 'bsm_rate_per_sec']
INPUTS = [f for f in FEATURES if f not in REMOVED]
NAMES = {'RF': 'Forest eight inputs', 'MLP': 'MLP eight inputs'}
REFERENCES = {'RF': 'Forest ten inputs', 'MLP': 'MLP ten inputs'}


def input_matrix(rows):
    """An explicit allowlist prevents metadata or removed fields entering X."""
    X = np.array([[float(r[f]) if r[f] else np.nan for f in INPUTS] for r in rows])
    assert X.shape == (len(rows), 8) and not np.isinf(X).any()
    return X


def run():
    targets = [OUT/n for n in ['window_feature_dependency.json', 'window_feature_dependency_predictions.csv']]
    if any(p.exists() for p in targets): raise RuntimeError('Dependency outputs exist; refusing to overwrite')
    base = json.loads((OUT/'window_model_comparison.json').read_text())
    manifest = json.loads((OUT/'window_dataset_v1.json').read_text())
    for f, digest in manifest['artifact_sha256'].items():
        assert hashlib.sha256((OUT/f).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((OUT/'training_windows_v1.csv').read_bytes()).hexdigest() == base['window_csv_sha256']
    assert hashlib.sha256((OUT/'window_dataset_v1.json').read_bytes()).hexdigest() == base['manifest_sha256']
    assert hashlib.sha256((Path(__file__).parent/'compare_window_models.py').read_bytes()).hexdigest() == base['code_sha256']
    assert FEATURES == base['features'] == manifest['classifier_features'] and SEEDS == base['seeds']
    assert RF == base['rf_settings'] and json.loads(json.dumps(MLP)) == base['mlp_settings']
    assert sklearn.__version__ == base['versions']['sklearn'] and np.__version__ == base['versions']['numpy']
    rows = read('training_windows_v1.csv')
    assert len(rows) == 1393 and all(r['split'] == 'train' and 1 <= int(r['pass_index']) <= 10 for r in rows)
    ids = [r['timestamp_unix_ms'] for r in rows]; needed = set(ids); assert len(needed) == len(rows)
    X = input_matrix(rows); y = np.array([r['segment_id'] for r in rows]); labels = np.unique(y)
    assert labels.tolist() == base['labels']
    folds = np.array([int(r['cv_fold']) for r in rows]); passes = np.array([int(r['pass_index']) for r in rows])
    points = {r['timestamp_unix_ms']: r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in needed}
    assert set(points) == needed
    roads = {f['properties']['segment_id']: f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    geo = geography(rows, labels, ids, roads, points)
    saved = {(r['model'], int(r['seed']), r['timestamp_unix_ms']): r for r in read('window_model_predictions.csv')}
    assert set(saved) == {(NEW[k], s, t) for k in ['RF', 'MLP'] for s in SEEDS for t in ids}
    runs = []; predictions = []; paired = []
    for kind in ['RF', 'MLP']:
        for seed in SEEDS:
            p = np.empty((len(y), len(labels))); seen = np.zeros(len(y), int); details = []
            for outer in range(1, 6):
                pipe, detail = fit_one(kind, seed, X, y, labels, folds, passes, outer)
                held = folds == outer; p[held] = pipe.predict_proba(X[held]); seen[held] += 1
                assert pipe.n_features_in_ == 8
                details.append(detail)
                print(f'{kind} eight inputs seed={seed} fold={outer}' + (f' epoch={detail["selected_epoch"]}' if kind == 'MLP' else ''), flush=True)
            assert np.all(seen == 1)
            oldrows = [saved[NEW[kind], seed, t] for t in ids]
            for a, b in zip(rows, oldrows):
                for k in ['segment_id', 'cv_fold', 'pass_index']: assert a[k] == b[k]
            oldp = np.array([json.loads(r['probabilities']) for r in oldrows])
            assert np.array_equal(labels[oldp.argmax(axis=1)], [r['predicted_segment_id'] for r in oldrows])
            oldrun = next(r for r in base['runs'] if r['model'] == NEW[kind] and r['seed'] == seed)
            for name, pp, training in [(REFERENCES[kind], oldp, []), (NAMES[kind], p, details)]:
                assert np.isfinite(pp).all() and (pp >= 0).all() and np.allclose(pp.sum(axis=1), 1)
                m, d = evaluate(y, pp, labels, geo)
                if name == REFERENCES[kind]:
                    for k in KEYS: np.testing.assert_allclose(m[k], oldrun['metrics'][k])
                by_fold = [dict(fold=f, support=int((folds == f).sum()), metrics=evaluate(y[folds == f], pp[folds == f], labels, geo[folds == f])[0]) for f in range(1, 6)]
                runs.append(dict(model=name, kind=kind, seed=seed, inputs=INPUTS if training else FEATURES, metrics=m, folds=by_fold, training=training))
                if training:
                    for i, row in enumerate(rows):
                        predictions.append(dict(model=name, seed=seed, timestamp_unix_ms=ids[i], pass_index=int(passes[i]), cv_fold=int(folds[i]),
                                                segment_id=y[i], predicted_segment_id=labels[pp[i].argmax()], probabilities=json.dumps(pp[i].tolist()), distance_m=float(d[i])))
            for f in [0, 1, 2, 3, 4, 5]:
                ix = np.ones(len(y), bool) if f == 0 else folds == f
                om = evaluate(y[ix], oldp[ix], labels, geo[ix])[0]; nm = evaluate(y[ix], p[ix], labels, geo[ix])[0]
                a = labels[oldp[ix].argmax(axis=1)] == y[ix]; b = labels[p[ix].argmax(axis=1)] == y[ix]
                paired.append(dict(kind=kind, seed=seed, fold=f, support=int(ix.sum()), delta={k: nm[k]-om[k] for k in KEYS},
                                   ten_only_correct=int((a & ~b).sum()), eight_only_correct=int((~a & b).sum()),
                                   both_correct=int((a & b).sum()), both_wrong=int((~a & ~b).sum())))
    names = [REFERENCES['RF'], NAMES['RF'], REFERENCES['MLP'], NAMES['MLP']]
    summary = [dict(model=name, **{k: float(np.mean([r['metrics'][k] for r in runs if r['model'] == name])) for k in KEYS}) for name in names]
    report = dict(protocol='Drop freshness and BSM rate together; fixed paired training-window CV; validation/test excluded.',
                  features=INPUTS, reference_features=FEATURES, removed_features=REMOVED, labels=labels.tolist(),
                  display_names=base['display_names'], support=base['support'], seeds=SEEDS,
                  rf_settings=RF, mlp_settings=MLP, stopping=base['stopping'],
                  forest_candidate_counts=dict(ten_inputs=3, eight_inputs=2),
                  window_csv_sha256=base['window_csv_sha256'], manifest_sha256=base['manifest_sha256'],
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dependency_sha256={f: hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in ['compare_window_models.py', 'compare_models.py', 'group_stopped_mlp.py', 'optimize_models.py', 'inspect_forest_leaf_scores.py', 'segment_road.py']},
                  reference_sha256={f: hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in ['window_model_comparison.json', 'window_model_predictions.csv', 'road_segments.geojson']},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
                  runs=runs, summary=summary, paired=paired,
                  limitations=['Reused same-capture folds; dependent windows and repeated seeds are not independent datasets.',
                               'Two fields removed together; forest candidate count and MLP optimization change with input dimension.',
                               'Neither individual feature causality nor deployment availability of remaining inputs is established.',
                               'Label/window physical-time alignment remains unverified, including 59 last-seen gaps over one second.',
                               'Wrong-only geographic subsets differ; distances are actual point to predicted centerline.',
                               'No calibration, spoof detection, acceptance threshold or final model selection.'])
    targets[0].write_text(json.dumps(report, indent=2)+'\n')
    with targets[1].open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1): run()
