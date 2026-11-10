"""Explicit development-only checks; never loads saved test evaluations."""
import hashlib
import json
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from sklearn.metrics import log_loss, recall_score

import weighted_stopping_mlp as experiment
from compare_models import OUT
from group_stopped_mlp import choose_epoch
from optimize_models import evaluate, feature_matrix, geography, load_split
from weighted_group_mlp import fitting_weights


class WeightedStoppingTests(unittest.TestCase):
    def test_weights_come_from_fit_not_stopping_distribution(self):
        labels = np.array(['a', 'b'])
        _, _, cw = fitting_weights(np.array(['a'] * 9 + ['b']), labels)
        held = np.array(['a', 'a', 'a', 'b'])
        w = experiment.stop_weights(held, labels, cw)
        np.testing.assert_allclose(w, [10/18, 10/18, 10/18, 5])
        # Ordinary loss prefers epoch 1; fitting-weighted loss prefers epoch 2.
        p1 = np.array([[.99, .01], [.99, .01], [.99, .01], [.8, .2]])
        p2 = np.array([[.1, .9], [.1, .9], [.1, .9], [.1, .9]])
        losses = [log_loss(held, p, labels=labels) for p in (p1, p2)]
        weighted = [log_loss(held, p, labels=labels, sample_weight=w) for p in (p1, p2)]
        self.assertEqual(choose_epoch(losses)[0], 1)
        self.assertEqual(choose_epoch(weighted)[0], 2)

    def test_inner_matches_legacy_optimization_path(self):
        # Compare the loss history to the legacy trainer before either can stop.
        from weighted_group_mlp import train_weighted
        labels = np.array(['a', 'b'])
        X = np.array([[np.nan, 1], [1, 0], [2, 1], [3, 0], [50, 3], [60, 2]], float)
        y = np.array(['a', 'a', 'a', 'b', 'a', 'b'])
        fit = np.array([True] * 4 + [False] * 2); held = ~fit
        with warnings.catch_warnings(), patch.object(experiment, 'MAX_EPOCHS', 3):
            warnings.filterwarnings('ignore', message='Got `batch_size`')
            epoch, detail = experiment.fit_inner(X, y, fit, held, labels, 17)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Got `batch_size`')
            _, meta, _, history = train_weighted(X, y, fit, labels, 17, held=held)
        np.testing.assert_allclose(detail['unweighted_log_loss'], history[:3], rtol=0, atol=1e-14)
        self.assertEqual(detail['counts'], meta['counts'])
        np.testing.assert_allclose(detail['median'], np.nanmedian(X[fit], axis=0))
        self.assertEqual(epoch, choose_epoch(detail['weighted_log_loss'])[0])

    def test_saved_development_results(self):
        report = json.loads((OUT / 'weighted_stopping_mlp.json').read_text())
        rows = load_split('train'); ids = [r['timestamp_unix_ms'] for r in rows]
        self.assertEqual(len(rows), 1613)
        self.assertEqual(hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(), report['training_rows_sha256'])
        labels = np.array(report['labels']); y = np.array([r['segment_id'] for r in rows])
        folds = np.array([int(r['cv_fold']) for r in rows]); passes = np.array([int(r['pass_index']) for r in rows])
        X = feature_matrix(rows)
        from compare_models import read
        predictions = read('weighted_stopping_mlp.csv')
        self.assertEqual({int(r['pass_index']) for r in predictions}, set(range(1, 11)))
        needed = set(ids)
        points = {r['timestamp_unix_ms']: r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in needed}
        roads = {f['properties']['segment_id']: f['geometry']['coordinates'] for f in json.loads((OUT / 'road_segments.geojson').read_text())['features']}
        geo = geography(rows, labels, ids, roads, points)
        old = {(int(r['seed']), r['timestamp_unix_ms']): r for r in read('weighted_group_mlp_predictions.csv')}
        for run in report['runs']:
            saved = {r['timestamp_unix_ms']: r for r in predictions if int(r['seed']) == run['seed']}
            self.assertEqual(set(saved), set(ids))
            p = np.array([json.loads(saved[t]['probabilities']) for t in ids])
            np.testing.assert_allclose(p.sum(axis=1), 1)
            self.assertTrue(np.isfinite(p).all() and (p >= 0).all())
            metrics, distances = evaluate(y, p, labels, geo)
            for key in experiment.KEYS:
                self.assertAlmostEqual(metrics[key], run['metrics'][key])
            np.testing.assert_allclose(distances, [float(saved[t]['distance_m']) for t in ids])
            pred = labels[p.argmax(axis=1)]
            np.testing.assert_allclose(recall_score(y, pred, labels=labels, average=None, zero_division=0), run['metrics']['per_segment_recall'])
            prior = np.array([json.loads(old[run['seed'], t]['probabilities']) for t in ids])
            for pair in (v for v in report['paired'] if v['seed'] == run['seed']):
                m = np.ones(len(y), bool) if pair['fold'] == 0 else folds == pair['fold']
                baseline = evaluate(y[m], prior[m], labels, geo[m])[0]
                candidate = evaluate(y[m], p[m], labels, geo[m])[0]
                for key in experiment.KEYS:
                    self.assertAlmostEqual(candidate[key] - baseline[key], pair['delta'][key])
                a = labels[prior[m].argmax(axis=1)] == y[m]; b = pred[m] == y[m]
                self.assertEqual(int((~a & b).sum()), pair['candidate_only_correct'])
                self.assertEqual(int((a & ~b).sum()), pair['baseline_only_correct'])
            for d in run['stopping']:
                fit = (folds != d['fold']) & (folds != d['inner_fold']); stop = folds == d['inner_fold']
                masks = dict(inner_fit=fit, inner_stop=stop, outer_fit=folds != d['fold'], outer_held=folds == d['fold'])
                for k, m in masks.items():
                    self.assertEqual(experiment.counts(y[m], labels), d['class_support'][k])
                    self.assertEqual(sorted(set(passes[m].tolist())), d['pass_groups'][k])
                for a, b in [('inner_fit', 'inner_stop'), ('inner_fit', 'outer_held'), ('inner_stop', 'outer_held')]:
                    self.assertFalse(set(d['pass_groups'][a]) & set(d['pass_groups'][b]))
                for m, meta in [(fit, d), (masks['outer_fit'], d['outer_preprocessing_and_weights'])]:
                    _, n, cw = fitting_weights(y[m], labels)
                    self.assertEqual(n.tolist(), meta['counts'])
                    np.testing.assert_allclose(cw, meta['class_weights'])
                    med = np.nanmedian(X[m], axis=0)
                    np.testing.assert_allclose(med, meta['median'])
                    np.testing.assert_allclose(np.mean(np.where(np.isnan(X[m]), med, X[m]), axis=0), meta['scale_mean'])
                self.assertEqual(choose_epoch(d['weighted_log_loss']), (d['selected_epoch'], d['stopped_epoch']))
                self.assertGreaterEqual(min(d['class_support']['inner_fit']), 1)


if __name__ == '__main__':
    unittest.main()
