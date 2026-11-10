"""Explicit development-only checks; no historical validation/test scoring."""
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, recall_score
from threadpoolctl import threadpool_limits

from direct_weight_edits import (MAX_LOGIT_RMS, NAMES, OUT, STEPS, apply_delta,
                                 digest, edit_head, hidden_design, objective, read)
from optimize_models import feature_matrix
from segment_road import local_xy, point_to_line


class DirectEditTests(unittest.TestCase):
    def test_objective_is_macro_recall_and_balanced_loss(self):
        logits = np.array([[4., 0.], [3., 0.], [2., 0.], [1., 0.]])
        y = np.array([0, 0, 0, 1])
        score = objective(logits, y)
        self.assertEqual(score[0], .5)
        self.assertAlmostEqual(-score[1], log_loss(y, softmax(logits, axis=1), sample_weight=[1/3, 1/3, 1/3, 1]))

    def test_search_is_deterministic_bounded_and_retains_no_edit(self):
        h = np.column_stack([np.linspace(-1, 1, 12), np.ones(12)])
        y = np.array([0] * 6 + [1] * 6)
        head = np.zeros((2, 2))
        for bias_only in [True, False]:
            delta, trace = edit_head(h, y, head, 17, bias_only)
            again, same = edit_head(h, y, head, 17, bias_only)
            np.testing.assert_array_equal(delta, again); self.assertEqual(trace, same)
            self.assertEqual(len(trace), STEPS + 1)
            scores = [(t['macro_recall'], -t['balanced_log_loss']) for t in trace]
            self.assertEqual(scores, sorted(scores))
            self.assertLessEqual(np.sqrt(np.mean((h @ delta)**2)), MAX_LOGIT_RMS + 1e-12)
            np.testing.assert_allclose(delta.sum(axis=1), 0, atol=1e-12)
            if bias_only: np.testing.assert_array_equal(delta[:-1], 0)

    def test_saved_models_predictions_metrics_and_group_separation(self):
        report = json.loads((OUT / 'direct_weight_edits.json').read_text())
        self.assertEqual(digest(OUT / 'direct_weight_edits.joblib'), report['model_bundle_sha256'])
        self.assertEqual(digest(OUT.parent / 'direct_weight_edits.py'), report['code_sha256'])
        for name, value in report['source_sha256'].items(): self.assertEqual(digest(OUT / name), value)
        for name, value in report['dependency_sha256'].items(): self.assertEqual(digest(OUT.parent / name), value)
        bundle = joblib.load(OUT / 'direct_weight_edits.joblib')
        rows = read('training_windows_v1.csv'); X = feature_matrix(rows)
        y = np.array([r['segment_id'] for r in rows]); labels = np.array(report['labels'])
        folds = np.array([int(r['cv_fold']) for r in rows]); passes = np.array([int(r['pass_index']) for r in rows])
        saved = {(r['model'], int(r['seed']), r['timestamp_unix_ms']): r for r in read('direct_weight_edits_predictions.csv')}
        self.assertEqual(len(saved), len(rows) * 3 * 3)
        reconstructed = {}
        for seed in report['seeds']:
            run = next(r for r in report['runs'] if r['model'] == NAMES[0] and r['seed'] == seed)
            for d in run['training']:
                outer = d['outer_fold']; inner = d['inner_fold']
                fit = (folds != outer) & (folds != inner); adapt = folds == inner; held = folds == outer
                self.assertEqual(inner, outer % 5 + 1)
                for a, b in [(fit, adapt), (fit, held), (adapt, held)]: self.assertFalse(set(passes[a]) & set(passes[b]))
                for k, m in [('fit', fit), ('adapt', adapt), ('held', held)]:
                    self.assertEqual(d['pass_groups'][k], sorted(set(passes[m].tolist())))
                    self.assertEqual(d['support'][k], [(y[m] == v).sum() for v in labels])
                base = bundle['models'][seed, outer, NAMES[0]]
                np.testing.assert_allclose(base[0].statistics_, np.nanmedian(X[fit], axis=0))
                np.testing.assert_allclose(base[1].mean_, base[0].transform(X[fit]).mean(axis=0))
                h = hidden_design(base, X[adapt]); head = np.vstack([base[-1].coefs_[-1], base[-1].intercepts_[-1]])
                for name in NAMES:
                    pipe = bundle['models'][seed, outer, name]
                    if name != NAMES[0]:
                        edit = d['edits'][name]; delta = np.array(edit['delta'])
                        for a, b in zip(base[-1].coefs_[:-1], pipe[-1].coefs_[:-1]): np.testing.assert_array_equal(a, b)
                        for a, b in zip(base[-1].intercepts_[:-1], pipe[-1].intercepts_[:-1]): np.testing.assert_array_equal(a, b)
                        np.testing.assert_allclose(pipe[-1].coefs_[-1] - base[-1].coefs_[-1], delta[:-1], atol=1e-12)
                        np.testing.assert_allclose(pipe[-1].intercepts_[-1] - base[-1].intercepts_[-1], delta[-1], atol=1e-12)
                        self.assertLessEqual(np.sqrt(np.mean((h @ delta)**2)), MAX_LOGIT_RMS + 1e-12)
                        self.assertEqual(edit['changed_parameters'], np.count_nonzero(delta))
                        # Fail if adaptation ever tries to use training/optimizer methods.
                        with patch.object(type(base[-1]), 'partial_fit', side_effect=AssertionError('gradient update')), patch.object(type(base[-1]), 'fit', side_effect=AssertionError('fit')):
                            replay, trace = edit_head(h, np.searchsorted(labels, y[adapt]), head, seed * 100 + outer, name == NAMES[1])
                            edited = apply_delta(base, replay)
                        np.testing.assert_array_equal(delta, replay); self.assertEqual(edit['trace'], trace)
                        np.testing.assert_allclose(edited.predict_proba(X[held]), pipe.predict_proba(X[held]))
                        if name == NAMES[1]: np.testing.assert_array_equal(delta[:-1], 0)
                    p = pipe.predict_proba(X[held])
                    for i, pp in zip(np.flatnonzero(held), p):
                        old = saved[name, seed, rows[i]['timestamp_unix_ms']]
                        self.assertEqual(old['segment_id'], y[i]); self.assertEqual(int(old['cv_fold']), outer)
                        np.testing.assert_allclose(json.loads(old['probabilities']), pp)
        for run in report['runs']:
            name = run['model']; seed = run['seed']
            p = np.array([json.loads(saved[name, seed, r['timestamp_unix_ms']]['probabilities']) for r in rows])
            reconstructed[name, seed] = p
            for mask, metrics in [(np.ones(len(y), bool), run['metrics'])] + [(folds == f['fold'], f['metrics']) for f in run['folds']]:
                yy = y[mask]; pp = p[mask]; pred = labels[pp.argmax(axis=1)]
                self.assertAlmostEqual(metrics['accuracy'], accuracy_score(yy, pred))
                self.assertAlmostEqual(metrics['macro_recall'], recall_score(yy, pred, labels=labels, average='macro', zero_division=0))
                self.assertAlmostEqual(metrics['macro_f1'], f1_score(yy, pred, labels=labels, average='macro', zero_division=0))
                self.assertAlmostEqual(metrics['log_loss'], log_loss(yy, pp, labels=labels))
                np.testing.assert_allclose(metrics['per_segment_recall'], recall_score(yy, pred, labels=labels, average=None, zero_division=0))
                expected = np.eye(len(labels))[np.searchsorted(labels, yy)]
                self.assertAlmostEqual(metrics['multiclass_brier'], np.mean(np.sum((pp - expected)**2, axis=1)))
                self.assertAlmostEqual(metrics['top3_accuracy'], np.mean(np.any(np.argsort(pp, axis=1)[:, -3:] == np.searchsorted(labels, yy)[:, None], axis=1)))
                np.testing.assert_array_equal(metrics['confusion_matrix'], confusion_matrix(yy, pred, labels=labels))
                distances = np.array([float(saved[name, seed, r['timestamp_unix_ms']]['distance_m']) for r in rows])[mask]
                self.assertAlmostEqual(metrics['all_distance_median'], np.median(distances))
                self.assertAlmostEqual(metrics['all_distance_p90'], np.quantile(distances, .9, method='inverted_cdf'))
                self.assertAlmostEqual(metrics['wrong_distance_median'], np.median(distances[pred != yy]))
                self.assertAlmostEqual(metrics['wrong_distance_p90'], np.quantile(distances[pred != yy], .9, method='inverted_cdf'))
        # Recompute sampled distances from raw coordinates and the predicted road.
        wanted = {r['timestamp_unix_ms'] for r in rows}
        points = {r['timestamp_unix_ms']: r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in wanted}
        lon0 = np.mean([float(r['longitude']) for r in points.values()]); lat0 = np.mean([float(r['latitude']) for r in points.values()])
        roads = {f['properties']['segment_id']: f['geometry']['coordinates'] for f in json.loads((OUT / 'road_segments.geojson').read_text())['features']}
        for name in NAMES:
            for r in rows[::97]:
                p = points[r['timestamp_unix_ms']]; old = saved[name, 17, r['timestamp_unix_ms']]
                xy = local_xy(float(p['longitude']), float(p['latitude']), lon0, lat0)
                self.assertAlmostEqual(float(old['distance_m']), point_to_line(xy, roads[old['predicted_segment_id']], lon0, lat0)[0])
        for pair in report['paired']:
            mask = np.ones(len(y), bool) if pair['fold'] == 0 else folds == pair['fold']
            a = reconstructed[NAMES[0], pair['seed']][mask].argmax(axis=1)
            b = reconstructed[pair['model'], pair['seed']][mask].argmax(axis=1)
            yy = np.searchsorted(labels, y[mask])
            self.assertEqual(pair['baseline_only_correct'], int(((a == yy) & (b != yy)).sum()))
            self.assertEqual(pair['edited_only_correct'], int(((a != yy) & (b == yy)).sum()))
            self.assertAlmostEqual(pair['delta']['macro_recall'], recall_score(yy, b, labels=np.arange(len(labels)), average='macro') - recall_score(yy, a, labels=np.arange(len(labels)), average='macro'))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main()
