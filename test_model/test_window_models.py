"""Named training-only checks for the fixed window-model comparison."""
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

import compare_window_models as experiment
from compare_models import FEATURES, OUT, SEEDS, read
from group_stopped_mlp import choose_epoch
from optimize_models import feature_matrix
from segment_road import local_xy, point_to_line


class WindowModelTests(unittest.TestCase):
    def test_mlp_calls_use_only_inner_fit_stop_and_outer_fit(self):
        folds = np.repeat(np.arange(1, 6), 2); passes = folds*2-1
        X = np.arange(20, dtype=float).reshape(10, 2)
        y = np.tile(['a', 'b'], 5); labels = np.array(['a', 'b'])
        fit = folds != 1; inner_fit = (folds != 1) & (folds != 2); stop = folds == 2
        class Part:
            statistics_ = np.median(X[fit], axis=0)
            mean_ = np.mean(X[fit], axis=0)
            scale_ = np.ones(2)
        class Pipe:
            classes_ = labels
            def __getitem__(self, index): return Part()
        with patch.object(experiment, 'fit_inner', return_value=(3, [2., 1.5, 1.])) as inner, patch.object(experiment, 'refit_outer', return_value=Pipe()) as refit:
            _, detail = experiment.fit_one('MLP', 17, X, y, labels, folds, passes, 1)
        np.testing.assert_array_equal(inner.call_args.args[2], inner_fit)
        np.testing.assert_array_equal(inner.call_args.args[3], stop)
        np.testing.assert_array_equal(refit.call_args.args[2], fit)
        self.assertEqual(refit.call_args.args[-1], 3)
        self.assertFalse(set(detail['inner_fit_passes']) & set(detail['outer_held_passes']))
        self.assertFalse(set(detail['inner_stop_passes']) & set(detail['outer_held_passes']))

    def test_saved_metrics_pairing_stopping_and_geography(self):
        r = json.loads((OUT/'window_model_comparison.json').read_text())
        rows = read('training_windows_v1.csv'); ids = [v['timestamp_unix_ms'] for v in rows]
        self.assertEqual(len(rows), 1393)
        self.assertEqual({v['split'] for v in rows}, {'train'})
        self.assertEqual({int(v['pass_index']) for v in rows}, set(range(1, 11)))
        self.assertEqual(r['features'], FEATURES)
        self.assertEqual(hashlib.sha256((OUT/'training_windows_v1.csv').read_bytes()).hexdigest(), r['window_csv_sha256'])
        self.assertEqual(hashlib.sha256((Path(__file__).parent/'compare_window_models.py').read_bytes()).hexdigest(), r['code_sha256'])
        labels = np.array(r['labels']); y = np.array([v['segment_id'] for v in rows]); X = feature_matrix(rows)
        folds = np.array([int(v['cv_fold']) for v in rows]); needed = set(ids)
        new = read('window_model_predictions.csv')
        self.assertEqual(len(new), 2*len(SEEDS)*len(rows))
        sources = {'Forest window fit': new, 'MLP window fit': new,
                   'Forest published-row fit': [v for v in read('forest_leaf_scores_predictions.csv') if int(v['leaf']) == 2],
                   'MLP published-row fit': read('group_stopped_mlp_predictions.csv')}
        probabilities = {}
        points = {v['timestamp_unix_ms']: v for v in read('location_points_segmented.csv') if v['timestamp_unix_ms'] in needed}
        roads = {v['properties']['segment_id']: v['geometry']['coordinates'] for v in json.loads((OUT/'road_segments.geojson').read_text())['features']}
        lon0 = np.mean([float(points[t]['longitude']) for t in ids]); lat0 = np.mean([float(points[t]['latitude']) for t in ids])
        for run in r['runs']:
            name, seed = run['model'], run['seed']
            ss = {v['timestamp_unix_ms']: v for v in sources[name] if int(v['seed']) == seed and v['timestamp_unix_ms'] in needed
                  and ('model' not in v or v['model'] == name)}
            self.assertEqual(set(ss), needed)
            p = np.array([json.loads(ss[t]['probabilities']) for t in ids]); probabilities[name, seed] = p
            self.assertTrue(np.isfinite(p).all() and (p >= 0).all()); np.testing.assert_allclose(p.sum(axis=1), 1)
            pred = labels[p.argmax(axis=1)]; m = run['metrics']
            self.assertAlmostEqual(accuracy_score(y, pred), m['accuracy'])
            self.assertAlmostEqual(recall_score(y, pred, labels=labels, average='macro', zero_division=0), m['macro_recall'])
            self.assertAlmostEqual(f1_score(y, pred, labels=labels, average='macro', zero_division=0), m['macro_f1'])
            self.assertAlmostEqual(log_loss(y, p, labels=labels), m['log_loss'])
            self.assertAlmostEqual(np.mean(np.sum((p-(y[:, None] == labels))**2, axis=1)), m['multiclass_brier'])
            np.testing.assert_allclose(recall_score(y, pred, labels=labels, average=None, zero_division=0), m['per_segment_recall'])
            top = np.argsort(p, axis=1)[:, -3:]
            self.assertAlmostEqual(np.mean((top == np.searchsorted(labels, y)[:, None]).any(axis=1)), m['top3_accuracy'])
            for fold in run['folds']:
                ix = folds == fold['fold']; self.assertEqual(int(ix.sum()), fold['support'])
                self.assertAlmostEqual(accuracy_score(y[ix], pred[ix]), fold['metrics']['accuracy'])
            for detail in run['training']:
                f = detail['outer_fold']; fit = folds != f
                np.testing.assert_allclose(np.nanmedian(X[fit], axis=0), detail['outer_median'])
                self.assertEqual(int(fit.sum()), detail['outer_fit_rows'])
                self.assertFalse(set(detail['outer_fit_passes']) & set(detail['outer_held_passes']))
                if 'inner_fold' in detail:
                    inner = detail['inner_fold']; ii = fit & (folds != inner); stop = folds == inner
                    self.assertEqual(choose_epoch(detail['inner_log_loss']), (detail['selected_epoch'], detail['stopped_epoch']))
                    self.assertFalse(set(detail['inner_fit_passes']) & set(detail['inner_stop_passes']))
                    self.assertFalse(set(detail['inner_stop_passes']) & set(detail['outer_held_passes']))
                    med = np.nanmedian(X[ii], axis=0); filled = np.where(np.isnan(X[ii]), med, X[ii])
                    np.testing.assert_allclose(med, detail['inner_expected_median'])
                    np.testing.assert_allclose(filled.mean(axis=0), detail['inner_expected_scale_mean'])
                    med = np.nanmedian(X[fit], axis=0); filled = np.where(np.isnan(X[fit]), med, X[fit])
                    np.testing.assert_allclose(filled.mean(axis=0), detail['outer_scale_mean'])
                    scale = filled.std(axis=0); scale[scale == 0] = 1
                    np.testing.assert_allclose(scale, detail['outer_scale'])
                    self.assertEqual([int((y[stop] == v).sum()) for v in labels], detail['inner_stop_support'])
            if name in ['Forest window fit', 'MLP window fit']:
                distances = np.array([float(ss[t]['distance_m']) for t in ids]); wrong = pred != y
                for values, prefix in [(distances, 'all'), (distances[wrong], 'wrong')]:
                    self.assertAlmostEqual(np.median(values), m[prefix+'_distance_median'])
                    self.assertAlmostEqual(np.quantile(values, .9, method='inverted_cdf'), m[prefix+'_distance_p90'])
                for i in [0, 100, 500, 1000, 1392]:
                    q = points[ids[i]]; xy = local_xy(float(q['longitude']), float(q['latitude']), lon0, lat0)
                    self.assertAlmostEqual(point_to_line(xy, roads[pred[i]], lon0, lat0)[0], distances[i], places=6)
        for pair in r['paired']:
            a = probabilities[pair['reference'], pair['seed']]; b = probabilities[pair['candidate'], pair['seed']]
            ix = np.ones(len(y), bool) if pair['fold'] == 0 else folds == pair['fold']
            ac = labels[a[ix].argmax(axis=1)] == y[ix]; bc = labels[b[ix].argmax(axis=1)] == y[ix]
            self.assertEqual(int((ac & ~bc).sum()), pair['reference_only_correct'])
            self.assertEqual(int((~ac & bc).sum()), pair['candidate_only_correct'])
            self.assertAlmostEqual(float(bc.mean()-ac.mean()), pair['delta']['accuracy'])
            delta = recall_score(y[ix], labels[b[ix].argmax(axis=1)], labels=labels, average='macro', zero_division=0)-recall_score(y[ix], labels[a[ix].argmax(axis=1)], labels=labels, average='macro', zero_division=0)
            self.assertAlmostEqual(delta, pair['delta']['macro_recall'])


if __name__ == '__main__':
    unittest.main()
