"""Named training-only checks for macro-recall-focused class weighting."""
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import recall_score

import compare_window_weights as experiment
from compare_models import OUT, SEEDS, read
from compare_window_models import NEW
from group_stopped_mlp import choose_epoch
from moderate_forest_weights import moderate_weights
from optimize_models import feature_matrix
from segment_road import local_xy, point_to_line
from sqrt_group_mlp import fitting_weights
from test_window_feature_dependency import classification


class WindowWeightTests(unittest.TestCase):
    def test_shared_moderate_weight_rule_and_grouped_mlp_calls(self):
        labels = np.array(['a', 'b']); y = np.array(['a']*9+['b'])
        w, counts, cw = fitting_weights(y, labels); forest, fc = moderate_weights(y, labels)
        np.testing.assert_array_equal(counts, [9, 1]); np.testing.assert_array_equal(fc, counts)
        self.assertAlmostEqual(cw[1]/cw[0], 3)
        self.assertAlmostEqual(w.mean(), 1)
        np.testing.assert_allclose([forest[v] for v in labels], cw)
        X = np.arange(100, dtype=float).reshape(10, 10); y = np.tile(labels, 5)
        folds = np.repeat(np.arange(1, 6), 2); passes = folds*2-1
        class Pipe:
            classes_ = labels
            n_features_in_ = 10
        with patch.object(experiment, 'train_weighted', side_effect=[(Pipe(), {}, 3, [2., 1.5, 1.]), (Pipe(), {}, 0, [])]) as train:
            _, d = experiment.fit_weighted_one('MLP', 17, X, y, labels, folds, passes, 1)
        calls = train.call_args_list
        np.testing.assert_array_equal(calls[0].args[2], (folds != 1) & (folds != 2))
        np.testing.assert_array_equal(calls[0].kwargs['held'], folds == 2)
        np.testing.assert_array_equal(calls[1].args[2], folds != 1)
        self.assertEqual(calls[1].kwargs['epochs'], 3)
        self.assertFalse(set(d['inner_fit_passes']) & set(d['inner_stop_passes']))
        self.assertFalse(set(d['inner_stop_passes']) & set(d['outer_held_passes']))

    def test_saved_weights_preprocessing_metrics_and_pairing(self):
        r = json.loads((OUT/'window_moderate_weights.json').read_text()); rows = read('training_windows_v1.csv')
        ids = [v['timestamp_unix_ms'] for v in rows]; needed = set(ids)
        self.assertEqual(len(rows), 1393); self.assertEqual({v['split'] for v in rows}, {'train'})
        self.assertEqual({int(v['pass_index']) for v in rows}, set(range(1, 11)))
        self.assertEqual(r['primary_objective'], 'macro_recall')
        self.assertEqual(hashlib.sha256((Path(__file__).parent/'compare_window_weights.py').read_bytes()).hexdigest(), r['code_sha256'])
        for f, digest in r['dependency_sha256'].items():
            self.assertEqual(hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest(), digest)
        new = read('window_moderate_weights_predictions.csv'); old = read('window_model_predictions.csv')
        self.assertEqual(len(new), 2*len(SEEDS)*len(rows))
        X = feature_matrix(rows); labels = np.array(r['labels']); y = np.array([v['segment_id'] for v in rows]); folds = np.array([int(v['cv_fold']) for v in rows])
        probabilities = {}; distances = {}
        points = {v['timestamp_unix_ms']:v for v in read('location_points_segmented.csv') if v['timestamp_unix_ms'] in needed}
        roads = {v['properties']['segment_id']:v['geometry']['coordinates'] for v in json.loads((OUT/'road_segments.geojson').read_text())['features']}
        lon0 = np.mean([float(points[t]['longitude']) for t in ids]); lat0 = np.mean([float(points[t]['latitude']) for t in ids])
        for run in r['runs']:
            kind, seed = run['kind'], run['seed']; weighted = run['model'] == experiment.NAMES[kind]
            source = new if weighted else old; name = run['model'] if weighted else NEW[kind]
            ss = {v['timestamp_unix_ms']:v for v in source if v['model'] == name and int(v['seed']) == seed}
            self.assertEqual(set(ss), needed)
            p = np.array([json.loads(ss[t]['probabilities']) for t in ids]); pred = labels[p.argmax(axis=1)]
            d = np.array([float(ss[t]['distance_m']) for t in ids]); probabilities[kind, seed, weighted] = p; distances[kind, seed, weighted] = d
            self.assertTrue(np.isfinite(p).all() and (p >= 0).all()); np.testing.assert_allclose(p.sum(axis=1), 1)
            for k, value in classification(y, p, labels).items(): self.assertAlmostEqual(value, run['metrics'][k])
            np.testing.assert_allclose(recall_score(y, pred, labels=labels, average=None, zero_division=0), run['metrics']['per_segment_recall'])
            for values, prefix in [(d, 'all'), (d[pred != y], 'wrong')]:
                self.assertAlmostEqual(np.median(values), run['metrics'][prefix+'_distance_median'])
                self.assertAlmostEqual(np.quantile(values, .9, method='inverted_cdf'), run['metrics'][prefix+'_distance_p90'])
            for i in [0, 100, 500, 1000, 1392]:
                q = points[ids[i]]; xy = local_xy(float(q['longitude']), float(q['latitude']), lon0, lat0)
                self.assertAlmostEqual(point_to_line(xy, roads[pred[i]], lon0, lat0)[0], d[i], places=6)
            for detail in run['training']:
                outer = detail['outer_fold']; fit = folds != outer
                self.assertFalse(set(detail['outer_fit_passes']) & set(detail['outer_held_passes']))
                stages = [(fit, detail['outer_preprocessing_and_weights'])]
                if kind == 'MLP':
                    inner = detail['inner_fold']; ii = fit & (folds != inner); stop = folds == inner
                    stages.append((ii, detail['inner_preprocessing_and_weights']))
                    self.assertFalse(set(detail['inner_fit_passes']) & set(detail['inner_stop_passes']))
                    self.assertFalse(set(detail['inner_stop_passes']) & set(detail['outer_held_passes']))
                    self.assertEqual(choose_epoch(detail['inner_log_loss']), (detail['selected_epoch'], detail['stopped_epoch']))
                    self.assertEqual([int((y[stop] == v).sum()) for v in labels], detail['inner_stop_support'])
                for mask, meta in stages:
                    counts = np.array([(y[mask] == v).sum() for v in labels])
                    cw = 1/np.sqrt(counts); cw /= np.average(cw, weights=counts)
                    self.assertEqual(counts.tolist(), meta['counts'])
                    np.testing.assert_allclose(cw, meta['weights'] if kind == 'RF' else meta['class_weights'])
                    med = np.nanmedian(X[mask], axis=0); np.testing.assert_allclose(med, meta['median'])
                    if kind == 'MLP': np.testing.assert_allclose(np.where(np.isnan(X[mask]), med, X[mask]).mean(axis=0), meta['scale_mean'])
        self.assertEqual(len(r['paired']), 36)
        for pair in r['paired']:
            kind, seed = pair['kind'], pair['seed']; a = probabilities[kind, seed, False]; b = probabilities[kind, seed, True]
            ix = np.ones(len(y), bool) if pair['fold'] == 0 else folds == pair['fold']
            ac = labels[a[ix].argmax(axis=1)] == y[ix]; bc = labels[b[ix].argmax(axis=1)] == y[ix]
            self.assertEqual(int((ac & ~bc).sum()), pair['unweighted_only_correct'])
            self.assertEqual(int((~ac & bc).sum()), pair['weighted_only_correct'])
            am, bm = classification(y[ix], a[ix], labels), classification(y[ix], b[ix], labels)
            for k in am: self.assertAlmostEqual(bm[k]-am[k], pair['delta'][k])


if __name__ == '__main__':
    unittest.main()
