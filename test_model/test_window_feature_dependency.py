"""Explicit development-only checks for the bundled feature-removal experiment."""
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

import check_window_feature_dependency as experiment
from compare_models import FEATURES, OUT, SEEDS, read
from compare_window_models import NEW
from group_stopped_mlp import choose_epoch
from segment_road import local_xy, point_to_line


def classification(y, p, labels):
    pred = labels[p.argmax(axis=1)]
    top = np.argsort(p, axis=1)[:, -3:]
    return dict(accuracy=accuracy_score(y, pred), macro_recall=recall_score(y, pred, labels=labels, average='macro', zero_division=0),
                macro_f1=f1_score(y, pred, labels=labels, average='macro', zero_division=0), log_loss=log_loss(y, p, labels=labels),
                top3_accuracy=np.mean((top == np.searchsorted(labels, y)[:, None]).any(axis=1)),
                multiclass_brier=np.mean(np.sum((p-(y[:, None] == labels))**2, axis=1)))


class FeatureDependencyTests(unittest.TestCase):
    def test_input_allowlist_ignores_removed_fields_and_metadata(self):
        row = {f:str(i+1) for i, f in enumerate(experiment.INPUTS)}
        row.update(message_freshness_ms='must not parse', bsm_rate_per_sec='must not parse',
                   segment_id='label', latitude='must not parse', timestamp_unix_ms='must not parse',
                   analyzer_window_id='id', summary_to_ue_last_seen_ms='must not parse')
        np.testing.assert_array_equal(experiment.input_matrix([row]), np.arange(1, 9)[None, :])
        self.assertEqual(experiment.INPUTS, [f for f in FEATURES if f not in experiment.REMOVED])
        row[experiment.INPUTS[0]] = ''
        self.assertTrue(np.isnan(experiment.input_matrix([row])[0, 0]))

    def test_saved_metrics_groups_pairing_and_geography(self):
        r = json.loads((OUT/'window_feature_dependency.json').read_text())
        rows = read('training_windows_v1.csv'); ids = [v['timestamp_unix_ms'] for v in rows]; wanted = set(ids)
        self.assertEqual(len(rows), 1393)
        self.assertEqual({v['split'] for v in rows}, {'train'})
        self.assertEqual({int(v['pass_index']) for v in rows}, set(range(1, 11)))
        self.assertEqual(r['features'], experiment.INPUTS); self.assertEqual(r['reference_features'], FEATURES)
        self.assertEqual(r['forest_candidate_counts'], {'ten_inputs':3, 'eight_inputs':2})
        self.assertEqual(hashlib.sha256((Path(__file__).parent/'check_window_feature_dependency.py').read_bytes()).hexdigest(), r['code_sha256'])
        for f, digest in r['dependency_sha256'].items():
            self.assertEqual(hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest(), digest)
        new = read('window_feature_dependency_predictions.csv'); old = read('window_model_predictions.csv')
        self.assertEqual(len(new), 2*len(SEEDS)*len(rows))
        labels = np.array(r['labels']); y = np.array([v['segment_id'] for v in rows]); X = experiment.input_matrix(rows)
        folds = np.array([int(v['cv_fold']) for v in rows]); all_prob = {}; all_distance = {}
        points = {v['timestamp_unix_ms']:v for v in read('location_points_segmented.csv') if v['timestamp_unix_ms'] in wanted}
        roads = {v['properties']['segment_id']:v['geometry']['coordinates'] for v in json.loads((OUT/'road_segments.geojson').read_text())['features']}
        lon0 = np.mean([float(points[t]['longitude']) for t in ids]); lat0 = np.mean([float(points[t]['latitude']) for t in ids])
        for run in r['runs']:
            kind, seed = run['kind'], run['seed']; is_new = run['model'] == experiment.NAMES[kind]
            source = new if is_new else old; name = run['model'] if is_new else NEW[kind]
            ss = {v['timestamp_unix_ms']:v for v in source if v['model'] == name and int(v['seed']) == seed}
            self.assertEqual(set(ss), wanted)
            p = np.array([json.loads(ss[t]['probabilities']) for t in ids]); pred = labels[p.argmax(axis=1)]
            d = np.array([float(ss[t]['distance_m']) for t in ids])
            all_prob[kind, seed, is_new] = p; all_distance[kind, seed, is_new] = d
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
                self.assertEqual(len(detail['outer_median']), 8)
                np.testing.assert_allclose(np.nanmedian(X[fit], axis=0), detail['outer_median'])
                self.assertFalse(set(detail['outer_fit_passes']) & set(detail['outer_held_passes']))
                self.assertEqual([int((y[fit] == v).sum()) for v in labels], detail['outer_fit_support'])
                if 'inner_fold' in detail:
                    inner = detail['inner_fold']; ii = fit & (folds != inner); stop = folds == inner
                    self.assertEqual(choose_epoch(detail['inner_log_loss']), (detail['selected_epoch'], detail['stopped_epoch']))
                    self.assertFalse(set(detail['inner_fit_passes']) & set(detail['inner_stop_passes']))
                    self.assertFalse(set(detail['inner_stop_passes']) & set(detail['outer_held_passes']))
                    med = np.nanmedian(X[ii], axis=0)
                    np.testing.assert_allclose(med, detail['inner_expected_median'])
                    np.testing.assert_allclose(np.where(np.isnan(X[ii]), med, X[ii]).mean(axis=0), detail['inner_expected_scale_mean'])
                    med = np.nanmedian(X[fit], axis=0); filled = np.where(np.isnan(X[fit]), med, X[fit])
                    np.testing.assert_allclose(filled.mean(axis=0), detail['outer_scale_mean'])
                    scale = filled.std(axis=0); scale[scale == 0] = 1
                    np.testing.assert_allclose(scale, detail['outer_scale'])
                    self.assertEqual([int((y[stop] == v).sum()) for v in labels], detail['inner_stop_support'])
        self.assertEqual(len(r['paired']), 36)
        for pair in r['paired']:
            kind, seed = pair['kind'], pair['seed']; a = all_prob[kind, seed, False]; b = all_prob[kind, seed, True]
            ix = np.ones(len(y), bool) if pair['fold'] == 0 else folds == pair['fold']
            ac = labels[a[ix].argmax(axis=1)] == y[ix]; bc = labels[b[ix].argmax(axis=1)] == y[ix]
            self.assertEqual(int((ac & ~bc).sum()), pair['ten_only_correct'])
            self.assertEqual(int((~ac & bc).sum()), pair['eight_only_correct'])
            am, bm = classification(y[ix], a[ix], labels), classification(y[ix], b[ix], labels)
            for k in am: self.assertAlmostEqual(bm[k]-am[k], pair['delta'][k])


if __name__ == '__main__':
    unittest.main()
