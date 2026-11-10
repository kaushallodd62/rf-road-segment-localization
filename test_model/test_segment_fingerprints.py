"""Targeted development-only checks; never invoke a historical test evaluator."""
import csv
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import audit_segment_fingerprints as audit


class FingerprintTests(unittest.TestCase):
    def test_fitting_only_preprocessing_and_missing_feature_rejection(self):
        x = np.array([[1., np.nan], [3., 4.], [10000., -1000.]])
        _, held, detail = audit.transform(x, np.array([True, True, False]), np.array([False, False, True]))
        self.assertEqual(detail['medians'], [2., 4.])
        self.assertEqual(detail['mean'], [2., 4.])
        np.testing.assert_allclose(held, [[9998., -1004.]])
        with self.assertRaises(ValueError):
            audit.transform(np.array([[np.nan], [2.]]), np.array([True, False]), np.array([False, True]))

    def test_neighbors_ties_and_unsupported_labels(self):
        ds, do, valid, other = audit.nearest_distances(np.array([[0.], [2.]]), np.array([[1.], [0.], [1.]]),
                                                     np.array(['a', 'b']), np.array(['a', 'b', 'c']))
        np.testing.assert_array_equal(valid, [True, True, False])
        self.assertEqual(ds[0], do[0])
        self.assertLess(do[1], ds[1])
        self.assertEqual(other[1], 'a')

    def test_stable_separation_vs_common_pass_drift(self):
        rows = []; labels = []; values = []; along = []
        for p in range(1, 11):
            for s in range(2):
                for j in range(5):
                    rows.append(dict(pass_index=str(p), direction='N' if p % 2 else 'S'))
                    labels.append(str(s)); values.append([10*s + .01*p + .001*j]); along.append(5+j)
        y = np.array(labels); x = np.array(values); along = np.array(along)
        stable = audit.signature_diagnostics(rows, x, y, along, ['value'])
        self.assertTrue(all(r['separation_to_drift'] > 1 for r in stable['contrasts']))
        shifted = x + np.array([int(r['pass_index'])*100 for r in rows])[:, None]
        drift = audit.signature_diagnostics(rows, shifted, y, along, ['value'])
        self.assertTrue(all(r['separation_to_drift'] < 1 for r in drift['contrasts']))
        self.assertEqual(len(stable['matched_position']), 40)
        self.assertTrue(all(min(r['support']) == 5 for r in stable['matched_position']))

    def test_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/audit.OUTPUTS[-1]; path.write_text('preserve')
            with patch.object(audit, 'OUT', Path(temp)), self.assertRaises(RuntimeError): audit.run()
            self.assertEqual(path.read_text(), 'preserve')

    def test_saved_metrics_provenance_and_held_prediction_replay(self):
        report = json.loads((audit.OUT/audit.OUTPUTS[0]).read_text())
        self.assertEqual(report['code_sha256'], audit.digest(Path(audit.__file__)))
        for name, expected in report['source_sha256'].items():
            self.assertEqual(audit.digest(audit.OUT/name), expected)
        with (audit.OUT/audit.OUTPUTS[1]).open(newline='') as f:
            neighbors = list(csv.DictReader(f))
        for result in report['neighbors']:
            records = [r for r in neighbors if (r['view'], r['scope']) == (result['view'], result['scope'])]
            valid = [r for r in records if int(r['eligible'])]
            self.assertEqual(len(valid), result['eligible'])
            wins = [float(r['other_distance']) < float(r['same_distance']) for r in valid]
            self.assertAlmostEqual(float(np.mean(wins)), result['other_closer_rate'])
            self.assertEqual(sum(float(r['other_distance']) == float(r['same_distance']) for r in valid), result['ties'])
            self.assertTrue(all(1 <= int(r['pass_index']) <= 10 for r in records))
        with (audit.OUT/audit.OUTPUTS[2]).open(newline='') as f:
            predictions = list(csv.DictReader(f))
        groups = defaultdict(list)
        for r in predictions:
            groups[r['view'], r['direction'], r['a'], r['b']].append(r)
        source = {r['timestamp_unix_ms']: r for r in audit.read('training_windows_v1.csv')}
        self.assertEqual(len(source), 1393)
        context = {r['timestamp_unix_ms']: r for r in report['geometry_context']}
        for result in report['neighbors']:
            selected = [r for r in neighbors if (r['view'], r['scope']) == (result['view'], result['scope']) and int(r['eligible'])]
            for query in [selected[0], selected[len(selected)//2], selected[-1]]:
                row = source[query['timestamp_unix_ms']]
                def allowed(r):
                    if result['scope'] == 'same_direction': return r['direction'] == row['direction']
                    if result['scope'] == 'gap_at_most_1000ms': return int(r['summary_to_ue_last_seen_ms']) <= 1000
                    if result['scope'] == 'interior_over_25m':
                        d = context[r['timestamp_unix_ms']]['boundary_distance_m']
                        return d is None or d > 25
                    return True
                reference = [r for r in source.values() if r['cv_fold'] != row['cv_fold'] and allowed(r)]
                xx = audit.matrix(reference, report['features'][result['view']])
                qq = audit.matrix([row], report['features'][result['view']])
                med = np.nanmedian(xx, axis=0); xx = np.where(np.isnan(xx), med, xx)
                scale = xx.std(axis=0); scale[scale == 0] = 1
                distances = np.sqrt(np.sum(((xx-np.where(np.isnan(qq), med, qq))/scale)**2, axis=1))
                same = np.array([r['segment_id'] == row['segment_id'] for r in reference])
                self.assertAlmostEqual(distances[same].min(), float(query['same_distance']))
                self.assertAlmostEqual(distances[~same].min(), float(query['other_distance']))
        for pair in report['pairs']:
            saved = groups[pair['view'], pair['direction'], pair['a'], pair['b']]
            self.assertEqual(len({r['timestamp_unix_ms'] for r in saved}), len(saved))
            if not saved: continue
            y = np.array([r['segment_id'] == pair['b'] for r in saved]); p = np.array([float(r['probability_b']) for r in saved])
            self.assertAlmostEqual(balanced_accuracy_score(y, p >= .5), pair['pooled']['balanced_accuracy'])
            self.assertAlmostEqual(roc_auc_score(y, p), pair['pooled']['auc'])
            for fold in pair['folds']:
                part = [r for r in saved if int(r['cv_fold']) == fold['fold']]
                if not fold['eligible']:
                    self.assertEqual(part, []); continue
                self.assertFalse(set(fold['fit_passes']) & set(fold['held_passes']))
                self.assertTrue(all(int(source[r['timestamp_unix_ms']]['pass_index']) in fold['held_passes'] for r in part))
                fitting = [r for r in source.values() if r['segment_id'] in [pair['a'], pair['b']]
                           and int(r['cv_fold']) != fold['fold']
                           and (pair['direction'] == 'all' or r['direction'] == pair['direction'])]
                fit_x = audit.matrix(fitting, report['features'][pair['view']])
                fit_med = np.nanmedian(fit_x, axis=0)
                np.testing.assert_allclose(fold['preprocessing']['medians'], fit_med)
                np.testing.assert_allclose(fold['preprocessing']['mean'], np.where(np.isnan(fit_x), fit_med, fit_x).mean(axis=0))
                x = audit.matrix([source[r['timestamp_unix_ms']] for r in part], report['features'][pair['view']])
                detail = fold['preprocessing']; med = np.array(detail['medians'])
                z = (np.where(np.isnan(x), med, x)-detail['mean'])/detail['scale']
                logits = z @ np.array(fold['coefficients'])[0] + fold['intercept'][0]
                np.testing.assert_allclose(1/(1+np.exp(-logits)), [float(r['probability_b']) for r in part], atol=1e-12)
                yy = np.array([r['segment_id'] == pair['b'] for r in part]); pp = np.array([float(r['probability_b']) for r in part])
                self.assertAlmostEqual(balanced_accuracy_score(yy, pp >= .5), fold['metrics']['balanced_accuracy'])
        self.assertEqual(len(report['pairs']), 480)


if __name__ == '__main__':
    unittest.main()
