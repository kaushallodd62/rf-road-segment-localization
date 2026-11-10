"""Training-only checks for score blending, geographic tails and uncertainty sets."""
import ast
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import recall_score

import evaluate_balanced_localization as experiment
from compare_models import OUT, SEEDS, read
from compare_window_models import NEW
from compare_window_weights import NAMES
from segment_road import local_xy, point_to_line
from test_window_feature_dependency import classification


class BalancedLocalizationTests(unittest.TestCase):
    def test_set_coverage_specificity_and_ties(self):
        labels = np.array(['a', 'b', 'c']); y = np.array(['a', 'c'])
        p = np.array([[.6, .3, .1], [.1, .6, .3]]); geo = np.array([[2., 20., 200.], [500., 50., 5.]])
        one = experiment.candidate_set_metrics(y, p, labels, geo, 1)
        two = experiment.candidate_set_metrics(y, p, labels, geo, 2)
        self.assertEqual(one['coverage'], .5); self.assertEqual(two['coverage'], 1.)
        self.assertAlmostEqual(two['closest_centerline_median'], 3.5)
        self.assertAlmostEqual(two['farthest_centerline_median'], 35.)
        tied = experiment.candidate_set_metrics(np.array(['b']), np.array([[.4, .3, .3]]), labels, np.array([[1., 2., 3.]]), 2)
        self.assertEqual(tied['guaranteed_coverage'], 0.)
        self.assertEqual(tied['possible_coverage'], 1.)

    def test_screen_and_frontier_do_not_hide_tradeoffs(self):
        base = dict(model=experiment.BASELINE, accuracy=.5, macro_recall=.3, macro_top3_coverage=.7,
                    log_loss=2., multiclass_brier=.7, all_distance_p90=1000.)
        better = dict(base, model='better', accuracy=.55, macro_recall=.35, log_loss=1.9)
        tradeoff = dict(base, model='tradeoff', accuracy=.6, macro_recall=.2)
        screen, frontier = experiment.research_screen([base, better, tradeoff])
        self.assertTrue(screen[1]['passes_baseline_screen'])
        self.assertEqual(screen[2]['failed_axes'], ['macro_recall'])
        self.assertEqual(frontier[0]['dominated_by'], ['better'])
        self.assertTrue(frontier[2]['nondominated'])
        tree = ast.parse(Path(experiment.__file__).read_text())
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ['fit', 'partial_fit', 'fit_transform'] for n in ast.walk(tree)))

    def test_saved_blend_metrics_geography_and_guards(self):
        report = json.loads((OUT/'balanced_localization.json').read_text())
        rows = read('training_windows_v1.csv'); ids = [r['timestamp_unix_ms'] for r in rows]; needed = set(ids)
        self.assertEqual(len(rows), 1393); self.assertEqual({r['split'] for r in rows}, {'train'})
        self.assertEqual({int(r['pass_index']) for r in rows}, set(range(1, 11)))
        self.assertEqual(hashlib.sha256(Path(experiment.__file__).read_bytes()).hexdigest(), report['code_sha256'])
        for f, digest in report['source_sha256'].items(): self.assertEqual(hashlib.sha256((OUT/f).read_bytes()).hexdigest(), digest)
        y = np.array([r['segment_id'] for r in rows]); labels = np.array(report['labels']); folds = np.array([int(r['cv_fold']) for r in rows])
        unweighted = read('window_model_predictions.csv'); weighted = read('window_moderate_weights_predictions.csv')
        blend = read('balanced_localization_blend_predictions.csv')
        self.assertEqual(len(blend), len(rows)*len(SEEDS))
        sources = [(experiment.BASELINE, unweighted, NEW['RF']), ('MLP unweighted', unweighted, NEW['MLP']),
                   ('Forest moderate weights', weighted, NAMES['RF']), ('MLP moderate weights', weighted, NAMES['MLP'])]
        probabilities = {}
        for name, data, oldname in sources:
            for seed in SEEDS:
                ss = {r['timestamp_unix_ms']:r for r in data if r['model'] == oldname and int(r['seed']) == seed}
                self.assertEqual(set(ss), needed)
                probabilities[name, seed] = np.array([json.loads(ss[t]['probabilities']) for t in ids])
        points = {r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in needed}
        roads = {f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
        lon0 = np.mean([float(points[t]['longitude']) for t in ids]); lat0 = np.mean([float(points[t]['latitude']) for t in ids])
        for seed in SEEDS:
            ss = {r['timestamp_unix_ms']:r for r in blend if int(r['seed']) == seed}
            pp = np.array([json.loads(ss[t]['probabilities']) for t in ids]); probabilities[experiment.BLEND, seed] = pp
            np.testing.assert_allclose(pp, .5*(probabilities['Forest moderate weights', seed]+probabilities['MLP unweighted', seed]), rtol=0, atol=1e-15)
            self.assertTrue(np.isfinite(pp).all() and (pp >= 0).all()); np.testing.assert_allclose(pp.sum(axis=1), 1)
            for i in [0, 100, 500, 1000, 1392]:
                q = points[ids[i]]; xy = local_xy(float(q['longitude']), float(q['latitude']), lon0, lat0)
                pred = labels[pp[i].argmax()]
                self.assertAlmostEqual(point_to_line(xy, roads[pred], lon0, lat0)[0], float(ss[ids[i]]['distance_m']), places=6)
        old_reports = [json.loads((OUT/f).read_text()) for f in ['window_model_comparison.json', 'window_moderate_weights.json']]
        for run in report['runs']:
            name, seed = run['model'], run['seed']; p = probabilities[name, seed]; pred = labels[p.argmax(axis=1)]
            for k, value in classification(y, p, labels).items(): self.assertAlmostEqual(value, run['metrics'][k])
            np.testing.assert_allclose(recall_score(y, pred, labels=labels, average=None, zero_division=0), run['metrics']['per_segment_recall'])
            if name != experiment.BLEND:
                oldname = next(v[2] for v in sources if v[0] == name)
                old = next(r for rr in old_reports for r in rr['runs'] if r['model'] == oldname and r['seed'] == seed)
                for k in experiment.KEYS: self.assertAlmostEqual(old['metrics'][k], run['metrics'][k])
            d = []
            for i,t in enumerate(ids):
                q=points[t];xy=local_xy(float(q['longitude']),float(q['latitude']),lon0,lat0)
                d.append(point_to_line(xy,roads[pred[i]],lon0,lat0)[0])
            d=np.array(d)
            for threshold in experiment.DISTANCES:
                bad=d>threshold
                self.assertAlmostEqual(bad.mean(),run['metrics'][f'distance_over_{threshold}m'])
                self.assertAlmostEqual(np.mean([bad[y==v].mean() for v in labels]),run['metrics'][f'macro_distance_over_{threshold}m'])
            for v in run['candidate_sets']:
                k=v['size']; selected=p.argmax(axis=1)[:,None] if k==1 else np.argsort(p,axis=1)[:,-k:]
                hit=np.any(selected==np.searchsorted(labels,y)[:,None],axis=1)
                self.assertAlmostEqual(hit.mean(),v['coverage'])
                self.assertAlmostEqual(np.mean([hit[y==c].mean() for c in labels]),v['macro_coverage'])
                self.assertLessEqual(v['guaranteed_coverage'],v['coverage']+1e-12)
                self.assertLessEqual(v['coverage'],v['possible_coverage']+1e-12)
                if k==16:self.assertEqual(v['coverage'],1.)
        screen,frontier=experiment.research_screen(report['summary'])
        self.assertEqual(screen,report['baseline_screen']);self.assertEqual(frontier,report['pareto_frontier'])
        for pair in report['paired']:
            ix=np.ones(len(y),bool) if pair['fold']==0 else folds==pair['fold']
            a=probabilities[experiment.BASELINE,pair['seed']][ix];b=probabilities[pair['model'],pair['seed']][ix]
            ac=labels[a.argmax(axis=1)]==y[ix];bc=labels[b.argmax(axis=1)]==y[ix]
            self.assertEqual(int((ac&~bc).sum()),pair['baseline_only_correct'])
            self.assertEqual(int((~ac&bc).sum()),pair['candidate_only_correct'])
            am,bm=classification(y[ix],a,labels),classification(y[ix],b,labels)
            for k in am:self.assertAlmostEqual(bm[k]-am[k],pair['delta'][k])


if __name__ == '__main__': unittest.main()
