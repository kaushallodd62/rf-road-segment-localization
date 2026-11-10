"""Independently verify grouped development search and saved predictions."""
import csv
import json
import unittest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

from compare_models import OUT, SEEDS
from optimize_models import CONFIGS, MLP_CONFIGS, RF_CONFIGS, load_split, name, selected_configs


class OptimizationChecks(unittest.TestCase):
    def test_predeclared_grid_and_shortlist(self):
        self.assertEqual(len(RF_CONFIGS),12);self.assertEqual(len(MLP_CONFIGS),3)
        self.assertEqual(len(set(name(c) for c in CONFIGS)),15)
        summary=[dict(name=name(c),config=c,macro_recall=float(i),log_loss=30.-i) for i,c in enumerate(CONFIGS)]
        chosen=selected_configs(summary)
        self.assertIn(name(RF_CONFIGS[-1]),chosen);self.assertIn(name(RF_CONFIGS[0]),chosen)
        self.assertIn(name(MLP_CONFIGS[-1]),chosen);self.assertIn(name(MLP_CONFIGS[1]),chosen)
        self.assertEqual(len(chosen),len(set(chosen)))
        with self.assertRaises(AssertionError):load_split('test')

    @unittest.skipUnless((OUT/'resumed_optimization.json').exists(),'search has not completed')
    def test_saved_development_results(self):
        report=json.loads((OUT/'resumed_optimization.json').read_text())
        rows={split:{r['timestamp_unix_ms']:r for r in load_split(split)} for split in ['train','validation']}
        with (OUT/'resumed_optimization_predictions.csv').open(newline='') as f: predictions=list(csv.DictReader(f))
        self.assertEqual(len(report['cv_runs']),15*3)
        self.assertEqual(len(report['validation_runs']),len(report['shortlist'])*3)
        self.assertEqual(len(predictions),15*3*len(rows['train'])+len(report['shortlist'])*3*len(rows['validation']))
        self.assertEqual({r['split'] for r in predictions},{'training_cv','validation_development'})
        labels=np.array(report['labels'])
        for split,run_key in [('train','cv_runs'),('validation','validation_runs')]:
            designation='training_cv' if split=='train' else 'validation_development'
            for run in report[run_key]:
                subset=[r for r in predictions if r['split']==designation and r['model']==run['name'] and int(r['seed'])==run['seed']]
                self.assertEqual(len(subset),len(rows[split]));self.assertEqual({r['timestamp_unix_ms'] for r in subset},set(rows[split]))
                y=np.array([r['segment_id'] for r in subset]);p=np.array([json.loads(r['probabilities']) for r in subset])
                self.assertTrue(np.isfinite(p).all() and (p>=0).all());np.testing.assert_allclose(p.sum(axis=1),1)
                pred=labels[p.argmax(axis=1)];np.testing.assert_array_equal(pred,[r['predicted_segment_id'] for r in subset])
                m=run['metrics'];self.assertAlmostEqual(accuracy_score(y,pred),m['accuracy'])
                self.assertAlmostEqual(recall_score(y,pred,labels=labels,average='macro',zero_division=0),m['macro_recall'])
                self.assertAlmostEqual(f1_score(y,pred,labels=labels,average='macro',zero_division=0),m['macro_f1'])
                self.assertAlmostEqual(log_loss(y,p,labels=labels),m['log_loss'])
                self.assertAlmostEqual(np.mean(np.sum((p-(y[:,None]==labels[None,:]))**2,axis=1)),m['multiclass_brier'])
                indices=np.searchsorted(labels,y)
                self.assertAlmostEqual((np.argsort(p,axis=1)[:,-3:]==indices[:,None]).any(axis=1).mean(),m['top3_accuracy'])
                np.testing.assert_allclose(recall_score(y,pred,labels=labels,average=None,zero_division=0),m['per_segment_recall'])
                d=np.array([float(r['distance_m']) for r in subset]);wrong=y!=pred
                self.assertTrue(np.isfinite(d).all() and (d>=0).all())
                self.assertAlmostEqual(np.median(d),m['all_distance_median']);self.assertAlmostEqual(np.quantile(d,.9,method='inverted_cdf'),m['all_distance_p90'])
                self.assertAlmostEqual(np.median(d[wrong]),m['wrong_distance_median']);self.assertAlmostEqual(np.quantile(d[wrong],.9,method='inverted_cdf'),m['wrong_distance_p90'])
                for row in subset:
                    old=rows[split][row['timestamp_unix_ms']]
                    self.assertEqual(row['segment_id'],old['segment_id']);self.assertEqual(row['pass_index'],old['pass_index'])
                    if split=='train':self.assertEqual(row['cv_fold'],old['cv_fold'])
                if split=='train' and not run['reused']:
                    for d in run['training']:
                        self.assertFalse(set(d['fit_passes'])&set(d['held_passes']))
                        if run['config']['family']=='MLP':
                            self.assertFalse(set(d['inner_fit_passes'])&set(d['inner_stop_passes']))
                            self.assertFalse((set(d['inner_fit_passes'])|set(d['inner_stop_passes']))&set(d['held_passes']))
        # The retained controls were fitted before this search; compare full vectors.
        with (OUT/'final_validation_predictions.csv').open(newline='') as f:
            old={(r['model'],int(r['seed']),r['timestamp_unix_ms']):r for r in csv.DictReader(f)}
        control={'RF leaf=2 features=sqrt weight=none':'Forest leaf 2',
                 'RF leaf=10 features=sqrt weight=none':'Forest leaf 10',
                 'MLP alpha=0.01':'MLP grouped stop'}
        for row in predictions:
            if row['split']=='validation_development' and row['model'] in control:
                prior=old[control[row['model']],int(row['seed']),row['timestamp_unix_ms']]
                self.assertEqual(row['predicted_segment_id'],prior['predicted_segment_id'])
                np.testing.assert_allclose(json.loads(row['probabilities']),json.loads(prior['probabilities']),rtol=0,atol=1e-14)
        from segment_road import local_xy, point_to_line
        relevant=set(rows['train'])|set(rows['validation'])
        with (OUT/'location_points_segmented.csv').open(newline='') as f:
            points={r['timestamp_unix_ms']:r for r in csv.DictReader(f) if r['timestamp_unix_ms'] in relevant}
        roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
        lon0=np.mean([float(points[t]['longitude']) for t in rows['train']]);lat0=np.mean([float(points[t]['latitude']) for t in rows['train']])
        # Check distinct models, seeds and passes against the actual geometry.
        observed=set()
        for row in predictions:
            key=(row['split'],row['model'],row['seed'],row['pass_index'])
            if key in observed:continue
            observed.add(key);p=points[row['timestamp_unix_ms']]
            xy=local_xy(float(p['longitude']),float(p['latitude']),lon0,lat0)
            self.assertAlmostEqual(float(row['distance_m']),point_to_line(xy,roads[row['predicted_segment_id']],lon0,lat0)[0])
        self.assertEqual(report['shortlist'],selected_configs(report['cv_summary']))
        for kind in ['cv','validation']:
            for summary in report[kind+'_summary']:
                rr=[r for r in report[kind+'_runs'] if r['name']==summary['name']]
                for k in ['accuracy','macro_recall','log_loss','all_distance_p90']:
                    self.assertAlmostEqual(float(np.mean([r['metrics'][k] for r in rr])),summary[k])


if __name__=='__main__':unittest.main()
