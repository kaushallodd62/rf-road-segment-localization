"""Verify grouped calibration splits and saved development results."""
import csv
import json
import unittest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

from compare_models import OUT
from group_calibrated_forest import BASES, grouped_splits
from optimize_models import load_split
from segment_road import local_xy, point_to_line


class GroupCalibratedForestChecks(unittest.TestCase):
    def test_inner_groups_are_disjoint_and_cover_once(self):
        y=np.array(['a','b']*4);labels=np.unique(y);folds=np.repeat([1,2,3,4],2);passes=np.arange(1,9)
        splits,meta=grouped_splits(y,folds,passes,labels)
        self.assertEqual(len(splits),4)
        np.testing.assert_array_equal(np.sort(np.concatenate([v[1] for v in splits])),np.arange(len(y)))
        for (fit,held),entry in zip(splits,meta):
            self.assertFalse(set(passes[fit])&set(passes[held]));self.assertEqual(entry['held_rows'],len(held))
        with self.assertRaises(AssertionError):grouped_splits(np.array(['a','b']),np.array([1,2]),np.array([1,2]),labels)

    def test_calibration_imputation_is_inner_fit_only(self):
        from unittest.mock import patch
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from group_calibrated_forest import fit_calibrated
        class RecordingImputer(SimpleImputer):
            seen=[]
            def fit(self,X,y=None):
                type(self).seen.append(set(np.asarray(X)[:,0].astype(int)))
                return super().fit(X,y)
        X=np.column_stack((np.arange(80),np.linspace(0,1,80)))
        X[::7,1]=np.nan;y=np.array(['a','b']*40);folds=np.repeat(np.arange(1,5),20);passes=folds
        def small_rf(kind,seed):return make_pipeline(RecordingImputer(strategy='median'),RandomForestClassifier(n_estimators=4,random_state=seed,n_jobs=1))
        with patch('group_calibrated_forest.model',side_effect=small_rf):
            fitted,metadata=fit_calibrated(X,y,folds,passes,np.unique(y),2,17)
        expected=[set(np.flatnonzero(folds!=f)) for f in range(1,5)]
        self.assertEqual(len(RecordingImputer.seen),5)
        self.assertEqual(RecordingImputer.seen[:4],expected)
        self.assertEqual(RecordingImputer.seen[4],set(range(80)))
        np.testing.assert_allclose(fitted.predict_proba(X).sum(axis=1),1)

    @unittest.skipUnless((OUT/'group_calibrated_forest.json').exists(),'calibration run has not completed')
    def test_saved_calibration_and_groups(self):
        report=json.loads((OUT/'group_calibrated_forest.json').read_text());names={b[0] for b in BASES}
        data={s:{r['timestamp_unix_ms']:r for r in load_split(s)} for s in ['train','validation']}
        labels=np.array(report['labels'])
        with (OUT/'group_calibrated_forest_predictions.csv').open(newline='') as f:predictions=list(csv.DictReader(f))
        old={}
        with (OUT/'resumed_optimization_predictions.csv').open(newline='') as f:
            for r in csv.DictReader(f):
                if r['model'] in names:old[r['split'],r['model'],int(r['seed']),r['timestamp_unix_ms']]=r
        self.assertEqual(len(predictions),2*3*(len(data['train'])+len(data['validation'])))
        self.assertEqual({r['split'] for r in predictions},{'training_cv','validation_development'})
        relevant=set(data['train'])|set(data['validation'])
        with (OUT/'location_points_segmented.csv').open(newline='') as f:points={r['timestamp_unix_ms']:r for r in csv.DictReader(f) if r['timestamp_unix_ms'] in relevant}
        roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
        lon0=np.mean([float(points[t]['longitude']) for t in data['train']]);lat0=np.mean([float(points[t]['latitude']) for t in data['train']])
        for run in [x for x in report['runs'] if x['variant']=='calibrated']:
            subset=[r for r in predictions if r['split']==run['split'] and r['base']==run['base'] and int(r['seed'])==run['seed']]
            split='train' if run['split']=='training_cv' else 'validation'
            self.assertEqual(len(subset),len(data[split]));self.assertEqual({r['timestamp_unix_ms'] for r in subset},set(data[split]))
            y=np.array([r['segment_id'] for r in subset]);p=np.array([json.loads(r['probabilities']) for r in subset]);pred=labels[p.argmax(axis=1)]
            self.assertTrue(np.isfinite(p).all() and (p>=0).all());np.testing.assert_allclose(p.sum(axis=1),1)
            np.testing.assert_array_equal(pred,[r['predicted_segment_id'] for r in subset]);m=run['metrics']
            self.assertAlmostEqual(accuracy_score(y,pred),m['accuracy'])
            self.assertAlmostEqual(recall_score(y,pred,labels=labels,average='macro',zero_division=0),m['macro_recall'])
            self.assertAlmostEqual(f1_score(y,pred,labels=labels,average='macro',zero_division=0),m['macro_f1'])
            self.assertAlmostEqual(log_loss(y,p,labels=labels),m['log_loss'])
            self.assertAlmostEqual(np.mean(np.sum((p-(y[:,None]==labels[None,:]))**2,axis=1)),m['multiclass_brier'])
            ix=np.searchsorted(labels,y);self.assertAlmostEqual((np.argsort(p,axis=1)[:,-3:]==ix[:,None]).any(axis=1).mean(),m['top3_accuracy'])
            self.assertEqual(int((p[np.arange(len(y)),ix]==0).sum()),m['true_score_zero'])
            np.testing.assert_allclose(recall_score(y,pred,labels=labels,average=None,zero_division=0),m['per_segment_recall'])
            d=np.array([float(r['distance_m']) for r in subset]);wrong=pred!=y
            self.assertAlmostEqual(np.median(d),m['all_distance_median']);self.assertAlmostEqual(np.quantile(d,.9,method='inverted_cdf'),m['all_distance_p90'])
            self.assertAlmostEqual(np.median(d[wrong]),m['wrong_distance_median']);self.assertAlmostEqual(np.quantile(d[wrong],.9,method='inverted_cdf'),m['wrong_distance_p90'])
            prior=[];seen=set()
            for row in subset:
                t=row['timestamp_unix_ms'];original=data[split][t]
                self.assertEqual(row['segment_id'],original['segment_id']);self.assertEqual(row['pass_index'],original['pass_index'])
                if split=='train':self.assertEqual(row['cv_fold'],original['cv_fold'])
                prior.append(old[run['split'],run['base'],run['seed'],t]['predicted_segment_id'])
                if row['pass_index'] not in seen:
                    seen.add(row['pass_index']);point=points[t];xy=local_xy(float(point['longitude']),float(point['latitude']),lon0,lat0)
                    self.assertAlmostEqual(float(row['distance_m']),point_to_line(xy,roads[row['predicted_segment_id']],lon0,lat0)[0])
            prior=np.array(prior);pair=next(x for x in report['pairs'] if x['split']==run['split'] and x['base']==run['base'] and x['seed']==run['seed'])
            self.assertEqual(int(((prior!=y)&(pred==y)).sum()),pair['only_calibrated_correct'])
            self.assertEqual(int(((prior==y)&(pred!=y)).sum()),pair['only_baseline_correct'])
            for outer in run['training']:
                self.assertFalse(set(outer['outer_fit_passes'])&set(outer['outer_held_passes']))
                union=[]
                for inner in outer['inner_splits']:
                    self.assertFalse(set(inner['fit_passes'])&set(inner['held_passes']))
                    self.assertEqual(set(inner['fit_passes'])|set(inner['held_passes']),set(outer['outer_fit_passes']))
                    self.assertFalse((set(inner['fit_passes'])|set(inner['held_passes']))&set(outer['outer_held_passes']))
                    union.extend(inner['held_passes'])
                self.assertEqual(sorted(union),sorted(outer['outer_fit_passes']))
        for s in report['summary']:
            rr=[v for v in report['runs'] if v['split']==s['split'] and v['base']==s['base'] and v['variant']==s['variant']]
            for k in ['accuracy','macro_recall','log_loss','multiclass_brier','true_score_zero','all_distance_p90']:
                self.assertAlmostEqual(np.mean([v['metrics'][k] for v in rr]),s[k])


if __name__=='__main__':unittest.main()
