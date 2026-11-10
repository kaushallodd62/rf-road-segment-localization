"""Check fitting-only missingness indicator results against saved predictions."""
import csv
import json
import unittest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

from compare_models import FEATURES, OUT
from compare_freshness_indicator import BASES, add_indicator
from optimize_models import load_split
from segment_road import local_xy, point_to_line


class FreshnessIndicatorChecks(unittest.TestCase):
    def test_indicator_is_per_row_and_retains_other_values(self):
        X=np.ones((3,len(FEATURES)));idx=FEATURES.index('message_freshness_ms')
        X[0,idx]=np.nan;X[1,idx]=56.;X[2,idx]=np.nan
        result=add_indicator(X)
        np.testing.assert_array_equal(result[:,-1],[1,0,1]);np.testing.assert_equal(result[:,:-1],X)
        self.assertEqual(result.shape[1],len(FEATURES)+1)

    @unittest.skipUnless((OUT/'freshness_indicator.json').exists(),'indicator run has not completed')
    def test_saved_predictions_and_pairs(self):
        report=json.loads((OUT/'freshness_indicator.json').read_text())
        data={s:{r['timestamp_unix_ms']:r for r in load_split(s)} for s in ['train','validation']}
        labels=np.array(report['labels'])
        with (OUT/'freshness_indicator_predictions.csv').open(newline='') as f:predictions=list(csv.DictReader(f))
        names={b[0] for b in BASES};baseline={}
        with (OUT/'resumed_optimization_predictions.csv').open(newline='') as f:
            for r in csv.DictReader(f):
                if r['model'] in names:baseline[r['split'],r['model'],int(r['seed']),r['timestamp_unix_ms']]=r
        self.assertEqual(len(predictions),3*3*(len(data['train'])+len(data['validation'])))
        self.assertEqual({r['split'] for r in predictions},{'training_cv','validation_development'})
        relevant=set(data['train'])|set(data['validation'])
        with (OUT/'location_points_segmented.csv').open(newline='') as f:points={r['timestamp_unix_ms']:r for r in csv.DictReader(f) if r['timestamp_unix_ms'] in relevant}
        roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
        lon0=np.mean([float(points[t]['longitude']) for t in data['train']]);lat0=np.mean([float(points[t]['latitude']) for t in data['train']])
        for run in [r for r in report['runs'] if r['variant']=='indicator']:
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
            np.testing.assert_allclose(recall_score(y,pred,labels=labels,average=None,zero_division=0),m['per_segment_recall'])
            distance=np.array([float(r['distance_m']) for r in subset]);wrong=pred!=y
            self.assertTrue(np.isfinite(distance).all() and (distance>=0).all())
            self.assertAlmostEqual(np.median(distance),m['all_distance_median']);self.assertAlmostEqual(np.quantile(distance,.9,method='inverted_cdf'),m['all_distance_p90'])
            self.assertAlmostEqual(np.median(distance[wrong]),m['wrong_distance_median']);self.assertAlmostEqual(np.quantile(distance[wrong],.9,method='inverted_cdf'),m['wrong_distance_p90'])
            old=[];missing=[];seen=set()
            for row in subset:
                t=row['timestamp_unix_ms'];original=data[split][t]
                self.assertEqual(row['segment_id'],original['segment_id']);self.assertEqual(row['pass_index'],original['pass_index'])
                if split=='train':self.assertEqual(row['cv_fold'],original['cv_fold'])
                self.assertEqual(row['freshness_missing']=='True',original['message_freshness_ms']=='')
                old.append(baseline[run['split'],run['base'],run['seed'],t]['predicted_segment_id']);missing.append(original['message_freshness_ms']=='')
                if row['pass_index'] not in seen:
                    seen.add(row['pass_index']);point=points[t];xy=local_xy(float(point['longitude']),float(point['latitude']),lon0,lat0)
                    self.assertAlmostEqual(float(row['distance_m']),point_to_line(xy,roads[row['predicted_segment_id']],lon0,lat0)[0])
            old=np.array(old);missing=np.array(missing);pair=next(v for v in report['pairs'] if v['split']==run['split'] and v['base']==run['base'] and v['seed']==run['seed'])
            self.assertEqual(int(((old!=y)&(pred==y)).sum()),pair['only_indicator_correct'])
            self.assertEqual(int(((old==y)&(pred!=y)).sum()),pair['only_baseline_correct'])
            for group in pair['subgroups']:
                mask=missing==group['missing'];self.assertEqual(int(mask.sum()),group['rows'])
                self.assertEqual(int((old[mask]==y[mask]).sum()),group['baseline_correct'])
                self.assertEqual(int((pred[mask]==y[mask]).sum()),group['indicator_correct'])
        for summary in report['summary']:
            rr=[v for v in report['runs'] if v['split']==summary['split'] and v['base']==summary['base'] and v['variant']==summary['variant']]
            for k in ['accuracy','macro_recall','log_loss','all_distance_p90']:
                self.assertAlmostEqual(float(np.mean([v['metrics'][k] for v in rr])),summary[k])


if __name__=='__main__':unittest.main()
