"""Checks for fit-only forest class weights and development predictions."""
import csv
import json
import unittest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

from compare_models import OUT
from moderate_forest_weights import moderate_weights
from optimize_models import load_split
from segment_road import local_xy, point_to_line


class ModerateForestChecks(unittest.TestCase):
    def test_weights_are_mean_one_and_follow_inverse_square_root(self):
        y=np.array(['a']*9+['b']*4);labels=np.array(['a','b'])
        weights,counts=moderate_weights(y,labels)
        np.testing.assert_array_equal(counts,[9,4]);self.assertAlmostEqual(weights['b']/weights['a'],1.5)
        self.assertAlmostEqual(np.mean([weights[v] for v in y]),1.)
        with self.assertRaises(AssertionError):moderate_weights(y,np.array(['a','b','c']))

    @unittest.skipUnless((OUT/'moderate_forest_weights.json').exists(),'run has not completed')
    def test_saved_development_results(self):
        report=json.loads((OUT/'moderate_forest_weights.json').read_text())
        previous=json.loads((OUT/'resumed_optimization.json').read_text())
        data={s:{r['timestamp_unix_ms']:r for r in load_split(s)} for s in ['train','validation']}
        labels=np.array(report['labels'])
        with (OUT/'moderate_forest_weights_predictions.csv').open(newline='') as f:predictions=list(csv.DictReader(f))
        self.assertEqual({r['split'] for r in predictions},{'training_cv','validation_development'})
        self.assertEqual(len(predictions),2*3*len(data['train'])+3*3*len(data['validation']))
        relevant=set(data['train'])|set(data['validation'])
        with (OUT/'location_points_segmented.csv').open(newline='') as f:points={r['timestamp_unix_ms']:r for r in csv.DictReader(f) if r['timestamp_unix_ms'] in relevant}
        roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
        lon0=np.mean([float(points[t]['longitude']) for t in data['train']]);lat0=np.mean([float(points[t]['latitude']) for t in data['train']])
        for run in report['runs']:
            subset=[r for r in predictions if r['split']==run['split'] and r['model']==run['name'] and int(r['seed'])==run['seed']]
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
            distances=np.array([float(r['distance_m']) for r in subset]);wrong=pred!=y
            self.assertAlmostEqual(np.median(distances),m['all_distance_median'])
            self.assertAlmostEqual(np.quantile(distances,.9,method='inverted_cdf'),m['all_distance_p90'])
            self.assertAlmostEqual(np.median(distances[wrong]),m['wrong_distance_median'])
            seen=set()
            for r in subset:
                original=data[split][r['timestamp_unix_ms']]
                self.assertEqual(r['segment_id'],original['segment_id']);self.assertEqual(r['pass_index'],original['pass_index'])
                if split=='train':self.assertEqual(r['cv_fold'],original['cv_fold'])
                if r['pass_index'] not in seen:
                    seen.add(r['pass_index']);point=points[r['timestamp_unix_ms']]
                    xy=local_xy(float(point['longitude']),float(point['latitude']),lon0,lat0)
                    self.assertAlmostEqual(float(r['distance_m']),point_to_line(xy,roads[r['predicted_segment_id']],lon0,lat0)[0])
            if run['split']=='training_cv':
                for f in run['training']:
                    fit=[r for r in data['train'].values() if int(r['cv_fold'])!=f['fold']]
                    held=[r for r in data['train'].values() if int(r['cv_fold'])==f['fold']]
                    self.assertEqual(set(f['fit_passes']),{int(r['pass_index']) for r in fit})
                    self.assertEqual(set(f['held_passes']),{int(r['pass_index']) for r in held})
                    self.assertFalse(set(f['fit_passes'])&set(f['held_passes']))
                    counts=np.array([sum(r['segment_id']==l for r in fit) for l in labels]);raw=1/np.sqrt(counts)
                    expected=raw/np.average(raw,weights=counts)
                    np.testing.assert_array_equal(f['counts'],counts);np.testing.assert_allclose(f['weights'],expected)
            elif run['training']['weights'] is not None:
                counts=np.array([sum(r['segment_id']==l for r in data['train'].values()) for l in labels]);raw=1/np.sqrt(counts)
                np.testing.assert_array_equal(run['training']['counts'],counts)
                np.testing.assert_allclose(run['training']['weights'],raw/np.average(raw,weights=counts))
        for kind in ['cv','validation']:
            key='training_cv' if kind=='cv' else 'validation_development'
            for summary in report[kind+'_summary']:
                rr=[x for x in report['runs'] if x['split']==key and x['name']==summary['name']]
                if rr:
                    for k in ['accuracy','macro_recall','log_loss','all_distance_p90']:
                        self.assertAlmostEqual(np.mean([v['metrics'][k] for v in rr]),summary[k])
                else:
                    old=next(v for v in previous[kind+'_summary'] if v['name']==summary['name'])
                    self.assertEqual(summary,old)


if __name__=='__main__':unittest.main()
