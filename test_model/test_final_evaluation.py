"""Checks for predeclared selection, frozen evaluation, and saved results."""
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

from compare_models import OUT, FEATURES, read
from final_evaluate import NAMES, SELECTION_RULE, digest, fit_candidate, metric_bundle, select
from group_stopped_mlp import choose_epoch
from segment_road import local_xy, point_to_line


class FinalEvaluationChecks(unittest.TestCase):
    def test_selection_prioritizes_macro_then_score_then_accuracy(self):
        rows=[dict(model=n,macro_recall=.3,log_loss=2.,accuracy=.5) for n in NAMES]
        rows[1]['macro_recall']=.31
        self.assertEqual(select(rows),NAMES[1])
        rows[0]['macro_recall']=.31;rows[0]['log_loss']=1.9
        self.assertEqual(select(rows),NAMES[0])
        rows[1]['log_loss']=1.9;rows[1]['accuracy']=.6
        self.assertEqual(select(rows),NAMES[1])
        rows[0]['accuracy']=.6
        self.assertEqual(select(rows),NAMES[0])

    def test_inner_split_excludes_stopping_group_from_fitting(self):
        labels=np.array(['a','b']);y=np.array(['a','b']*5);X=np.arange(100,dtype=float).reshape(10,10)
        folds=np.repeat(np.arange(1,6),2);passes=np.arange(1,11)
        class Pipe:
            classes_=labels
        with patch('final_evaluate.fit_inner',return_value=(2,[1.,.9])) as inner, patch('final_evaluate.refit_outer',return_value=Pipe()) as refit:
            pipe,info=fit_candidate('MLP grouped stop',17,X,y,folds,passes,labels)
        args=inner.call_args.args
        np.testing.assert_array_equal(args[2],folds!=5);np.testing.assert_array_equal(args[3],folds==5)
        np.testing.assert_array_equal(refit.call_args.args[2],np.ones(len(y),bool))
        self.assertEqual(info['inner_stopping_passes'],[9,10]);self.assertEqual(info['selected_epoch'],2)

    def test_metric_bundle_on_known_predictions(self):
        labels=np.array(['a','b','c','d']);y=np.array(['a','b','b','c'])
        p=np.array([[.7,.1,.1,.1],[.1,.6,.2,.1],[.5,.2,.2,.1],[.1,.2,.6,.1]])
        d=np.array([1.,3.,100.,4.]);m,bins,top3,true=metric_bundle(y,p,labels,d)
        self.assertAlmostEqual(m['accuracy'],.75)
        self.assertAlmostEqual(m['macro_recall'],(1+.5+1+0)/4)
        self.assertAlmostEqual(m['all_distance_median'],3.5)
        self.assertAlmostEqual(m['wrong_distance_median'],100.)
        self.assertEqual(sum(b['count'] for b in bins),4)
        np.testing.assert_allclose(true,[.7,.6,.2,.6])

    @unittest.skipUnless((OUT/'final_validation.json').exists() and (OUT/'final_selection.json').exists(), 'validation has not been scored')
    def test_validation_metrics_and_selection(self):
        from sklearn.metrics import confusion_matrix
        validation=json.loads((OUT/'final_validation.json').read_text())
        selection=json.loads((OUT/'final_selection.json').read_text())
        data={r['timestamp_unix_ms']:r for r in read('modeling_dataset.csv') if r['split']=='validation'}
        training=[r for r in read('modeling_dataset.csv') if r['split']=='train']
        labels=np.array(validation['labels']); predictions=read('final_validation_predictions.csv')
        self.assertEqual(len(predictions),4*3*len(data))
        self.assertEqual(digest(list(data.values())),selection['validation_rows_sha256'])
        self.assertEqual(validation['selected_model'],max(validation['summary'],key=lambda v:v['macro_recall'])['model'])
        self.assertEqual(selection['selected_model'],validation['selected_model'])
        all_ids=set(data)|{v['timestamp_unix_ms'] for v in training}
        points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in all_ids}
        roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
        lon0=np.mean([float(points[r['timestamp_unix_ms']]['longitude']) for r in training]);lat0=np.mean([float(points[r['timestamp_unix_ms']]['latitude']) for r in training])
        for run in validation['runs']:
            subset=[r for r in predictions if r['model']==run['model'] and int(r['seed'])==run['seed']]
            self.assertEqual(len(subset),len(data));self.assertEqual({r['timestamp_unix_ms'] for r in subset},set(data))
            y=np.array([r['segment_id'] for r in subset]);p=np.array([json.loads(r['probabilities']) for r in subset])
            self.assertTrue(np.isfinite(p).all() and (p>=0).all());np.testing.assert_allclose(p.sum(axis=1),1)
            pred=labels[p.argmax(axis=1)];np.testing.assert_array_equal(pred,[r['predicted_segment_id'] for r in subset])
            idx=np.searchsorted(labels,y);top3=(np.argsort(p,axis=1)[:,-3:]==idx[:,None]).any(axis=1)
            np.testing.assert_array_equal(top3,[r['in_top3']=='True' for r in subset])
            np.testing.assert_allclose(p[np.arange(len(y)),idx],[float(r['true_segment_score']) for r in subset])
            m=run['metrics'];self.assertAlmostEqual(accuracy_score(y,pred),m['accuracy'])
            self.assertAlmostEqual(recall_score(y,pred,labels=labels,average='macro',zero_division=0),m['macro_recall'])
            self.assertAlmostEqual(f1_score(y,pred,labels=labels,average='macro',zero_division=0),m['macro_f1'])
            self.assertAlmostEqual(log_loss(y,p,labels=labels),m['log_loss'])
            self.assertAlmostEqual(top3.mean(),m['top3_accuracy'])
            self.assertAlmostEqual(np.mean(np.sum((p-(y[:,None]==labels[None,:]))**2,axis=1)),m['multiclass_brier'])
            np.testing.assert_allclose(recall_score(y,pred,labels=labels,average=None,zero_division=0),m['per_segment_recall'])
            d=np.array([float(r['distance_m']) for r in subset]);wrong=y!=pred
            self.assertAlmostEqual(np.median(d),m['all_distance_median'])
            self.assertAlmostEqual(np.quantile(d,.9,method='inverted_cdf'),m['all_distance_p90'])
            self.assertAlmostEqual(np.median(d[wrong]),m['wrong_distance_median'])
            for row in subset:
                original=data[row['timestamp_unix_ms']]
                for k in ['segment_id','pass_index','direction']:self.assertEqual(row[k],original[k])
                point=points[row['timestamp_unix_ms']];xy=local_xy(float(point['longitude']),float(point['latitude']),lon0,lat0)
                self.assertAlmostEqual(float(row['distance_m']),point_to_line(xy,roads[row['predicted_segment_id']],lon0,lat0)[0])
            for v in run['passes']:
                mask=np.array([int(row['pass_index'])==v['pass_index'] for row in subset])
                self.assertEqual(int(mask.sum()),v['rows'])
                self.assertAlmostEqual(accuracy_score(y[mask],pred[mask]),v['metrics']['accuracy'])
            if run['model'].startswith('MLP'):
                d=run['training'];self.assertEqual(d['inner_stopping_passes'],[9,10])
                self.assertEqual(d['inner_fit_passes'],list(range(1,9)))
                self.assertEqual(choose_epoch(d['inner_log_loss']), (d['selected_epoch'],d['stopped_epoch']))
        for s in validation['summary']:
            rr=[v for v in validation['runs'] if v['model']==s['model']]
            for k in ['accuracy','macro_recall','log_loss','all_distance_p90']:
                self.assertAlmostEqual(float(np.mean([v['metrics'][k] for v in rr])),s[k])

    @unittest.skipUnless((OUT/'final_test.json').exists() and (OUT/'final_reference_seed17.joblib').exists(), 'test has not been scored')
    def test_saved_holdout_outputs_and_reference_pipeline(self):
        validation=json.loads((OUT/'final_validation.json').read_text())
        selection=json.loads((OUT/'final_selection.json').read_text())
        report=json.loads((OUT/'final_test.json').read_text())
        self.assertEqual(selection['rule'],SELECTION_RULE)
        self.assertEqual(selection['selected_model'],select(validation['summary']))
        self.assertEqual(report['selected_model'],selection['selected_model'])
        self.assertEqual(selection['validation_rows_sha256'],validation['held_rows_sha256'])
        data={r['timestamp_unix_ms']:r for r in read('modeling_dataset.csv') if r['split']=='test'}
        training=[r for r in read('modeling_dataset.csv') if r['split']=='train']
        self.assertEqual(digest(training),selection['training_rows_sha256'])
        self.assertEqual(digest(list(data.values())),report['held_rows_sha256'])
        labels=np.array(report['labels']); predictions=read('final_test_predictions.csv')
        self.assertEqual(len(predictions),3*len(data))
        points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in set(data)|{v['timestamp_unix_ms'] for v in training}}
        roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
        lon0=np.mean([float(points[r['timestamp_unix_ms']]['longitude']) for r in training]);lat0=np.mean([float(points[r['timestamp_unix_ms']]['latitude']) for r in training])
        for run in report['runs']:
            subset=[r for r in predictions if int(r['seed'])==run['seed']]
            self.assertEqual(len(subset),len(data));self.assertEqual({r['timestamp_unix_ms'] for r in subset},set(data))
            y=np.array([r['segment_id'] for r in subset]);p=np.array([json.loads(r['probabilities']) for r in subset])
            np.testing.assert_allclose(p.sum(axis=1),1);self.assertTrue(np.isfinite(p).all())
            pred=labels[p.argmax(axis=1)];np.testing.assert_array_equal(pred,[r['predicted_segment_id'] for r in subset])
            m=run['metrics'];self.assertAlmostEqual(accuracy_score(y,pred),m['accuracy'])
            self.assertAlmostEqual(recall_score(y,pred,labels=labels,average='macro',zero_division=0),m['macro_recall'])
            self.assertAlmostEqual(f1_score(y,pred,labels=labels,average='macro',zero_division=0),m['macro_f1'])
            self.assertAlmostEqual(log_loss(y,p,labels=labels),m['log_loss'])
            self.assertAlmostEqual(np.mean(np.sum((p-(y[:,None]==labels[None,:]))**2,axis=1)),m['multiclass_brier'])
            ix=np.searchsorted(labels,y);true=p[np.arange(len(y)),ix]
            top3=(np.argsort(p,axis=1)[:,-3:]==ix[:,None]).any(axis=1)
            np.testing.assert_allclose(true,[float(r['true_segment_score']) for r in subset])
            np.testing.assert_array_equal(top3,[r['in_top3']=='True' for r in subset])
            self.assertAlmostEqual(top3.mean(),m['top3_accuracy'])
            np.testing.assert_allclose(recall_score(y,pred,labels=labels,average=None,zero_division=0),m['per_segment_recall'])
            dist=np.array([float(r['distance_m']) for r in subset]);wrong=y!=pred
            self.assertAlmostEqual(np.median(dist),m['all_distance_median'])
            self.assertAlmostEqual(np.quantile(dist,.9,method='inverted_cdf'),m['all_distance_p90'])
            self.assertAlmostEqual(np.median(dist[wrong]),m['wrong_distance_median'])
            self.assertAlmostEqual(np.quantile(dist[wrong],.9,method='inverted_cdf'),m['wrong_distance_p90'])
            for row in subset:
                original=data[row['timestamp_unix_ms']]
                for k in ['segment_id','pass_index','direction']:self.assertEqual(row[k],original[k])
                point=points[row['timestamp_unix_ms']];xy=local_xy(float(point['longitude']),float(point['latitude']),lon0,lat0)
                self.assertAlmostEqual(float(row['distance_m']),point_to_line(xy,roads[row['predicted_segment_id']],lon0,lat0)[0])
            if run['seed']==17:
                X=np.array([[float(data[r['timestamp_unix_ms']][f]) if data[r['timestamp_unix_ms']][f] else np.nan for f in FEATURES] for r in subset])
                np.testing.assert_allclose(joblib.load(OUT/'final_reference_seed17.joblib').predict_proba(X),p)
        self.assertEqual(sum(report['support']),len(data))


if __name__=='__main__':unittest.main()
