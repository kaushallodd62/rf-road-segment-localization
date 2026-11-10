"""Targeted causal-history checks; no historical final evaluation is invoked."""
import csv
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import accuracy_score,recall_score

import compare_temporal_history as experiment
from group_stopped_mlp import choose_epoch,refit_outer


class TemporalTests(unittest.TestCase):
    def examples(self,times,labels=None):
        return [dict(timestamp_unix_ms=str(t),node_id='node',session_id='session',pass_index='1',cv_fold='1',direction='N',
                     segment_id=labels[i] if labels else 'a') for i,t in enumerate(times)]

    def test_causal_order_resets_and_label_independence(self):
        rows=self.examples([3000,1000,2000,5000,6000,7000],['b','a','a','b','b','b'])
        history,runs,resets=experiment.histories(rows,['ue']*len(rows))
        np.testing.assert_array_equal(history[0],[1,2,0]);np.testing.assert_array_equal(history[5],[3,4,5])
        self.assertEqual(resets,{'start':1,'gap':1})
        self.assertTrue((history[3]<0).all())
        original=history.copy();rows[0]['segment_id']='another-label'
        np.testing.assert_array_equal(experiment.histories(rows,['ue']*len(rows))[0],original)
        for field in ['node_id','session_id','pass_index','direction']:
            copy=self.examples([1000,2000,3000]);copy[2][field]='different'
            self.assertTrue((experiment.histories(copy,['ue']*3)[0][2]<0).all())
        self.assertTrue((experiment.histories(self.examples([1000,2000,3000]),['ue','ue','other'])[0][2]<0).all())
        rows=self.examples([1000,2000,3000]);rows[2]['cv_fold']='2'
        with self.assertRaises(ValueError):experiment.histories(rows,['ue']*3)
        with self.assertRaises(ValueError):experiment.histories(self.examples([1000,1000]),['ue']*2)

    def test_input_lags_and_no_future_dependency(self):
        rows=self.examples([1000,2000,3000,4000]);history,_,_=experiment.histories(rows,['ue']*4)
        x=np.array([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
        a=experiment.input_history(x,np.ones(4),history,np.array([2]))
        np.testing.assert_array_equal(a,[[5.,6.,3.,4.,1.,2.,1.]])
        x[3]=999
        np.testing.assert_array_equal(a,experiment.input_history(x,np.ones(4),history,np.array([2])))
        with self.assertRaises(ValueError):experiment.input_history(x,np.ones(4),history,np.array([1]))

    def test_transition_delay_and_missed_events(self):
        rows=self.examples([1000,2000,3000,4000,5000,6000,7000],['a','a','b','b','c','c','d'])
        history,runs,_=experiment.histories(rows,['ue']*7);queries=np.array([2,3,4,5,6])
        metrics=experiment.transition_metrics(rows,runs,queries,np.array(['a','b','b','b','d']))
        self.assertEqual(metrics['eligible_changes'],3)
        self.assertEqual(metrics['immediate_correct'],1)
        self.assertEqual(metrics['never_correct_before_exit'],1)
        self.assertEqual([e['first_correct_after_windows'] for e in metrics['events']],[1,None,0])
        self.assertEqual(metrics['events'][0]['first_correct_after_export_ms'],1000)
        self.assertEqual(metrics['prediction_switches'],2)

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/experiment.FILES[-1];p.write_text('preserve')
            with patch.object(experiment,'OUT',Path(temp)),self.assertRaises(RuntimeError):experiment.run()
            self.assertEqual(p.read_text(),'preserve')

    def test_saved_lineage_metrics_and_refit_predictions(self):
        report=json.loads((experiment.OUT/experiment.FILES[0]).read_text())
        self.assertEqual(experiment.digest(Path(experiment.__file__)),report['code_sha256'])
        for name,h in report['source_sha256'].items():self.assertEqual(experiment.digest(experiment.OUT/name),h)
        rows=experiment.read('training_windows_v1.csv');source={r['timestamp_unix_ms']:r for r in rows}
        with (experiment.OUT/experiment.FILES[2]).open(newline='') as f:lineage=list(csv.DictReader(f))
        self.assertEqual(len(lineage),1335);self.assertEqual(len({r['timestamp_unix_ms'] for r in lineage}),1335)
        selected=[source[r['timestamp_unix_ms']] for r in lineage];labels=np.array(report['labels']);y=np.array([r['segment_id'] for r in selected]);folds=np.array([int(r['cv_fold']) for r in selected])
        current=np.column_stack([experiment.input_matrix(selected),experiment.encode_direction(selected)])
        past=np.array([[*(float(source[r[t]][feature]) if source[r[t]][feature] else np.nan for t in ['timestamp_unix_ms','previous_timestamp_unix_ms','previous_previous_timestamp_unix_ms'] for feature in experiment.INPUTS),int(source[r['timestamp_unix_ms']]['direction']=='N')] for r in lineage])
        for line in lineage:
            rr=[source[line[t]] for t in ['previous_previous_timestamp_unix_ms','previous_timestamp_unix_ms','timestamp_unix_ms']]
            self.assertTrue(all(0<int(b['timestamp_unix_ms'])-int(a['timestamp_unix_ms'])<=1500 for a,b in zip(rr,rr[1:])))
            for field in ['node_id','session_id','pass_index','direction','cv_fold']:self.assertEqual(len({r[field] for r in rr}),1)
        with (experiment.OUT/experiment.FILES[1]).open(newline='') as f:predictions=list(csv.DictReader(f))
        groups=defaultdict(dict)
        for r in predictions:groups[r['model'],int(r['seed'])][r['timestamp_unix_ms']]=r
        saved={(r['model'],int(r['seed']),r['timestamp_unix_ms']):r for r in experiment.read('direction_feature_v1_predictions.csv')}
        for run in report['runs']:
            records=groups[run['model'],run['seed']];self.assertEqual(set(records),{r['timestamp_unix_ms'] for r in lineage})
            pp=np.array([json.loads(records[r['timestamp_unix_ms']]['probabilities']) for r in lineage]);pred=labels[pp.argmax(1)]
            self.assertAlmostEqual(accuracy_score(y,pred),run['metrics']['accuracy'])
            self.assertAlmostEqual(recall_score(y,pred,labels=labels,average='macro'),run['metrics']['macro_recall'])
            if run['arm'] in experiment.ARMS[:2]:
                expected=[]
                for row in lineage:
                    times=[row['timestamp_unix_ms']] if run['arm']==experiment.ARMS[0] else [row[k] for k in ['timestamp_unix_ms','previous_timestamp_unix_ms','previous_previous_timestamp_unix_ms']]
                    expected.append(np.mean([json.loads(saved[experiment.ADDED[run['kind']],run['seed'],t]['probabilities']) for t in times],axis=0))
                np.testing.assert_allclose(pp,expected,atol=1e-12)
                continue
            features=current if run['arm']==experiment.ARMS[2] else past
            for detail in run['training']:
                outer=detail['outer_fold'];fit=folds!=outer;held=~fit
                self.assertFalse(set(detail['outer_fit_passes'])&set(detail['outer_held_passes']))
                np.testing.assert_allclose(detail['outer_median'],np.nanmedian(features[fit],axis=0))
                if run['kind']=='MLP':
                    self.assertEqual(choose_epoch(detail['inner_log_loss']),(detail['selected_epoch'],detail['stopped_epoch']))
                    inner_fit=fit&(folds!=detail['inner_fold'])
                    np.testing.assert_allclose(detail['inner_expected_median'],np.nanmedian(features[inner_fit],axis=0))
                if outer==1 and run['seed']==17:
                    if run['kind']=='RF':
                        pipe=experiment.fit_added('RF',17,features,y,labels,folds,np.array([int(r['pass_index']) for r in selected]),1)[0]
                    else:pipe=refit_outer(features,y,fit,labels,17,detail['selected_epoch'])
                    np.testing.assert_allclose(pipe.predict_proba(features[held]),pp[held],atol=1e-12)
        self.assertEqual(len(report['runs']),24)


if __name__=='__main__':unittest.main()
