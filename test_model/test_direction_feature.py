"""Development-only direction experiment checks."""
import csv
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import accuracy_score, recall_score

import compare_direction_feature as experiment
from group_stopped_mlp import choose_epoch, refit_outer


class DirectionTests(unittest.TestCase):
    def test_encoding_and_training_only_direction_prior(self):
        np.testing.assert_array_equal(experiment.encode_direction([{'direction':'S'},{'direction':'N'}]),[0,1])
        with self.assertRaises(ValueError): experiment.encode_direction([{'direction':'unknown'}])
        y=np.array(['a','a','b','b']); d=np.array([0,1,0,1]); fit=np.array([True,True,False,False]); held=~fit
        p, counts=experiment.direction_prior(y,d,fit,held,np.array(['a','b']))
        np.testing.assert_allclose(p,[[2/3,1/3],[2/3,1/3]])
        self.assertEqual(counts,{'0':[1,0],'1':[1,0]})

    def test_pass_balanced_effects_and_insufficient_support(self):
        rows=[]; values=[]; along=[]
        for p in range(1,11):
            for j in range(5 if p>1 else 50):
                north=p%2==0
                rows.append(dict(segment_id='a',pass_index=str(p),direction='N' if north else 'S'))
                values.append([10*int(north)]);along.append(10)
        effects=experiment.direction_effects(rows,np.array(values,float),np.array(along),['value'])
        self.assertEqual(len(effects),2)
        self.assertTrue(all(r['supported'] and r['north_minus_south']==10 and r['gap_to_change'] is None for r in effects))
        limited=experiment.direction_effects(rows[:50],np.array(values[:50],float),np.array(along[:50]),['value'])
        self.assertTrue(all(not r['supported'] for r in limited))

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/experiment.FILES[0];path.write_text('preserve')
            with patch.object(experiment,'OUT',Path(temp)),self.assertRaises(RuntimeError): experiment.run()
            self.assertEqual(path.read_text(),'preserve')

    def test_saved_metrics_folds_preprocessing_and_predictions(self):
        report=json.loads((experiment.OUT/experiment.FILES[0]).read_text())
        self.assertEqual(report['code_sha256'],experiment.digest(Path(experiment.__file__)))
        for name,h in report['source_sha256'].items(): self.assertEqual(h,experiment.digest(experiment.OUT/name))
        rows=experiment.read('training_windows_v1.csv'); ids=[r['timestamp_unix_ms'] for r in rows]
        self.assertTrue(all(r['split']=='train' and int(r['pass_index'])<=10 for r in rows))
        x=np.column_stack([experiment.input_matrix(rows),experiment.encode_direction(rows)])
        labels=np.array(report['labels']); y=np.array([r['segment_id'] for r in rows]); folds=np.array([int(r['cv_fold']) for r in rows])
        with (experiment.OUT/experiment.FILES[1]).open(newline='') as f: saved=list(csv.DictReader(f))
        grouped=defaultdict(dict)
        for r in saved: grouped[r['model'],int(r['seed'])][r['timestamp_unix_ms']]=r
        for run in report['runs']:
            records=grouped[run['model'],run['seed']]
            self.assertEqual(set(records),set(ids))
            ordered=[records[t] for t in ids]; p=np.array([json.loads(r['probabilities']) for r in ordered]); pred=labels[p.argmax(1)]
            np.testing.assert_allclose(p.sum(1),1)
            self.assertAlmostEqual(accuracy_score(y,pred),run['metrics']['accuracy'])
            self.assertAlmostEqual(recall_score(y,pred,labels=labels,average='macro'),run['metrics']['macro_recall'])
            for row,r in zip(rows,ordered):
                for key in ['segment_id','pass_index','cv_fold','direction']:self.assertEqual(row[key],r[key])
            if run['model'] not in experiment.ADDED.values(): continue
            for detail in run['training']:
                outer=detail['outer_fold'];fit=folds!=outer;held=~fit
                self.assertFalse(set(detail['outer_fit_passes'])&set(detail['outer_held_passes']))
                np.testing.assert_allclose(detail['outer_median'],np.nanmedian(x[fit],axis=0))
                if run['model']==experiment.ADDED['MLP']:
                    inner=detail['inner_fold'];inner_fit=fit&(folds!=inner)
                    self.assertEqual(choose_epoch(detail['inner_log_loss']),(detail['selected_epoch'],detail['stopped_epoch']))
                    np.testing.assert_allclose(detail['inner_expected_median'],np.nanmedian(x[inner_fit],axis=0))
                    med=np.nanmedian(x[fit],axis=0)
                    np.testing.assert_allclose(detail['outer_scale_mean'],np.where(np.isnan(x[fit]),med,x[fit]).mean(axis=0))
                if outer==1 and run['seed']==17:
                    if run['model']==experiment.ADDED['RF']:
                        pipe=experiment.model('RF',17);pipe[-1].set_params(max_features=2);pipe.fit(x[fit],y[fit])
                        self.assertTrue(all(t.max_features_==2 for t in pipe[-1].estimators_))
                    else:
                        pipe=refit_outer(x,y,fit,labels,17,detail['selected_epoch'])
                    np.testing.assert_allclose(pipe.predict_proba(x[held]),p[held],atol=1e-12)
                    wrong=x[held].copy();wrong[:,-1]=1-wrong[:,-1]
                    name=experiment.FLIPPED['RF' if run['model']==experiment.ADDED['RF'] else 'MLP']
                    expected=np.array([json.loads(grouped[name,17][ids[i]]['probabilities']) for i in np.flatnonzero(held)])
                    np.testing.assert_allclose(pipe.predict_proba(wrong),expected,atol=1e-12)
        self.assertEqual(len(report['runs']),19)


if __name__=='__main__':
    unittest.main()
