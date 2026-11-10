"""Targeted sequence checks and saved model replay; development data only."""
import csv
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import accuracy_score,recall_score

import compare_rnn_histories as experiment


class RecurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def test_shared_queries_causal_windows_and_group_resets(self):
        rows=[dict(timestamp_unix_ms=str(1000*i),node_id='n',session_id='s',pass_index='1',direction='N',cv_fold='1',segment_id='a' if i<8 else 'b') for i in range(1,13)]
        q,h,_,_=experiment.common_histories(rows,['ue']*12)
        np.testing.assert_array_equal(q,[9,10,11]);np.testing.assert_array_equal(h[0],np.arange(10))
        for length in experiment.LENGTHS:
            self.assertTrue(np.all(h[:,-length:][:,-1]==q))
            self.assertTrue(np.all(h[:,-length:]<=q[:,None]))
        rows[5]['timestamp_unix_ms']='5500'
        for row in rows[6:]:row['timestamp_unix_ms']=str(int(row['timestamp_unix_ms'])+4000)
        q,_,_,_=experiment.common_histories(rows,['ue']*12)
        self.assertEqual(len(q),0)

    def test_shared_preprocessing_uses_unique_fitting_sources(self):
        x=np.array([[0.,np.nan],[2.,4.],[10.,8.],[999.,999.]])
        windows=np.array([[0,1],[1,2],[2,3]]);fit=np.array([True,True,False])
        z,d=experiment.sequence_transform(x,windows,fit)
        self.assertEqual(d['source_indices'],[0,1,2])
        np.testing.assert_allclose(d['median'],[2.,6.])
        np.testing.assert_allclose(d['mean'],[4.,6.])
        modified=x.copy();modified[3]=100000
        zz,dd=experiment.sequence_transform(modified,windows,fit)
        self.assertEqual(d,dd);np.testing.assert_array_equal(z[:2],zz[:2])

    def test_recurrent_gradients_hidden_reset_and_determinism(self):
        torch.manual_seed(17);model=experiment.SmallRNN(2,2).double()
        x=torch.randn(2,3,2,dtype=torch.float64,requires_grad=True);d=torch.tensor([0.,1.],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(model,(x,d),fast_mode=True))
        model.eval()
        with torch.no_grad():
            batch=model(x,d);single=model(x[:1],d[:1])
        np.testing.assert_allclose(batch[:1].numpy(),single.numpy(),atol=1e-12)
        rng=np.random.RandomState(4);seq=rng.normal(size=(12,3,2)).astype(np.float32);direction=np.arange(12)%2;y=np.arange(12)%2
        y[8:]=0  # A stopping portion can legitimately lack a rare fitting class.
        fit=np.arange(12)<8;stop=~fit
        a,epoch,losses=experiment.train_rnn(seq,direction,y,fit,2,17,epochs=5,stop=stop)
        b,e,ll=experiment.train_rnn(seq,direction,y,fit,2,17,epochs=5,stop=stop)
        self.assertEqual(epoch,e);self.assertEqual(losses,ll)
        for key in a.state_dict():torch.testing.assert_close(a.state_dict()[key],b.state_dict()[key],rtol=0,atol=0)

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/experiment.FILES[-1];path.write_text('preserve')
            with patch.object(experiment,'OUT',Path(temp)),self.assertRaises(RuntimeError):experiment.run()
            self.assertEqual(path.read_text(),'preserve')

    def test_saved_metrics_context_isolation_and_numpy_rnn_replay(self):
        r=json.loads((experiment.OUT/experiment.FILES[0]).read_text())
        self.assertEqual(r['code_sha256'],experiment.digest(Path(experiment.__file__)))
        for name,h in r['source_sha256'].items():self.assertEqual(h,experiment.digest(experiment.OUT/name))
        rows=experiment.read('training_windows_v1.csv');source={row['timestamp_unix_ms']:(i,row) for i,row in enumerate(rows)};x=experiment.input_matrix(rows)
        with (experiment.OUT/experiment.FILES[2]).open(newline='') as f:lineage=list(csv.DictReader(f))
        self.assertEqual(len(lineage),1217)
        windows=np.array([[source[t][0] for t in json.loads(row['history_timestamps'])] for row in lineage])
        selected=[source[row['timestamp_unix_ms']][1] for row in lineage];y=np.array([row['segment_id'] for row in selected]);folds=np.array([int(row['cv_fold']) for row in selected]);direction=experiment.encode_direction(selected);labels=np.array(r['labels'])
        for row,h in zip(lineage,windows):
            self.assertEqual(rows[h[-1]]['timestamp_unix_ms'],row['timestamp_unix_ms'])
            for field in ['pass_index','cv_fold','node_id','session_id','direction']:self.assertEqual(len({rows[i][field] for i in h}),1)
            self.assertTrue(all(0<int(rows[b]['timestamp_unix_ms'])-int(rows[a]['timestamp_unix_ms'])<=1500 for a,b in zip(h,h[1:])))
        with (experiment.OUT/experiment.FILES[1]).open(newline='') as f:predictions=list(csv.DictReader(f))
        grouped=defaultdict(dict)
        for row in predictions:grouped[row['family'],int(row['length']),int(row['seed'])][row['timestamp_unix_ms']]=row
        for run in r['runs']:
            saved=grouped[run['family'],run['length'],run['seed']]
            self.assertEqual(set(saved),{row['timestamp_unix_ms'] for row in lineage})
            pp=np.array([json.loads(saved[row['timestamp_unix_ms']]['probabilities']) for row in lineage]);pred=labels[pp.argmax(1)]
            self.assertAlmostEqual(accuracy_score(y,pred),run['metrics']['accuracy'])
            self.assertAlmostEqual(recall_score(y,pred,labels=labels,average='macro'),run['metrics']['macro_recall'])
            history=windows[:,-run['length']:]
            for detail in run['training']:
                outer=detail['outer_fold'];fit=folds!=outer;held=~fit
                if run['family']=='RF':
                    features=np.column_stack([x[history[:,::-1]].reshape(len(lineage),-1),direction])
                    np.testing.assert_allclose(detail['outer_median'],np.nanmedian(features[fit],axis=0))
                    self.assertFalse(set(detail['outer_fit_passes'])&set(detail['outer_held_passes']))
                    if outer==1 and run['seed']==17:
                        pipe,_=experiment.fit_added('RF',17,features,y,labels,folds,np.array([int(row['pass_index']) for row in selected]),outer)
                        np.testing.assert_allclose(pipe.predict_proba(features[held]),pp[held],atol=1e-12)
                    continue
                self.assertEqual(experiment.choose_epoch(detail['inner_log_loss'],30,.0001),(detail['selected_epoch'],detail['stopped_epoch']))
                self.assertFalse(set(detail['fit_passes'])&set(detail['held_passes']))
                self.assertFalse(set(detail['inner_fit_passes'])&set(detail['stop_passes']))
                for key,mask in [('inner_preprocessing',fit&(folds!=detail['inner_fold'])),('outer_preprocessing',fit)]:
                    indices=np.unique(history[mask]);pre=detail[key]
                    self.assertEqual(pre['source_indices'],indices.tolist())
                    self.assertFalse(set(indices)&set(np.unique(history[held])))
                    med=np.nanmedian(x[indices],axis=0);np.testing.assert_allclose(pre['median'],med)
                    np.testing.assert_allclose(pre['mean'],np.where(np.isnan(x[indices]),med,x[indices]).mean(0))
                pre=detail['outer_preprocessing'];seq=((np.where(np.isnan(x[history[held]]),pre['median'],x[history[held]])-pre['mean'])/pre['scale']).astype(np.float32)
                state={k:np.array(v,dtype=np.float32) for k,v in detail['state'].items()}
                hidden=np.zeros((held.sum(),16),np.float32)
                for t in range(run['length']):
                    hidden=np.tanh(seq[:,t]@state['recurrent.weight_ih_l0'].T+state['recurrent.bias_ih_l0']+hidden@state['recurrent.weight_hh_l0'].T+state['recurrent.bias_hh_l0'])
                logits=np.column_stack([hidden,direction[held].astype(np.float32)])@state['output.weight'].T+state['output.bias']
                logits-=logits.max(1,keepdims=True);prob=np.exp(logits);prob/=prob.sum(1,keepdims=True)
                np.testing.assert_allclose(prob,pp[held],atol=2e-6,rtol=2e-5)
        self.assertEqual(len(r['runs']),24)


if __name__=='__main__':unittest.main()
