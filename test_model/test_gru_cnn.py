"""Targeted checks for fixed GRU/CNN development comparison."""
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

import compare_gru_cnn as experiment


def independent_probabilities(family,sequence,direction,state):
    s={k:np.array(v,dtype=np.float32) for k,v in state.items()}
    if family=='GRU':
        hidden=np.zeros((len(sequence),16),np.float32)
        for step in range(sequence.shape[1]):
            xi=sequence[:,step]@s['recurrent.weight_ih_l0'].T+s['recurrent.bias_ih_l0']
            hh=hidden@s['recurrent.weight_hh_l0'].T+s['recurrent.bias_hh_l0']
            xr,xz,xn=np.split(xi,3,axis=1);hr,hz,hn=np.split(hh,3,axis=1)
            reset=1/(1+np.exp(-(xr+hr)));update=1/(1+np.exp(-(xz+hz)))
            candidate=np.tanh(xn+reset*hn);hidden=(1-update)*candidate+update*hidden
    else:
        signal=sequence.transpose(0,2,1)
        for layer,dilation in enumerate([1,2,4]):
            weight=s[f'convolutions.{layer}.weight'];bias=s[f'convolutions.{layer}.bias']
            result=np.broadcast_to(bias[None,:,None],(len(sequence),16,sequence.shape[1])).copy()
            for t in range(sequence.shape[1]):
                for k in range(3):
                    source=t-(2-k)*dilation
                    if source>=0:result[:,:,t]+=signal[:,:,source]@weight[:,:,k].T
            signal=np.maximum(result,0)
        hidden=signal[:,:,-1]
    logits=np.column_stack([hidden,direction.astype(np.float32)])@s['output.weight'].T+s['output.bias']
    logits-=logits.max(1,keepdims=True);p=np.exp(logits);return p/p.sum(1,keepdims=True)


class SequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def test_gradients_and_per_query_independence(self):
        for family in ['GRU','CNN']:
            torch.manual_seed(17);model=experiment.SequenceModel(family,2,2).double()
            x=torch.randn(2,10,2,dtype=torch.float64,requires_grad=True)
            d=torch.tensor([0.,1.],dtype=torch.float64,requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(model,(x,d),fast_mode=True))
            with torch.no_grad():
                np.testing.assert_allclose(model(x,d)[:1].numpy(),model(x[:1],d[:1]).numpy(),atol=1e-12)

    def test_cnn_prefix_causality_and_oldest_step_reach(self):
        torch.manual_seed(17);model=experiment.SequenceModel('CNN',2,2).double()
        x=torch.randn(2,10,2,dtype=torch.float64);changed=x.clone();changed[:,5:]=1000
        with torch.no_grad():
            torch.testing.assert_close(model.encode(x)[:,:5],model.encode(changed)[:,:5],rtol=0,atol=0)
        for parameter in model.parameters():torch.nn.init.constant_(parameter,.1)
        x=torch.ones(1,10,2,dtype=torch.float64,requires_grad=True)
        gradient=torch.autograd.grad(model.encode(x)[:,-1].sum(),x)[0]
        self.assertTrue((gradient[:,0]>0).all())

    def test_training_determinism_and_independent_forward(self):
        rng=np.random.RandomState(4);seq=rng.normal(size=(12,10,2)).astype(np.float32)
        direction=np.arange(12)%2;y=np.arange(12)%2;y[8:]=0;fit=np.arange(12)<8
        for family in ['GRU','CNN']:
            a,epochs,losses=experiment.train_model(family,seq,direction,y,fit,2,17,epochs=5,stop=~fit)
            b,ee,ll=experiment.train_model(family,seq,direction,y,fit,2,17,epochs=5,stop=~fit)
            self.assertEqual((epochs,losses),(ee,ll))
            for key in a.state_dict():torch.testing.assert_close(a.state_dict()[key],b.state_dict()[key],atol=0,rtol=0)
            expected=independent_probabilities(family,seq,direction,{k:v.detach().numpy().tolist() for k,v in a.state_dict().items()})
            np.testing.assert_allclose(expected,experiment.predict(a,seq,direction),atol=2e-6,rtol=2e-5)

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/experiment.FILES[-1];p.write_text('preserve')
            with patch.object(experiment,'OUT',Path(temp)),self.assertRaises(RuntimeError):experiment.run()
            self.assertEqual(p.read_text(),'preserve')

    def test_saved_metrics_preprocessing_and_all_weight_replays(self):
        report=json.loads((experiment.OUT/experiment.FILES[0]).read_text())
        self.assertEqual(report['code_sha256'],experiment.digest(Path(experiment.__file__)))
        for name,h in report['source_sha256'].items():self.assertEqual(h,experiment.digest(experiment.OUT/name))
        rows=experiment.read('training_windows_v1.csv');source={r['timestamp_unix_ms']:(i,r) for i,r in enumerate(rows)}
        lineage=experiment.read('rnn_history_lengths_v1_lineage.csv');windows=np.array([[source[t][0] for t in json.loads(r['history_timestamps'])] for r in lineage])
        selected=[source[r['timestamp_unix_ms']][1] for r in lineage];self.assertEqual(len(selected),1217)
        x=experiment.input_matrix(rows);direction=experiment.encode_direction(selected);labels=np.array(report['labels'])
        y=np.array([r['segment_id'] for r in selected]);folds=np.array([int(r['cv_fold']) for r in selected])
        with (experiment.OUT/experiment.FILES[1]).open(newline='') as f:saved=list(csv.DictReader(f))
        grouped=defaultdict(dict)
        for r in saved:grouped[r['family'],int(r['seed'])][r['timestamp_unix_ms']]=r
        references={(r['family'],int(r['seed']),r['timestamp_unix_ms']):r for r in experiment.read('rnn_history_lengths_v1_predictions.csv') if int(r['length'])==10}
        for run in report['runs']:
            records=grouped[run['family'],run['seed']];self.assertEqual(set(records),{r['timestamp_unix_ms'] for r in lineage})
            p=np.array([json.loads(records[r['timestamp_unix_ms']]['probabilities']) for r in lineage]);pred=labels[p.argmax(1)]
            self.assertAlmostEqual(accuracy_score(y,pred),run['metrics']['accuracy'])
            self.assertAlmostEqual(recall_score(y,pred,labels=labels,average='macro'),run['metrics']['macro_recall'])
            if run['family'] in ['RF','RNN']:
                expected=np.array([json.loads(references[run['family'],run['seed'],r['timestamp_unix_ms']]['probabilities']) for r in lineage])
                np.testing.assert_allclose(p,expected,atol=1e-12);continue
            for detail in run['training']:
                outer=detail['outer_fold'];fit=folds!=outer;held=~fit
                self.assertEqual(experiment.choose_epoch(detail['inner_log_loss'],30,.0001),(detail['selected_epoch'],detail['stopped_epoch']))
                self.assertFalse(set(detail['fit_passes'])&set(detail['held_passes']))
                self.assertFalse(set(detail['inner_fit_passes'])&set(detail['stop_passes']))
                for key,mask in [('inner_preprocessing',fit&(folds!=detail['inner_fold'])),('outer_preprocessing',fit)]:
                    indices=np.unique(windows[mask]);pre=detail[key]
                    self.assertEqual(pre['source_indices'],indices.tolist())
                    self.assertFalse(set(indices)&set(np.unique(windows[held])))
                    med=np.nanmedian(x[indices],axis=0);filled=np.where(np.isnan(x[indices]),med,x[indices]);scale=filled.std(0);scale[scale==0]=1
                    np.testing.assert_allclose(pre['median'],med);np.testing.assert_allclose(pre['mean'],filled.mean(0));np.testing.assert_allclose(pre['scale'],scale)
                pre=detail['outer_preprocessing'];seq=((np.where(np.isnan(x[windows[held]]),pre['median'],x[windows[held]])-pre['mean'])/pre['scale']).astype(np.float32)
                expected=independent_probabilities(run['family'],seq,direction[held],detail['state'])
                np.testing.assert_allclose(expected,p[held],atol=3e-6,rtol=3e-5)
        self.assertEqual(len(report['runs']),12)


if __name__=='__main__':unittest.main()
