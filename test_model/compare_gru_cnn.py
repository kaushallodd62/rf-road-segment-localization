#!/usr/bin/env python3
"""One fixed GRU and causal CNN on the existing ten-observation population."""
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from compare_rnn_histories import (INPUTS,OUT,SEEDS,SETTINGS,KEYS,common_histories,
                                   sequence_transform,predict,choose_epoch,digest,
                                   encode_direction,input_matrix,read,paired_changes)
from compare_temporal_history import transition_metrics
from optimize_models import evaluate,geography

STEM='gru_causal_cnn_v1'
FILES=[STEM+s for s in ['.json','_predictions.csv','.html']]
FAMILIES=['RF','RNN','GRU','CNN']


class SequenceModel(nn.Module):
    def __init__(self,family,features,classes):
        super().__init__();self.family=family
        if family=='GRU':self.recurrent=nn.GRU(features,16,batch_first=True)
        elif family=='CNN':self.convolutions=nn.ModuleList([nn.Conv1d(features if i==0 else 16,16,3,dilation=d) for i,d in enumerate([1,2,4])])
        else:raise ValueError('Expected GRU or CNN')
        self.output=nn.Linear(17,classes)

    def encode(self,sequence):
        if self.family=='GRU':return self.recurrent(sequence)[0]
        x=sequence.transpose(1,2)
        for layer in self.convolutions:
            x=torch.relu(layer(nn.functional.pad(x,(2*layer.dilation[0],0))))
        return x.transpose(1,2)

    def forward(self,sequence,direction):
        return self.output(torch.cat([self.encode(sequence)[:,-1],direction[:,None]],dim=1))


def train_model(family,sequence,direction,y,fit,classes,seed,epochs=None,stop=None):
    torch.manual_seed(seed);net=SequenceModel(family,sequence.shape[2],classes)
    optimizer=torch.optim.Adam(net.parameters(),lr=SETTINGS['learning_rate'],weight_decay=SETTINGS['weight_decay'])
    x=torch.from_numpy(sequence);d=torch.tensor(direction,dtype=torch.float32);target=torch.tensor(y,dtype=torch.long)
    rng=np.random.RandomState(seed);fitting=np.flatnonzero(fit);losses=[];best=float('inf');stale=0;selected=0
    for epoch in range(1,(epochs if epochs is not None else SETTINGS['max_epochs'])+1):
        net.train();order=rng.permutation(fitting)
        for start in range(0,len(order),SETTINGS['batch_size']):
            batch=order[start:start+SETTINGS['batch_size']];optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(net(x[batch],d[batch]),target[batch])
            if not torch.isfinite(loss):raise ValueError('Nonfinite sequence training loss')
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),SETTINGS['gradient_clip']);optimizer.step()
        if stop is not None:
            net.eval()
            with torch.no_grad():value=float(nn.functional.cross_entropy(net(x[stop],d[stop]),target[stop]))
            if not np.isfinite(value):raise ValueError('Nonfinite sequence stopping loss')
            losses.append(value)
            if value<best-SETTINGS['min_delta']:best=value;selected=epoch;stale=0
            else:stale+=1
            if stale>=SETTINGS['patience']:break
    if stop is not None:assert choose_epoch(losses,SETTINGS['patience'],SETTINGS['min_delta'])==(selected,len(losses))
    return net,selected,losses


def fit_sequence(family,x,windows,direction,y,folds,passes,outer,classes,seed):
    fit=folds!=outer;held=~fit;inner=outer%5+1;inner_fit=fit&(folds!=inner);stop=folds==inner
    for a,b in [(fit,held),(inner_fit,stop),(inner_fit,held),(stop,held)]:assert not set(passes[a])&set(passes[b])
    for mask in [fit,inner_fit]:assert set(y[mask])==set(range(classes))
    assert stop.any() and held.any()
    seq,pre_inner=sequence_transform(x,windows,inner_fit)
    _,epochs,losses=train_model(family,seq,direction,y,inner_fit,classes,seed,stop=stop)
    seq,pre_outer=sequence_transform(x,windows,fit)
    net,_,_=train_model(family,seq,direction,y,fit,classes,seed,epochs=epochs)
    return predict(net,seq[held],direction[held]),dict(outer_fold=outer,inner_fold=inner,selected_epoch=epochs,stopped_epoch=len(losses),
                hit_epoch_cap=len(losses)==SETTINGS['max_epochs'],inner_log_loss=losses,inner_preprocessing=pre_inner,outer_preprocessing=pre_outer,
                fit_passes=sorted(set(passes[fit].tolist())),held_passes=sorted(set(passes[held].tolist())),
                inner_fit_passes=sorted(set(passes[inner_fit].tolist())),stop_passes=sorted(set(passes[stop].tolist())),
                parameter_count=sum(p.numel() for p in net.parameters()),state={k:v.detach().numpy().tolist() for k,v in net.state_dict().items()})


def render(report):
    rows=''.join(f"<tr><td>{r['family']}</td><td>{100*r['accuracy']:.1f}%</td><td>{100*r['macro_recall']:.1f}%</td><td>{r['log_loss']:.3f}</td><td>{r['all_distance_p90']:.0f} m</td><td>{r['immediate_correct']:.1f}</td><td>{r['never_correct_before_exit']:.1f}</td></tr>" for r in report['summary'])
    cells=[]
    for i,s in sorted(enumerate(report['labels']),key=lambda item:report['display_names'][item[1]]):
        recalls=[np.mean([r['metrics']['per_segment_recall'][i] for r in report['runs'] if r['family']==f]) for f in FAMILIES]
        cells.append(f"<tr><td>{report['display_names'][s]}</td><td>{report['support'][s]}</td>"+''.join(f'<td>{100*v:.1f}%</td>' for v in recalls)+'</tr>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>GRU and causal CNN</title><style>body{font:16px/1.5 system-ui;color:#17283a;max-width:1150px;margin:32px auto;padding:0 20px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ccd5df;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}</style><h1>Do a GRU or causal CNN improve the ten-measurement result?</h1><p>All four models use the same 1,217 held-pass query observations, ten RF measurements per history, current reported direction, five training folds and three seeds. Forest and basic RNN results are reused without refitting.</p><p>The GRU has one 16-unit layer. The CNN has three 16-channel layers, kernel size three and dilations 1, 2 and 4. Padding is only on the left; each intermediate position can see only that position and earlier measurements. New neural models share the earlier RNN optimizer, stopping rule and fitting-only preprocessing.</p><h2>Overall results</h2><p>Macro recall weights every segment equally. Entry-correct counts correct predictions at the first new-segment observation. Missed changes have no correct prediction before the next recorded segment change. Counts average seeds; P90 is point-to-predicted-road error.</p><div class="scroll"><table><tr><th>Model</th><th>Accuracy</th><th>Macro recall</th><th>Log loss</th><th>P90 error</th><th>Entry correct</th><th>Missed changes</th></tr>'''+rows+'''</table></div><h2>Recall for each segment</h2><div class="scroll"><table><tr><th>Segment</th><th>Query windows</th><th>Forest</th><th>Basic RNN</th><th>GRU</th><th>Causal CNN</th></tr>'''+''.join(cells)+'''</table></div><h2>Limits</h2><ul>'''+''.join('<li>'+html.escape(v)+'</li>' for v in report['limitations'])+'''</ul><p><a href="gru_causal_cnn_v1.json">Full results, weights and stopping traces</a> · <a href="gru_causal_cnn_v1_predictions.csv">Predictions</a> · <a href="rnn_history_lengths_v1_lineage.csv">Shared ten-measurement histories</a></p></html>'''


def run():
    if any((OUT/f).exists() for f in FILES):raise RuntimeError('GRU/CNN artifacts exist; refusing overwrite')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    base=json.loads((OUT/'rnn_history_lengths_v1.json').read_text())
    for f,h in base['source_sha256'].items():assert digest(OUT/f)==h
    for f,h in base['dependency_sha256'].items():assert digest(Path(__file__).parent/f)==h
    assert digest(Path(__file__).parent/'compare_rnn_histories.py')==base['code_sha256']
    assert base['rnn_settings']==SETTINGS and base['versions']['torch']==torch.__version__
    rows=read('training_windows_v1.csv');ids=[r['timestamp_unix_ms'] for r in rows];needed=set(ids)
    assert len(rows)==len(needed)==1393 and all(r['split']=='train' and 1<=int(r['pass_index'])<=10 for r in rows)
    points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in needed}
    assert set(points)==needed and len({r['track_id'] for r in points.values()})==1
    assert hashlib.sha256(json.dumps([points[t] for t in ids],sort_keys=True).encode()).hexdigest()==base['training_points_sha256']
    queries,windows,runs,resets=common_histories(rows,[points[t]['track_id'] for t in ids]);assert len(queries)==1217
    lineage=read('rnn_history_lengths_v1_lineage.csv')
    assert [ids[i] for i in queries]==[r['timestamp_unix_ms'] for r in lineage]
    assert [[ids[i] for i in h] for h in windows]==[json.loads(r['history_timestamps']) for r in lineage]
    selected=[rows[i] for i in queries];x=input_matrix(rows);direction=encode_direction(selected)
    y=np.array([r['segment_id'] for r in selected]);labels=np.unique(y);assert labels.tolist()==base['labels']
    target=np.searchsorted(labels,y);folds=np.array([int(r['cv_fold']) for r in selected]);passes=np.array([int(r['pass_index']) for r in selected])
    roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    geo=geography(selected,labels,[ids[i] for i in queries],roads,points)
    reference={(r['family'],int(r['seed']),r['timestamp_unix_ms']):r for r in read('rnn_history_lengths_v1_predictions.csv') if int(r['length'])==10}
    results=[];predictions=[];probabilities={}
    def record(family,seed,pp,training):
        assert np.isfinite(pp).all() and (pp>=0).all() and np.allclose(pp.sum(1),1,atol=1e-6)
        pp=pp/pp.sum(1,keepdims=True);metrics,distances=evaluate(y,pp,labels,geo);pred=labels[pp.argmax(1)]
        confusion=Counter((a,b) for a,b in zip(y,pred) if a!=b)
        results.append(dict(family=family,seed=seed,metrics=metrics,training=training,transitions=transition_metrics(rows,runs,queries,pred),
                            folds=[dict(fold=f,metrics=evaluate(y[folds==f],pp[folds==f],labels,geo[folds==f])[0]) for f in range(1,6)],
                            confusions=[dict(actual=a,predicted=b,count=n) for (a,b),n in confusion.most_common()]))
        probabilities[family,seed]=pp
        for j,i in enumerate(queries):
            predictions.append(dict(family=family,seed=seed,timestamp_unix_ms=ids[i],pass_index=int(passes[j]),cv_fold=int(folds[j]),
                                    segment_id=y[j],predicted_segment_id=pred[j],probabilities=json.dumps(pp[j].tolist()),distance_m=float(distances[j])))
    for family in FAMILIES:
        for seed in SEEDS:
            if family in ['RF','RNN']:
                saved=[reference[family,seed,ids[i]] for i in queries]
                for row,old in zip(selected,saved):
                    for key in ['segment_id','pass_index','cv_fold']:assert row[key]==old[key]
                record(family,seed,np.array([json.loads(r['probabilities']) for r in saved]),[])
                old=next(r for r in base['runs'] if r['family']==family and r['length']==10 and r['seed']==seed)
                for key in KEYS:np.testing.assert_allclose(results[-1]['metrics'][key],old['metrics'][key])
                continue
            pp=np.empty((len(y),len(labels)));seen=np.zeros(len(y),int);details=[]
            for outer in range(1,6):
                held=folds==outer
                pp[held],detail=fit_sequence(family,x,windows,direction,target,folds,passes,outer,len(labels),seed)
                details.append(detail);seen[held]+=1
                print(f'{family} seed={seed} fold={outer} epoch={detail["selected_epoch"]}',flush=True)
            assert np.all(seen==1)
            record(family,seed,pp,details)
    paired=[dict(reference=ref,candidate=candidate,seed=seed,changes=paired_changes(y,probabilities[ref,seed],probabilities[candidate,seed],labels,geo,folds))
            for ref in ['RF','RNN'] for candidate in ['GRU','CNN'] for seed in SEEDS]
    summary=[]
    for family in FAMILIES:
        rr=[r for r in results if r['family']==family]
        summary.append(dict(family=family,**{k:float(np.mean([r['metrics'][k] for r in rr])) for k in KEYS},
                            **{k:float(np.mean([r['transitions'][k] for r in rr])) for k in ['immediate_correct','never_correct_before_exit','prediction_switches']}))
    report=dict(protocol='Fixed GRU and causal CNN against saved ten-observation forest/RNN; training only.',eligible_rows=len(y),length=10,
                rf_features=INPUTS,current_direction='reported north=1/south=0',labels=labels.tolist(),display_names=base['display_names'],support=base['support'],
                coverage=base['coverage'],settings=SETTINGS,cnn=dict(channels=16,kernel=3,dilations=[1,2,4],receptive_field=15,left_padding_only=True),
                seeds=SEEDS,runs=results,summary=summary,paired=paired,
                source_sha256={f:digest(OUT/f) for f in ['training_windows_v1.csv','rnn_history_lengths_v1.json','rnn_history_lengths_v1_predictions.csv','rnn_history_lengths_v1_lineage.csv','road_segments.geojson']},
                code_sha256=digest(Path(__file__)),dependency_sha256={f:digest(Path(__file__).parent/f) for f in ['compare_rnn_histories.py','compare_temporal_history.py','compare_direction_feature.py','optimize_models.py']},
                versions=base['versions'],
                limitations=['One fixed GRU and CNN recipe; parameter counts differ and no architecture is proven intrinsically best.',
                             'Reported direction and RF/location physical-time alignment remain assumptions.',
                             'Same capture, reused folds and overlapping histories limit generalization; tiny classes and absent held classes remain explicit.',
                             'First-correct delays apply only to detected changes; missed changes and immediate accuracy must also be considered.',
                             'No validation/test data, attack evaluation, independent capture, hyperparameter sweep or final-model promotion.'])
    (OUT/FILES[0]).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (OUT/FILES[1]).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    (OUT/FILES[2]).write_text(render(report));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
