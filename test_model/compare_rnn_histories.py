#!/usr/bin/env python3
"""Fixed forest versus small RNN comparison at 1/3/5/10 observations."""
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from torch import nn

from compare_direction_feature import INPUTS, OUT, SEEDS, digest, encode_direction, fit_added, input_matrix, paired_changes, read
from compare_temporal_history import GAP_MS, histories, transition_metrics
from compare_window_models import KEYS
from group_stopped_mlp import choose_epoch
from optimize_models import evaluate, geography

LENGTHS = [1,3,5,10]
SETTINGS = dict(hidden_units=16,learning_rate=.001,weight_decay=.0001,batch_size=64,
                max_epochs=500,patience=30,min_delta=.0001,gradient_clip=1.0)
STEM = 'rnn_history_lengths_v1'
FILES = [STEM+s for s in ['.json','_predictions.csv','_lineage.csv','.html']]


class SmallRNN(nn.Module):
    def __init__(self, features, classes):
        super().__init__()
        self.recurrent = nn.RNN(features,SETTINGS['hidden_units'],batch_first=True,nonlinearity='tanh')
        self.output = nn.Linear(SETTINGS['hidden_units']+1,classes)

    def forward(self, sequence, direction):
        _, hidden = self.recurrent(sequence)
        return self.output(torch.cat([hidden[-1],direction[:,None]],dim=1))


def common_histories(rows, tracks):
    _,runs,resets = histories(rows,tracks)
    indices = {i:run[j-9:j+1] for run in runs for j,i in enumerate(run) if j>=9}
    queries = np.array(sorted(indices),int)
    windows = np.array([indices[i] for i in queries],int).reshape(-1,10)
    for q,h in zip(queries,windows):
        assert h[-1]==q and len({rows[i]['cv_fold'] for i in h})==1
        assert all(int(rows[a]['timestamp_unix_ms'])<int(rows[b]['timestamp_unix_ms']) for a,b in zip(h,h[1:]))
    return queries,windows,runs,resets


def sequence_transform(x, windows, fit):
    unique = np.unique(windows[fit])
    values = x[unique]
    if np.isnan(values).all(axis=0).any():raise ValueError('Entirely missing fitting feature')
    median = np.nanmedian(values,axis=0)
    filled = np.where(np.isnan(values),median,values)
    mean = filled.mean(axis=0);scale = filled.std(axis=0);scale[scale==0]=1
    sequence = ((np.where(np.isnan(x[windows]),median,x[windows])-mean)/scale).astype(np.float32)
    assert np.isfinite(sequence).all()
    return sequence,dict(source_indices=unique.tolist(),median=median.tolist(),mean=mean.tolist(),scale=scale.tolist())


def train_rnn(sequence,direction,y,fit,classes,seed,epochs=None,stop=None):
    torch.manual_seed(seed)
    net = SmallRNN(sequence.shape[2],classes)
    optimizer = torch.optim.Adam(net.parameters(),lr=SETTINGS['learning_rate'],weight_decay=SETTINGS['weight_decay'])
    x = torch.from_numpy(sequence);d = torch.tensor(direction,dtype=torch.float32);target = torch.tensor(y,dtype=torch.long)
    rng = np.random.RandomState(seed);fitting = np.flatnonzero(fit);losses = [];best = float('inf');stale = 0;selected = 0
    for epoch in range(1,(epochs if epochs is not None else SETTINGS['max_epochs'])+1):
        net.train();order = rng.permutation(fitting)
        for begin in range(0,len(order),SETTINGS['batch_size']):
            batch = order[begin:begin+SETTINGS['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(net(x[batch],d[batch]),target[batch])
            if not torch.isfinite(loss):raise ValueError('Nonfinite recurrent training loss')
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),SETTINGS['gradient_clip']);optimizer.step()
        if stop is not None:
            net.eval()
            with torch.no_grad(): value = float(nn.functional.cross_entropy(net(x[stop],d[stop]),target[stop]))
            if not np.isfinite(value):raise ValueError('Nonfinite recurrent stopping loss')
            losses.append(value)
            if value<best-SETTINGS['min_delta']:best=value;selected=epoch;stale=0
            else:stale+=1
            if stale>=SETTINGS['patience']:break
    if stop is not None:
        assert choose_epoch(losses,SETTINGS['patience'],SETTINGS['min_delta'])==(selected,len(losses))
    return net,selected,losses


def predict(net,sequence,direction):
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(torch.from_numpy(sequence),torch.tensor(direction,dtype=torch.float32)),dim=1).numpy().astype(float)


def fit_recurrent(x, windows, direction, y, folds, passes, outer, classes, seed):
    fit=folds!=outer;held=~fit;inner=outer%5+1;inner_fit=fit&(folds!=inner);stop=folds==inner
    for a,b in [(fit,held),(inner_fit,stop),(inner_fit,held),(stop,held)]:assert not set(passes[a])&set(passes[b])
    for mask in [fit,inner_fit]:assert set(y[mask])==set(range(classes))
    assert stop.any() and held.any()
    seq,pre_inner=sequence_transform(x,windows,inner_fit)
    _,epochs,losses=train_rnn(seq,direction,y,inner_fit,classes,seed,stop=stop)
    seq,pre_outer=sequence_transform(x,windows,fit)
    net,_,_=train_rnn(seq,direction,y,fit,classes,seed,epochs=epochs)
    probabilities=predict(net,seq[held],direction[held])
    return probabilities,dict(outer_fold=outer,inner_fold=inner,selected_epoch=epochs,stopped_epoch=len(losses),
                              hit_epoch_cap=len(losses)==SETTINGS['max_epochs'],inner_log_loss=losses,
                              inner_preprocessing=pre_inner,outer_preprocessing=pre_outer,
                              fit_passes=sorted(set(passes[fit].tolist())),held_passes=sorted(set(passes[held].tolist())),
                              inner_fit_passes=sorted(set(passes[inner_fit].tolist())),stop_passes=sorted(set(passes[stop].tolist())),
                              state={k:v.detach().numpy().tolist() for k,v in net.state_dict().items()})


def render(report):
    rows=''.join(f"<tr><td>{r['family']}</td><td>{r['length']}</td><td>{r['accuracy']*100:.1f}%</td><td>{r['macro_recall']*100:.1f}%</td><td>{r['log_loss']:.3f}</td><td>{r['all_distance_p90']:.0f} m</td><td>{r['immediate_correct']:.1f}</td><td>{r['never_correct_before_exit']:.1f}</td></tr>" for r in report['summary'])
    segment_tables=[]
    for family in ['RF','RNN']:
        cells=[]
        for index,s in sorted(enumerate(report['labels']),key=lambda v:report['display_names'][v[1]]):
            values=[np.mean([r['metrics']['per_segment_recall'][index] for r in report['runs'] if r['family']==family and r['length']==length]) for length in LENGTHS]
            cells.append(f"<tr><td>{report['display_names'][s]}</td><td>{report['support'][s]}</td>"+''.join(f'<td>{v*100:.1f}%</td>' for v in values)+'</tr>')
        segment_tables.append(f'<h2>{family}: recall for each segment</h2><div class="scroll"><table><tr><th>Segment</th><th>Windows</th>'+''.join(f'<th>{length} observations</th>' for length in LENGTHS)+'</tr>'+''.join(cells)+'</table></div>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>History lengths and RNN</title><style>body{font:16px/1.5 system-ui;color:#17283a;max-width:1200px;margin:32px auto;padding:0 20px}table{border-collapse:collapse}th,td{padding:7px;border:1px solid #ccd5df;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}</style><h1>How much RF history helps?</h1><p>Every model fits and evaluates the same 1,217 training-query observations in five whole-pass folds, with three seeds. Length includes the current measurement. Histories restart at group changes and exported gaps over 1.5 seconds.</p><p>The forest receives flattened RF measurements plus current direction. The RNN reads eight RF fields in time order, then combines its final 16-unit hidden state with current direction. One fixed architecture and optimizer are used at every length.</p><h2>Results</h2><p>Macro recall gives each segment equal importance. P90 is the point-to-predicted-road error that 90% of queries stay within. Entry-correct counts correct predictions at the first new-segment observation; missed changes have no correct prediction before the next label change. Counts average three seeds.</p><div class="scroll"><table><tr><th>Model</th><th>Observations</th><th>Accuracy</th><th>Macro recall</th><th>Log loss</th><th>P90 error</th><th>Entry correct</th><th>Missed changes</th></tr>'''+rows+'</table></div>'+''.join(segment_tables)+'''<h2>Limits</h2><ul>'''+''.join('<li>'+html.escape(s)+'</li>' for s in report['limitations'])+'''</ul><p><a href="rnn_history_lengths_v1.json">Full results, weights, support and stopping histories</a> · <a href="rnn_history_lengths_v1_predictions.csv">Predictions</a> · <a href="rnn_history_lengths_v1_lineage.csv">History timestamps</a></p></html>'''


def run():
    if any((OUT/f).exists() for f in FILES):raise RuntimeError('RNN history artifacts exist; refusing overwrite')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    base=json.loads((OUT/'temporal_history_v1.json').read_text())
    for f,h in base['source_sha256'].items():assert digest(OUT/f)==h
    for f,h in base['dependency_sha256'].items():assert digest(Path(__file__).parent/f)==h
    assert digest(Path(__file__).parent/'compare_temporal_history.py')==base['code_sha256']
    assert base['seeds']==SEEDS
    rows=read('training_windows_v1.csv');ids=[r['timestamp_unix_ms'] for r in rows];needed=set(ids)
    assert len(rows)==len(needed)==1393 and all(r['split']=='train' and 1<=int(r['pass_index'])<=10 for r in rows)
    points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in needed}
    assert set(points)==needed and len({r['track_id'] for r in points.values()})==1
    queries,windows,runs,resets=common_histories(rows,[points[t]['track_id'] for t in ids]);assert len(queries)==1217
    selected=[rows[i] for i in queries];x=input_matrix(rows);direction=encode_direction(selected)
    y=np.array([r['segment_id'] for r in selected]);labels=np.unique(y);assert labels.tolist()==base['labels']
    target=np.searchsorted(labels,y);folds=np.array([int(r['cv_fold']) for r in selected]);passes=np.array([int(r['pass_index']) for r in selected])
    coverage=[]
    for outer in range(1,6):
        inner=outer%5+1
        portions={name:dict(rows=int(mask.sum()),support={s:int((y[mask]==s).sum()) for s in labels}) for name,mask in
                  [('outer_fit',folds!=outer),('held',folds==outer),('inner_fit',(folds!=outer)&(folds!=inner)),('stop',folds==inner)]}
        assert all(min(portions[name]['support'].values())>0 for name in ['outer_fit','inner_fit'])
        coverage.append(dict(outer_fold=outer,inner_fold=inner,portions=portions))
    roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    geo=geography(selected,labels,[ids[i] for i in queries],roads,points)
    results=[];predictions=[];probabilities={}
    for length in LENGTHS:
        history=windows[:,-length:]
        flat=np.column_stack([x[history[:,::-1]].reshape(len(queries),-1),direction])
        for family in ['RF','RNN']:
            for seed in SEEDS:
                pp=np.empty((len(queries),len(labels)));seen=np.zeros(len(queries),int);details=[]
                for outer in range(1,6):
                    held=folds==outer
                    if family=='RF':
                        pipe,detail=fit_added('RF',seed,flat,y,labels,folds,passes,outer)
                        pp[held]=pipe.predict_proba(flat[held]);assert pipe.n_features_in_==8*length+1
                    else:pp[held],detail=fit_recurrent(x,history,direction,target,folds,passes,outer,len(labels),seed)
                    details.append(detail);seen[held]+=1
                    print(f'{family} length={length} seed={seed} fold={outer}'+(f' epoch={detail["selected_epoch"]}' if family=='RNN' else ''),flush=True)
                assert np.all(seen==1) and np.isfinite(pp).all() and (pp>=0).all() and np.allclose(pp.sum(1),1,atol=1e-6)
                pp/=pp.sum(1,keepdims=True)
                metrics,distances=evaluate(y,pp,labels,geo);pred=labels[pp.argmax(1)]
                transitions=transition_metrics(rows,runs,queries,pred)
                confusion=Counter((a,b) for a,b in zip(y,pred) if a!=b)
                results.append(dict(family=family,length=length,seed=seed,metrics=metrics,training=details,transitions=transitions,
                                    folds=[dict(fold=f,metrics=evaluate(y[folds==f],pp[folds==f],labels,geo[folds==f])[0]) for f in range(1,6)],
                                    confusions=[dict(actual=a,predicted=b,count=n) for (a,b),n in confusion.most_common()]))
                probabilities[family,length,seed]=pp
                for j,i in enumerate(queries):
                    predictions.append(dict(family=family,length=length,seed=seed,timestamp_unix_ms=ids[i],pass_index=int(passes[j]),cv_fold=int(folds[j]),
                                            segment_id=y[j],predicted_segment_id=pred[j],probabilities=json.dumps(pp[j].tolist()),distance_m=float(distances[j])))
    paired=[]
    for family in ['RF','RNN']:
        for length in LENGTHS[1:]:
            for seed in SEEDS:paired.append(dict(reference=f'{family} L1',candidate=f'{family} L{length}',seed=seed,changes=paired_changes(y,probabilities[family,1,seed],probabilities[family,length,seed],labels,geo,folds)))
    for length in LENGTHS:
        for seed in SEEDS:paired.append(dict(reference=f'RF L{length}',candidate=f'RNN L{length}',seed=seed,changes=paired_changes(y,probabilities['RF',length,seed],probabilities['RNN',length,seed],labels,geo,folds)))
    summary=[]
    for family in ['RF','RNN']:
        for length in LENGTHS:
            rr=[r for r in results if r['family']==family and r['length']==length]
            summary.append(dict(family=family,length=length,**{k:float(np.mean([r['metrics'][k] for r in rr])) for k in KEYS},
                                **{k:float(np.mean([r['transitions'][k] for r in rr])) for k in ['immediate_correct','never_correct_before_exit','prediction_switches']}))
    report=dict(protocol='Fixed forest/RNN at L=1/3/5/10 on shared training queries only.',lengths=LENGTHS,eligible_rows=len(queries),excluded_queries=1393-len(queries),
                resets=resets,gap_reset_ms=GAP_MS,rf_features=INPUTS,current_direction='reported_direction_north; N=1, S=0',
                seeds=SEEDS,labels=labels.tolist(),display_names=base['display_names'],support={s:int((y==s).sum()) for s in labels},coverage=coverage,
                rnn_settings=SETTINGS,rf_settings=base['rf_settings'],runs=results,summary=summary,paired=paired,
                source_sha256={f:digest(OUT/f) for f in ['training_windows_v1.csv','temporal_history_v1.json','road_segments.geojson']},
                training_points_sha256=hashlib.sha256(json.dumps([points[t] for t in ids],sort_keys=True).encode()).hexdigest(),
                code_sha256=digest(Path(__file__)),dependency_sha256={f:digest(Path(__file__).parent/f) for f in ['compare_temporal_history.py','compare_direction_feature.py','compare_window_models.py','compare_models.py','group_stopped_mlp.py','optimize_models.py','requirements-sequences.txt']},
                versions=dict(numpy=np.__version__,sklearn=sklearn.__version__,torch=torch.__version__),
                limitations=['Only one small tanh RNN recipe is tested; results do not establish whether other recurrent models help.',
                             'All lengths use the same query labels; longer histories legitimately expose additional earlier RF context.',
                             'Reused development folds and overlapping histories within one capture limit selection and generalization claims.',
                             'Reported direction and physical RF/location timing remain assumptions; exported delays are not serving latency.',
                             'Shared RNN preprocessing uses unique fitting context rows; forests use fitting lag-column medians. Preprocessing differs by family.',
                             'First-correct delay is conditional on detection; missed events and entry accuracy must also be considered.',
                             'No validation/test data, attacks, independent capture, deployment decision, or broader hyperparameter search.'])
    (OUT/FILES[0]).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lineage=[dict(timestamp_unix_ms=ids[q],history_timestamps=json.dumps([ids[i] for i in h]),pass_index=rows[q]['pass_index'],cv_fold=rows[q]['cv_fold']) for q,h in zip(queries,windows)]
    for name,records in [(FILES[1],predictions),(FILES[2],lineage)]:
        with (OUT/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (OUT/FILES[3]).write_text(render(report));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
