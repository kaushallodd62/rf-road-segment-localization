#!/usr/bin/env python3
"""Paired leaf-2/10 forest comparison; fixed training-only histories."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from compare_models import model
from compare_rnn_histories import (OUT, SEEDS, KEYS, common_histories, digest,
    encode_direction, input_matrix, read, evaluate, geography)
from compare_direction_feature import paired_changes
from compare_temporal_history import transition_metrics

STEM = 'history_forest_leaf_sizes_v1'


def fit_fold(x,y,folds,passes,outer,seed):
    fit=folds!=outer;held=~fit
    assert not set(passes[fit]) & set(passes[held])
    assert set(y[fit])==set(y)
    pipe=model('RF',seed)
    pipe[-1].set_params(min_samples_leaf=10,max_features=2)
    pipe.fit(x[fit],y[fit])
    np.testing.assert_allclose(pipe[0].statistics_,np.nanmedian(x[fit],axis=0))
    return pipe.predict_proba(x[held]),dict(fold=outer,
        fit_passes=sorted(set(passes[fit].tolist())),held_passes=sorted(set(passes[held].tolist())),
        median=pipe[0].statistics_.tolist(),mean_leaves=float(np.mean([t.get_n_leaves() for t in pipe[-1].estimators_])))


def run():
    paths=[OUT/(STEM+s) for s in ['.json','_predictions.csv','.html']]
    if any(p.exists() for p in paths):raise RuntimeError('Refusing to overwrite leaf comparison')
    base=json.loads((OUT/'rnn_history_lengths_v1.json').read_text())
    for f,h in base['source_sha256'].items():assert digest(OUT/f)==h
    for f,h in base['dependency_sha256'].items():assert digest(Path(__file__).parent/f)==h
    assert digest(Path(__file__).parent/'compare_rnn_histories.py')==base['code_sha256']
    rows=read('training_windows_v1.csv');ids=[r['timestamp_unix_ms'] for r in rows]
    assert len(rows)==len(set(ids))==1393
    assert all(r['split']=='train' and 1<=int(r['pass_index'])<=10 for r in rows)
    points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in set(ids)}
    assert hashlib.sha256(json.dumps([points[t] for t in ids],sort_keys=True).encode()).hexdigest()==base['training_points_sha256']
    q,w,runs,_=common_histories(rows,[points[t]['track_id'] for t in ids])
    lineage=read('rnn_history_lengths_v1_lineage.csv')
    assert len(q)==1217 and [ids[i] for i in q]==[r['timestamp_unix_ms'] for r in lineage]
    assert [[ids[i] for i in h] for h in w]==[json.loads(r['history_timestamps']) for r in lineage]
    selected=[rows[i] for i in q];x=input_matrix(rows)
    x=np.column_stack([x[w[:,::-1]].reshape(len(q),-1),encode_direction(selected)])
    assert x.shape==(1217,81)
    y=np.array([r['segment_id'] for r in selected]);labels=np.unique(y)
    assert labels.tolist()==base['labels'] and SEEDS==base['seeds']
    folds=np.array([int(r['cv_fold']) for r in selected]);passes=np.array([int(r['pass_index']) for r in selected])
    roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    geo=geography(selected,labels,[ids[i] for i in q],roads,points)
    saved={(int(r['seed']),r['timestamp_unix_ms']):r for r in read('rnn_history_lengths_v1_predictions.csv') if r['family']=='RF' and int(r['length'])==10}
    results=[];predictions=[];paired=[]
    for seed in SEEDS:
        previous=np.array([json.loads(saved[seed,ids[i]]['probabilities']) for i in q])
        candidate=np.empty_like(previous);seen=np.zeros(len(y),int);details=[]
        for outer in range(1,6):
            held=folds==outer
            candidate[held],detail=fit_fold(x,y,folds,passes,outer,seed)
            seen[held]+=1;details.append(detail)
            print(f'leaf=10 seed={seed} fold={outer}',flush=True)
        assert np.all(seen==1)
        for leaf,pp in [(2,previous),(10,candidate)]:
            assert np.isfinite(pp).all() and (pp>=0).all() and np.allclose(pp.sum(1),1)
            metrics,dist=evaluate(y,pp,labels,geo);pred=labels[pp.argmax(1)]
            if leaf==2:
                old=next(r for r in base['runs'] if r['family']=='RF' and r['length']==10 and r['seed']==seed)
                for k in KEYS:np.testing.assert_allclose(metrics[k],old['metrics'][k])
            results.append(dict(leaf=leaf,seed=seed,metrics=metrics,training=details if leaf==10 else [],
                transitions=transition_metrics(rows,runs,q,pred),
                folds=[dict(fold=f,metrics=evaluate(y[folds==f],pp[folds==f],labels,geo[folds==f])[0]) for f in range(1,6)]))
            for j,i in enumerate(q):
                predictions.append(dict(leaf=leaf,seed=seed,timestamp_unix_ms=ids[i],pass_index=int(passes[j]),cv_fold=int(folds[j]),
                    segment_id=y[j],predicted_segment_id=pred[j],probabilities=json.dumps(pp[j].tolist()),distance_m=float(dist[j])))
        paired.append(dict(seed=seed,changes=paired_changes(y,previous,candidate,labels,geo,folds)))
    summary=[]
    for leaf in [2,10]:
        rr=[r for r in results if r['leaf']==leaf]
        summary.append(dict(leaf=leaf,**{k:float(np.mean([r['metrics'][k] for r in rr])) for k in KEYS},
            **{k:float(np.mean([r['transitions'][k] for r in rr])) for k in ['immediate_correct','never_correct_before_exit','prediction_switches']}))
    report=dict(protocol='Only leaf size changes; ten observations and reported direction; training passes 1–10 only.',
        eligible_rows=len(q),input_columns=81,labels=labels.tolist(),display_names=base['display_names'],support=base['support'],coverage=base['coverage'],
        seeds=SEEDS,rf_settings=base['rf_settings'],leaf_sizes=[2,10],runs=results,summary=summary,paired=paired,
        code_sha256=digest(Path(__file__)),versions=base['versions'],
        dependency_sha256={f:digest(Path(__file__).parent/f) for f in ['compare_rnn_histories.py','compare_models.py','compare_direction_feature.py','compare_temporal_history.py','optimize_models.py']},
        source_sha256={f:digest(OUT/f) for f in ['training_windows_v1.csv','rnn_history_lengths_v1.json','rnn_history_lengths_v1_predictions.csv','rnn_history_lengths_v1_lineage.csv','road_segments.geojson']},training_points_sha256=base['training_points_sha256'])
    paths[0].write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with paths[1].open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    cells=''.join(f"<tr><td>{r['leaf']}</td><td>{100*r['accuracy']:.2f}%</td><td>{100*r['macro_recall']:.2f}%</td><td>{r['all_distance_p90']:.0f} m</td></tr>" for r in summary)
    segments=''
    for i,s in sorted(enumerate(labels),key=lambda t:base['display_names'][t[1]]):
        values=[100*np.mean([r['metrics']['per_segment_recall'][i] for r in results if r['leaf']==leaf]) for leaf in [2,10]]
        segments+=f"<tr><td>{base['display_names'][s]}</td><td>{base['support'][s]}</td><td>{values[0]:.1f}%</td><td>{values[1]:.1f}%</td></tr>"
    paths[2].write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Forest leaf sizes</title><style>body{font:16px/1.5 system-ui;max-width:900px;margin:32px auto}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>Forest leaf sizes with history and direction</h1><p>Same 1,217 queries, five training folds and three seeds. Ten observations include the current measurement and nine earlier measurements. Only the minimum fitting examples per leaf changes. Leaf 2 reuses saved predictions. Macro recall gives each segment equal weight.</p><table><tr><th>Minimum leaf size</th><th>Accuracy</th><th>Macro recall</th><th>90th percentile error</th></tr>'+cells+'</table><h2>Recall by segment</h2><table><tr><th>Segment</th><th>Windows</th><th>Leaf 2</th><th>Leaf 10</th></tr>'+segments+'</table><p>Small classes, reported direction, overlapping histories and one capture limit these findings. Validation and test were not evaluated.</p><a href="'+STEM+'.json">Full results</a></html>')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
