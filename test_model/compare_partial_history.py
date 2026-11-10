#!/usr/bin/env python3
"""Keep early observations using causal partial histories; training folds only."""
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits
from compare_models import model
from compare_rnn_histories import (OUT,SEEDS,KEYS,INPUTS,common_histories,digest,
    encode_direction,input_matrix,read,evaluate,geography)
from compare_direction_feature import paired_changes
from compare_temporal_history import histories,transition_metrics

STEM='partial_history_v1'
ARMS=['saved reference','matched format','preserved examples']


def partial_histories(rows,tracks):
    _,runs,resets=histories(rows,tracks)
    windows=np.full((len(rows),10),-1,int);seen=np.zeros(len(rows),int)
    for run in runs:
        for j,i in enumerate(run):
            h=run[max(0,j-9):j+1][::-1]
            windows[i,:len(h)]=h;seen[i]+=1
            assert h[0]==i and len({rows[k]['cv_fold'] for k in h})==1
            assert all(int(rows[a]['timestamp_unix_ms'])>int(rows[b]['timestamp_unix_ms']) for a,b in zip(h,h[1:]))
    assert np.all(seen==1)
    return windows,runs,resets


def partial_inputs(x,direction,windows):
    valid=windows>=0
    values=np.where(valid[:,:,None],x[np.maximum(windows,0)],np.nan)
    return np.column_stack([values.reshape(len(windows),-1),direction,valid.sum(1)])


def fit_fold(x,y,folds,passes,eligible,outer,seed):
    fit=(folds!=outer)&eligible;held=folds==outer
    assert not set(passes[fit])&set(passes[held]) and set(y[fit])==set(y)
    assert not np.isnan(x[fit]).all(0).any()
    pipe=model('RF',seed);pipe[-1].set_params(max_features=2)
    pipe.fit(x[fit],y[fit]);np.testing.assert_allclose(pipe[0].statistics_,np.nanmedian(x[fit],axis=0))
    assert np.array_equal(pipe.classes_,np.unique(y)) and pipe.n_features_in_==82
    return pipe,dict(fold=outer,fit_rows=int(fit.sum()),held_rows=int(held.sum()),
        fit_passes=sorted(set(passes[fit].tolist())),held_passes=sorted(set(passes[held].tolist())),
        median=pipe[0].statistics_.tolist(),fit_support={s:int((fit&(y==s)).sum()) for s in np.unique(y)})


def render(p):
    def table(rows):
        return '<table><tr><th>Model</th><th>Queries</th><th>Accuracy</th><th>Macro recall</th><th>90th percentile error</th></tr>'+''.join(f"<tr><td>{r['arm']} ({r['scope']})</td><td>{r['queries']}</td><td>{100*r['accuracy']:.2f}%</td><td>{100*r['macro_recall']:.2f}%</td><td>{r['all_distance_p90']:.0f} m</td></tr>" for r in rows)+'</table>'
    cells=''
    for i,s in sorted(enumerate(p['labels']),key=lambda z:p['display_names'][z[1]]):
        vals=[np.mean([r['metrics']['per_segment_recall'][i] for r in p['runs'] if r['arm']==arm and r['scope']=='common']) for arm in ARMS]
        cells+=f"<tr><td>{p['display_names'][s]}</td><td>{p['support']['common'][s]}</td><td>{p['support']['all'][s]}</td>"+''.join(f'<td>{100*v:.1f}%</td>' for v in vals)+'</tr>'
    return '<!doctype html><html lang="en"><meta charset="utf-8"><title>Preserving partial histories</title><style>body{font:16px/1.5 system-ui;max-width:1150px;margin:32px auto;padding:0 16px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ccc;text-align:right}td:first-child{text-align:left}</style><h1>Keep observations with incomplete history</h1><p>Use available measurements, fill absent older slots with fitting-only medians, and include the number available. All models retain direction, minimum leaf size 2, 300 trees and two candidate columns per split. Three seeds and five whole-pass training folds.</p><h2>Same 1,217 queries</h2>'+table([r for r in p['summary'] if r['scope']=='common'])+'<h2>Additional coverage</h2><p>These populations differ from the original queries, so their overall scores are separate results.</p>'+table([r for r in p['summary'] if r['scope']!='common'])+'<h2>Recall on the same queries</h2><table><tr><th>Segment</th><th>Original queries</th><th>Queries with recovery</th><th>Saved reference</th><th>Matched format</th><th>Preserved examples</th></tr>'+cells+'</table><h2>Limitations</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in p['limitations'])+'</ul><a href="'+STEM+'.json">Detailed metrics, directions, passes and support</a> · <a href="'+STEM+'_lineage.csv">Input lineage</a></html>'


def run():
    paths=[OUT/(STEM+s) for s in ['.json','_predictions.csv','_lineage.csv','.html']]
    if any(p.exists() for p in paths):raise RuntimeError('Refusing to overwrite partial-history artifacts')
    base=json.loads((OUT/'rnn_history_lengths_v1.json').read_text())
    for f,h in base['source_sha256'].items():assert digest(OUT/f)==h
    for f,h in base['dependency_sha256'].items():assert digest(Path(__file__).parent/f)==h
    assert digest(Path(__file__).parent/'compare_rnn_histories.py')==base['code_sha256']
    assert base['rf_settings']==dict(n_estimators=300,min_samples_leaf=2,max_features=2,n_jobs=1)
    assert base['versions']['sklearn']==sklearn.__version__
    rows=read('training_windows_v1.csv');ids=[r['timestamp_unix_ms'] for r in rows];needed=set(ids)
    assert len(rows)==len(needed)==1393 and all(r['split']=='train' and 1<=int(r['pass_index'])<=10 for r in rows)
    points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in needed}
    assert hashlib.sha256(json.dumps([points[t] for t in ids],sort_keys=True).encode()).hexdigest()==base['training_points_sha256']
    tracks=[points[t]['track_id'] for t in ids];windows,runs,resets=partial_histories(rows,tracks)
    q,old_windows,_,_=common_histories(rows,tracks);complete=(windows>=0).all(1)
    assert len(q)==1217 and np.array_equal(np.flatnonzero(complete),q)
    np.testing.assert_array_equal(windows[q],old_windows[:,::-1])
    old_lineage=read('rnn_history_lengths_v1_lineage.csv')
    assert [[ids[i] for i in h] for h in old_windows]==[json.loads(r['history_timestamps']) for r in old_lineage]
    raw=input_matrix(rows);direction=encode_direction(rows);x=partial_inputs(raw,direction,windows)
    old_flat=np.column_stack([raw[old_windows[:,::-1]].reshape(len(q),-1),direction[q]])
    np.testing.assert_allclose(x[q,:81],old_flat,equal_nan=True);assert x.shape==(1393,82)
    y=np.array([r['segment_id'] for r in rows]);labels=np.unique(y);assert labels.tolist()==base['labels']
    folds=np.array([int(r['cv_fold']) for r in rows]);passes=np.array([int(r['pass_index']) for r in rows])
    roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    geo=geography(rows,labels,ids,roads,points)
    masks=dict(common=complete,recovered=~complete,all=np.ones(len(y),bool))
    # Geography uses population-derived projection origins; reuse the old common-query scoring exactly.
    common_geo=geography([rows[i] for i in q],labels,[ids[i] for i in q],roads,points)
    saved={(int(r['seed']),r['timestamp_unix_ms']):r for r in read('rnn_history_lengths_v1_predictions.csv') if r['family']=='RF' and int(r['length'])==10}
    results=[];predictions=[];paired=[];probabilities={}
    def record(arm,scope,seed,pp,training):
        mask=masks[scope];queries=np.flatnonzero(mask);g=common_geo if scope=='common' else geo[mask]
        assert np.isfinite(pp).all() and (pp>=0).all() and np.allclose(pp.sum(1),1)
        m,d=evaluate(y[mask],pp,labels,g);pred=labels[pp.argmax(1)]
        results.append(dict(arm=arm,scope=scope,seed=seed,metrics=m,training=training,
            transitions=transition_metrics(rows,runs,queries,pred),
            folds=[dict(fold=f,metrics=evaluate(y[mask][folds[mask]==f],pp[folds[mask]==f],labels,g[folds[mask]==f])[0]) for f in range(1,6) if (folds[mask]==f).any()],
            direction_recall={s:{dd:dict(windows=int(((y[mask]==s)&(direction[mask]==value)).sum()),recall=float(np.mean(pred[(y[mask]==s)&(direction[mask]==value)]==s)) if ((y[mask]==s)&(direction[mask]==value)).any() else None) for dd,value in [('N',1),('S',0)]} for s in labels}))
        probabilities[arm,scope,seed]=pp
        for j,i in enumerate(queries):predictions.append(dict(arm=arm,scope=scope,seed=seed,timestamp_unix_ms=ids[i],pass_index=int(passes[i]),cv_fold=int(folds[i]),available_history=int(x[i,-1]),segment_id=y[i],predicted_segment_id=pred[j],probabilities=json.dumps(pp[j].tolist()),distance_m=float(d[j])))
    for seed in SEEDS:
        old=np.array([json.loads(saved[seed,ids[i]]['probabilities']) for i in q])
        for i in q:
            for key in ['segment_id','pass_index','cv_fold']:assert rows[i][key]==saved[seed,ids[i]][key]
        record(ARMS[0],'common',seed,old,[])
        old_run=next(r for r in base['runs'] if r['family']=='RF' and r['length']==10 and r['seed']==seed)
        for k in KEYS:np.testing.assert_allclose(results[-1]['metrics'][k],old_run['metrics'][k])
        for arm,eligible in [(ARMS[1],complete),(ARMS[2],masks['all'])]:
            pp=np.empty((len(y),len(labels)));seen=np.zeros(len(y),int);details=[]
            for outer in range(1,6):
                pipe,detail=fit_fold(x,y,folds,passes,eligible,outer,seed)
                held=folds==outer;pp[held]=pipe.predict_proba(x[held]);seen[held]+=1
                fit=(folds!=outer)&eligible
                detail['direction_support']={s:{d:int((fit&(y==s)&(direction==value)).sum()) for d,value in [('N',1),('S',0)]} for s in labels}
                details.append(detail);print(f'{arm} seed={seed} fold={outer}',flush=True)
            assert np.all(seen==1)
            for scope in (['common'] if arm==ARMS[1] else ['common','recovered','all']):record(arm,scope,seed,pp[masks[scope]],details)
        for ref in ARMS[:2]:paired.append(dict(reference=ref,candidate=ARMS[2],seed=seed,changes=paired_changes(y[q],probabilities[ref,'common',seed],probabilities[ARMS[2],'common',seed],labels,common_geo,folds[q])))
    summary=[]
    for arm,scope in [(a,'common') for a in ARMS]+[(ARMS[2],s) for s in ['recovered','all']]:
        rr=[r for r in results if r['arm']==arm and r['scope']==scope]
        summary.append(dict(arm=arm,scope=scope,queries=int(masks[scope].sum()),**{k:float(np.mean([r['metrics'][k] for r in rr])) for k in KEYS},
            **{k:float(np.mean([r['transitions'][k] for r in rr])) for k in ['eligible_changes','immediate_correct','never_correct_before_exit']}))
    support={scope:{s:int((mask&(y==s)).sum()) for s in labels} for scope,mask in masks.items()}
    report=dict(protocol='Fixed partial-history preservation comparison, training passes 1–10 only.',arms=ARMS,seeds=SEEDS,labels=labels.tolist(),display_names=base['display_names'],support=support,
        features=[f'lag_{lag}:{f}' for lag in range(10) for f in INPUTS]+['reported_direction_north','available_history'],rf_settings=base['rf_settings'],resets=resets,
        runs=results,summary=summary,paired=paired,versions=dict(numpy=np.__version__,sklearn=sklearn.__version__),
        coverage=[dict(fold=f,**{scope:{s:int((mask&(folds==f)&(y==s)).sum()) for s in labels} for scope,mask in masks.items()}) for f in range(1,6)],
        source_sha256={f:digest(OUT/f) for f in ['training_windows_v1.csv','rnn_history_lengths_v1.json','rnn_history_lengths_v1_predictions.csv','rnn_history_lengths_v1_lineage.csv','road_segments.geojson']},training_points_sha256=base['training_points_sha256'],
        code_sha256=digest(Path(__file__)),dependency_sha256={f:digest(Path(__file__).parent/f) for f in ['compare_models.py','compare_rnn_histories.py','compare_temporal_history.py','compare_direction_feature.py','optimize_models.py']},
        limitations=['Same capture, overlapping histories and reused training folds; no independent-drive generalization established.',
            'Available-history count can encode location relative to run starts; independent start locations are needed to assess that shortcut.',
            'Missing older slots use fitting-column medians, not physical RF measurements. This is one fixed representation.',
            'Reported direction and physical RF/position timestamp alignment remain assumptions.',
            'Recovered/all-window scores use different query populations from the saved reference. Fixed 16-label macro recall includes unsupported labels as zero.',
            'Class weighting is unchanged. Validation/test data were not evaluated; existing reference and artifacts are preserved.'])
    paths[0].write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for path,records in [(paths[1],predictions),(paths[2],[dict(timestamp_unix_ms=ids[i],pass_index=int(passes[i]),cv_fold=int(folds[i]),available_history=int(x[i,-1]),history_timestamps=json.dumps([ids[k] if k>=0 else None for k in windows[i]])) for i in range(len(rows))])]:
        with path.open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    paths[3].write_text(render(report));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
