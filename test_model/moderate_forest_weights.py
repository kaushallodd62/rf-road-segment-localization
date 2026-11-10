#!/usr/bin/env python3
"""Compare moderate RF weights on training CV and validation development only."""
import csv
import hashlib
import html
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from compare_models import OUT, SEEDS, model, read
from optimize_models import evaluate, feature_matrix, geography, load_split

BASES=[('RF leaf=2 features=sqrt weight=none',2,'none'),('RF leaf=2 features=sqrt weight=balanced_subsample',2,'balanced'),
       ('RF leaf=5 features=sqrt weight=none',5,'none'),('RF leaf=5 features=sqrt weight=balanced_subsample',5,'balanced')]
NEW=[('RF leaf=2 features=sqrt weight=moderate',2,'moderate'),('RF leaf=5 features=sqrt weight=moderate',5,'moderate')]
VALID=['RF leaf=2 features=sqrt weight=none','RF leaf=2 features=sqrt weight=balanced_subsample',
       'RF leaf=5 features=sqrt weight=balanced_subsample']+[n for n,_,_ in NEW]


def moderate_weights(y,labels):
    counts=np.array([(y==label).sum() for label in labels]);assert len(y) and np.all(counts>0) and set(y)==set(labels)
    w=1/np.sqrt(counts);w=w/np.mean(w[np.searchsorted(labels,y)])
    return dict(zip(labels,w)),counts


def fit_forest(leaf,weight,seed,X,y,labels):
    pipe=model('RF',seed);counts=None;weights=None
    if weight=='moderate':
        weights,counts=moderate_weights(y,labels);cw=weights
    elif weight=='balanced':cw='balanced_subsample'
    else:assert weight=='none';cw=None
    pipe[-1].set_params(min_samples_leaf=leaf,class_weight=cw)
    pipe.fit(X,y);np.testing.assert_allclose(pipe[0].statistics_,np.nanmedian(X,axis=0))
    assert np.array_equal(pipe.classes_,labels)
    return pipe,dict(counts=counts.tolist() if counts is not None else None,weights=[float(weights[l]) for l in labels] if weights else None)


def run():
    targets=[OUT/n for n in ('moderate_forest_weights.json','moderate_forest_weights_predictions.csv','moderate_forest_weights.html')]
    if any(p.exists() for p in targets):raise RuntimeError('Results already exist; refusing to overwrite')
    prev=json.loads((OUT/'resumed_optimization.json').read_text());train=load_split('train');val=load_split('validation')
    assert len(train)==1613 and len(val)==514 and SEEDS==prev['seeds'] and sklearn.__version__==prev['versions']['sklearn']
    assert hashlib.sha256(json.dumps(train,sort_keys=True).encode()).hexdigest()==prev['training_rows_sha256']
    assert hashlib.sha256(json.dumps(val,sort_keys=True).encode()).hexdigest()==prev['validation_rows_sha256']
    ids=[r['timestamp_unix_ms'] for r in train];vid=[r['timestamp_unix_ms'] for r in val]
    assert len(set(ids))==len(ids) and len(set(vid))==len(vid) and not set(ids)&set(vid)
    X=feature_matrix(train);V=feature_matrix(val);y=np.array([r['segment_id'] for r in train]);vy=np.array([r['segment_id'] for r in val]);labels=np.unique(y)
    folds=np.array([int(r['cv_fold']) for r in train]);passes=np.array([int(r['pass_index']) for r in train]);vp=np.array([int(r['pass_index']) for r in val])
    assert set(folds)==set(range(1,6)) and set(passes)==set(range(1,11)) and set(vp)==set(range(11,14)) and set(vy)==set(labels)
    relevant=set(ids)|set(vid)
    points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in relevant}
    roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    geo=geography(train,labels,ids,roads,points);vgeo=geography(val,labels,ids,roads,points)
    runs=[];output=[]
    for cname,leaf,weight in NEW:
        for seed in SEEDS:
            p=np.empty((len(y),len(labels)));seen=np.zeros(len(y),int);details=[]
            for f in range(1,6):
                fit=folds!=f;held=~fit;assert not set(passes[fit])&set(passes[held])
                pipe,meta=fit_forest(leaf,weight,seed,X[fit],y[fit],labels)
                p[held]=pipe.predict_proba(X[held]);seen[held]+=1
                details.append(dict(fold=f,fit_passes=sorted(set(passes[fit].tolist())),held_passes=sorted(set(passes[held].tolist())),**meta))
            assert np.all(seen==1) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(axis=1),1)
            m,d=evaluate(y,p,labels,geo);pred=labels[p.argmax(axis=1)]
            byfold=[dict(fold=f,rows=int((folds==f).sum()),metrics=evaluate(y[folds==f],p[folds==f],labels,geo[folds==f])[0]) for f in range(1,6)]
            runs.append(dict(split='training_cv',name=cname,seed=seed,metrics=m,groups=byfold,training=details))
            for i,r in enumerate(train):output.append(dict(split='training_cv',model=cname,seed=seed,timestamp_unix_ms=ids[i],pass_index=int(passes[i]),cv_fold=int(folds[i]),segment_id=y[i],predicted_segment_id=pred[i],probabilities=json.dumps(p[i].tolist()),distance_m=float(d[i])))
            print(f'CV {cname} seed={seed}: acc={m["accuracy"]:.3f}, recall={m["macro_recall"]:.3f}',flush=True)
    for cname,leaf,weight in [BASES[1]]+NEW:
        for seed in SEEDS:
            pipe,meta=fit_forest(leaf,weight,seed,X,y,labels);p=pipe.predict_proba(V)
            assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(axis=1),1)
            m,d=evaluate(vy,p,labels,vgeo);pred=labels[p.argmax(axis=1)]
            groups=[dict(pass_index=int(g),rows=int((vp==g).sum()),metrics=evaluate(vy[vp==g],p[vp==g],labels,vgeo[vp==g])[0]) for g in sorted(set(vp))]
            runs.append(dict(split='validation_development',name=cname,seed=seed,metrics=m,groups=groups,training=meta))
            for i,r in enumerate(val):output.append(dict(split='validation_development',model=cname,seed=seed,timestamp_unix_ms=vid[i],pass_index=int(vp[i]),cv_fold='',segment_id=vy[i],predicted_segment_id=pred[i],probabilities=json.dumps(p[i].tolist()),distance_m=float(d[i])))
            print(f'Validation {cname} seed={seed}: acc={m["accuracy"]:.3f}, recall={m["macro_recall"]:.3f}',flush=True)
    keys=['accuracy','macro_recall','macro_f1','top3_accuracy','log_loss','multiclass_brier','all_distance_median','all_distance_p90','wrong_distance_median','wrong_distance_p90']
    cv_names=[b[0] for b in BASES]+[b[0] for b in NEW]
    cv_summary=[next(s for s in prev['cv_summary'] if s['name']==n) for n,_,_ in BASES]
    cv_summary += [dict(name=n,**{k:float(np.mean([r['metrics'][k] for r in runs if r['split']=='training_cv' and r['name']==n])) for k in keys}) for n,_,_ in NEW]
    validation_summary=[]
    for n in VALID:
        if n in prev['shortlist']:validation_summary.append(next(s for s in prev['validation_summary'] if s['name']==n))
        else:validation_summary.append(dict(name=n,**{k:float(np.mean([r['metrics'][k] for r in runs if r['split']=='validation_development' and r['name']==n])) for k in keys}))
    result=dict(protocol='Fit-dependent moderate forest weights; paired grouped training CV and validation development only.',
                labels=labels.tolist(),display_names=prev['display_names'],training_support=prev['training_support'],validation_support=prev['validation_support'],
                seeds=SEEDS,training_rows_sha256=prev['training_rows_sha256'],validation_rows_sha256=prev['validation_rows_sha256'],
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),versions={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
                previous_predictions_sha256=hashlib.sha256((OUT/'resumed_optimization_predictions.csv').read_bytes()).hexdigest(),
                cv_summary=cv_summary,validation_summary=validation_summary,runs=runs,
                limitations=['Validation was already used for development and this comparison was chosen after earlier results.',
                  'Same capture and inferred passes; seeds reuse rows. Repeated searches can overfit CV and validation.',
                  'Moderate weights use fit-portion counts; balanced_subsample recalculates per bootstrap, so both strength and weighting method differ.',
                  'Scores are not calibrated; actual-point-to-predicted-centerline distance is not road-network distance.',
                  'Previously viewed test rows and results were excluded.'])
    targets[0].write_text(json.dumps(result,indent=2)+'\n')
    with targets[1].open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    render(result,targets[2])


def render(r,target):
    esc=html.escape
    parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Moderate forest weights</title><style>body{font:15px/1.5 system-ui;background:#f6f8fb;color:#18212b}main{max-width:1400px;margin:auto;padding:24px}table{border-collapse:collapse;background:white;width:100%;margin:14px 0}th,td{padding:7px;border:1px solid #dbe2eb;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}</style><main><h1>Moderate forest class weights</h1><p>Training CV and validation development only. The same seeds reuse rows. Macro recall and accuracy are reported separately; no automatic winner is selected.</p>''']
    for title,key in [('Grouped training CV','cv_summary'),('Validation development','validation_summary')]:
        parts.append('<h2>'+title+'</h2><div class="scroll"><table><tr><th>Configuration</th><th>Accuracy</th><th>Macro recall</th><th>Macro F1</th><th>Top 3</th><th>Log loss</th><th>Brier</th><th>All median / P90 m</th><th>Wrong median / P90 m</th></tr>')
        for s in r[key]:
            parts.append('<tr><td>'+esc(s['name'])+'</td>'+''.join(f'<td>{s[k]:.1%}</td>' for k in ['accuracy','macro_recall','macro_f1','top3_accuracy'])+f'<td>{s["log_loss"]:.3f}</td><td>{s["multiclass_brier"]:.3f}</td><td>{s["all_distance_median"]:.1f} / {s["all_distance_p90"]:.1f}</td><td>{s["wrong_distance_median"]:.1f} / {s["wrong_distance_p90"]:.1f}</td></tr>')
        parts.append('</table></div>')
    parts.append('<h2>Limits</h2><ul>'+''.join('<li>'+esc(x)+'</li>' for x in r['limitations'])+'</ul><p>Complete seed, fold, pass, weight and segment metrics: <a href="moderate_forest_weights.json">JSON</a>. New predictions: <a href="moderate_forest_weights_predictions.csv">CSV</a>. Baseline predictions remain in resumed_optimization_predictions.csv.</p></main></html>')
    target.write_text(''.join(parts))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
