#!/usr/bin/env python3
"""Bounded grouped-CV search; validation is development data, test excluded."""
import csv
import hashlib
import html
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import log_loss, recall_score
from threadpoolctl import threadpool_limits

from compare_models import FEATURES, OUT, RF, SEEDS, model, read, scores
from group_stopped_mlp import MAX_EPOCHS, MIN_DELTA, PATIENCE, choose_epoch, make_mlp
from inspect_forest_leaf_scores import score_metrics
from segment_road import local_xy, point_to_line

RF_CONFIGS = [dict(family='RF', leaf=leaf, max_features=mf, class_weight=cw)
              for leaf,mf,cw in itertools.product((2,5,10),('sqrt',1.0),(None,'balanced_subsample'))]
MLP_CONFIGS = [dict(family='MLP',alpha=alpha) for alpha in (.001,.01,.1)]
CONFIGS = RF_CONFIGS+MLP_CONFIGS


def name(c):
    if c['family']=='RF':
        return f"RF leaf={c['leaf']} features={c['max_features']} weight={c['class_weight'] or 'none'}"
    return f"MLP alpha={c['alpha']}"


def fit_config(config, seed, X, y, labels, folds, passes, stop_fold):
    assert set(y)==set(labels) and not np.isnan(X).all(axis=0).any()
    if config['family']=='RF':
        pipe=model('RF',seed)
        pipe[-1].set_params(min_samples_leaf=config['leaf'],max_features=config['max_features'],class_weight=config['class_weight'])
        pipe.fit(X,y)
        np.testing.assert_allclose(pipe[0].statistics_,np.nanmedian(X,axis=0))
        return pipe,{}
    inner=folds==stop_fold;fit=~inner
    assert fit.any() and inner.any() and not (set(passes[fit])&set(passes[inner]))
    assert set(y[fit])==set(labels) and not np.isnan(X[fit]).all(axis=0).any()
    pipe=make_mlp(seed);pipe[-1].set_params(alpha=config['alpha'])
    a=pipe[:-1].fit_transform(X[fit]);b=pipe[:-1].transform(X[inner])
    np.testing.assert_allclose(pipe[0].statistics_,np.nanmedian(X[fit],axis=0))
    np.testing.assert_allclose(pipe[1].mean_,np.mean(pipe[0].transform(X[fit]),axis=0))
    history=[];best=float('inf');epoch_best=0;stale=0
    for epoch in range(1,MAX_EPOCHS+1):
        pipe[-1].partial_fit(a,y[fit],classes=labels)
        loss=float(log_loss(y[inner],pipe[-1].predict_proba(b),labels=labels));assert np.isfinite(loss)
        history.append(loss)
        if loss<best-MIN_DELTA:best=loss;epoch_best=epoch;stale=0
        else:stale+=1
        if stale>=PATIENCE:break
    assert choose_epoch(history)==(epoch_best,len(history))
    pipe=make_mlp(seed);pipe[-1].set_params(alpha=config['alpha'])
    a=pipe[:-1].fit_transform(X)
    np.testing.assert_allclose(pipe[0].statistics_,np.nanmedian(X,axis=0))
    np.testing.assert_allclose(pipe[1].mean_,np.mean(pipe[0].transform(X),axis=0))
    for _ in range(epoch_best):pipe[-1].partial_fit(a,y,classes=labels)
    assert np.array_equal(pipe.classes_,labels)
    return pipe,dict(inner_fold=int(stop_fold),inner_fit_passes=sorted(set(passes[fit].tolist())),
                     inner_stop_passes=sorted(set(passes[inner].tolist())),selected_epoch=epoch_best,
                     stopped_epoch=len(history),hit_epoch_cap=len(history)==MAX_EPOCHS,inner_log_loss=history)


def selected_configs(summary):
    chosen=[]
    for family in ('RF','MLP'):
        rr=[s for s in summary if s['config']['family']==family]
        for s in [max(rr,key=lambda x:x['macro_recall']),min(rr,key=lambda x:x['log_loss'])]:
            if s['name'] not in chosen:chosen.append(s['name'])
    for c in (RF_CONFIGS[0],MLP_CONFIGS[1]):
        if name(c) not in chosen:chosen.append(name(c))
    return chosen


def load_split(split):
    assert split in ('train','validation')
    with (OUT/'modeling_dataset.csv').open(newline='') as f:
        return [r for r in csv.DictReader(f) if r['split']==split]


def feature_matrix(rows):
    X=np.array([[float(r[f]) if r[f] else np.nan for f in FEATURES] for r in rows])
    assert not np.isinf(X).any()
    return X


def geography(rows, labels, train_points, roads, points):
    lon0=np.mean([float(points[t]['longitude']) for t in train_points]);lat0=np.mean([float(points[t]['latitude']) for t in train_points])
    distances=np.empty((len(rows),len(labels)))
    for i,row in enumerate(rows):
        p=points[row['timestamp_unix_ms']];assert p['matched_segment_id']==row['segment_id']
        xy=local_xy(float(p['longitude']),float(p['latitude']),lon0,lat0)
        for j,label in enumerate(labels):distances[i,j]=point_to_line(xy,roads[label],lon0,lat0)[0]
    assert np.isfinite(distances).all()
    return distances


def evaluate(y,p,labels,geo):
    m,bins,_,_=score_metrics(y,p,labels);pred=labels[p.argmax(axis=1)]
    d=geo[np.arange(len(y)),np.searchsorted(labels,pred)];wrong=pred!=y
    m.update(all_distance_median=float(np.median(d)),all_distance_p90=float(np.quantile(d,.9,method='inverted_cdf')),
             wrong_distance_median=float(np.median(d[wrong])) if wrong.any() else None,
             wrong_distance_p90=float(np.quantile(d[wrong],.9,method='inverted_cdf')) if wrong.any() else None,
             per_segment_recall=recall_score(y,pred,labels=labels,average=None,zero_division=0).tolist())
    return m,d


def run():
    targets=[OUT/n for n in ('resumed_optimization.json','resumed_optimization_predictions.csv','resumed_optimization.html')]
    if any(p.exists() for p in targets):raise RuntimeError('Search outputs already exist; refusing to overwrite')
    train=load_split('train');assert len(train)==1613
    baseline=json.loads((OUT/'model_comparison.json').read_text())
    digest=hashlib.sha256(json.dumps(train,sort_keys=True).encode()).hexdigest()
    assert digest==baseline['training_rows_sha256'] and FEATURES==baseline['features'] and RF==baseline['settings']['RF']
    assert SEEDS==baseline['seeds'] and sklearn.__version__==baseline['versions']['sklearn']
    ids=[r['timestamp_unix_ms'] for r in train];assert len(ids)==len(set(ids))
    X=feature_matrix(train);y=np.array([r['segment_id'] for r in train]);labels=np.unique(y)
    folds=np.array([int(r['cv_fold']) for r in train]);passes=np.array([int(r['pass_index']) for r in train])
    assert set(folds)==set(range(1,6)) and labels.tolist()==baseline['labels']
    needed=set(ids)
    points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in needed}
    roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    geo=geography(train,labels,ids,roads,points)
    saved={}
    for r in read('forest_leaf_scores_predictions.csv'):
        if int(r['leaf']) in (2,5,10):saved[('RF',int(r['leaf']),int(r['seed']),r['timestamp_unix_ms'])]=r
    for r in read('group_stopped_mlp_predictions.csv'):
        saved[('MLP',.01,int(r['seed']),r['timestamp_unix_ms'])]=r
    expected={(f,l,s,t) for f,l in [('RF',2),('RF',5),('RF',10),('MLP',.01)] for s in SEEDS for t in ids}
    assert set(saved)==expected
    runs=[];cv_rows=[]
    for config in CONFIGS:
        cname=name(config);reuse=(config['family']=='RF' and config['max_features']=='sqrt' and config['class_weight'] is None) or (config['family']=='MLP' and config['alpha']==.01)
        for seed in SEEDS:
            p=np.empty((len(y),len(labels)));seen=np.zeros(len(y),int);details=[]
            if reuse:
                identity=config['leaf'] if config['family']=='RF' else config['alpha']
                for i,t in enumerate(ids):
                    old=saved[config['family'],identity,seed,t]
                    assert old['segment_id']==y[i] and int(old['cv_fold'])==folds[i]
                    p[i]=json.loads(old['probabilities']);seen[i]+=1
            else:
                for outer in range(1,6):
                    fit=folds!=outer;held=~fit;assert not set(passes[fit])&set(passes[held])
                    pipe,detail=fit_config(config,seed,X[fit],y[fit],labels,folds[fit],passes[fit],outer%5+1)
                    p[held]=pipe.predict_proba(X[held]);seen[held]+=1
                    detail.update(fold=outer,fit_passes=sorted(set(passes[fit].tolist())),held_passes=sorted(set(passes[held].tolist())))
                    details.append(detail)
            assert np.all(seen==1) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(axis=1),1)
            m,d=evaluate(y,p,labels,geo);pred=labels[p.argmax(axis=1)]
            by_fold=[dict(fold=f,rows=int((folds==f).sum()),metrics=evaluate(y[folds==f],p[folds==f],labels,geo[folds==f])[0]) for f in range(1,6)]
            runs.append(dict(name=cname,config=config,seed=seed,metrics=m,folds=by_fold,training=details,reused=reuse))
            for i,row in enumerate(train):
                cv_rows.append(dict(split='training_cv',model=cname,seed=seed,timestamp_unix_ms=ids[i],pass_index=int(passes[i]),cv_fold=int(folds[i]),
                                    segment_id=y[i],predicted_segment_id=pred[i],probabilities=json.dumps(p[i].tolist()),distance_m=float(d[i])))
            print(f'CV {cname} seed={seed}: acc={m["accuracy"]:.3f} recall={m["macro_recall"]:.3f} loss={m["log_loss"]:.3f}',flush=True)
    keys=['accuracy','macro_recall','macro_f1','top3_accuracy','log_loss','multiclass_brier','all_distance_median','all_distance_p90','wrong_distance_median','wrong_distance_p90']
    cv_summary=[dict(name=name(c),config=c,**{k:float(np.mean([r['metrics'][k] for r in runs if r['name']==name(c)])) for k in keys}) for c in CONFIGS]
    selected=selected_configs(cv_summary)
    # This is the first point at which validation rows are loaded; test rows are never selected.
    val=load_split('validation');assert len(val)==514
    vid=[r['timestamp_unix_ms'] for r in val];assert len(vid)==len(set(vid)) and not set(ids)&set(vid)
    more={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in set(vid)};points.update(more)
    assert set(vid)<=set(points)
    vX=feature_matrix(val);vy=np.array([r['segment_id'] for r in val]);vpasses=np.array([int(r['pass_index']) for r in val])
    assert set(vy)==set(labels) and set(vpasses)==set(range(11,14)) and not set(passes)&set(vpasses)
    vg=geography(val,labels,ids,roads,points);vruns=[];vrows=[]
    for cname in selected:
        config=next(c for c in CONFIGS if name(c)==cname)
        for seed in SEEDS:
            pipe,detail=fit_config(config,seed,X,y,labels,folds,passes,5)
            p=pipe.predict_proba(vX);assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(axis=1),1)
            m,d=evaluate(vy,p,labels,vg);pred=labels[p.argmax(axis=1)]
            perpass=[dict(pass_index=int(z),rows=int((vpasses==z).sum()),metrics=evaluate(vy[vpasses==z],p[vpasses==z],labels,vg[vpasses==z])[0]) for z in sorted(set(vpasses))]
            vruns.append(dict(name=cname,config=config,seed=seed,metrics=m,passes=perpass,training=detail))
            for i,row in enumerate(val):
                vrows.append(dict(split='validation_development',model=cname,seed=seed,timestamp_unix_ms=vid[i],pass_index=int(vpasses[i]),cv_fold='',
                                  segment_id=vy[i],predicted_segment_id=pred[i],probabilities=json.dumps(p[i].tolist()),distance_m=float(d[i])))
            print(f'Validation {cname} seed={seed}: acc={m["accuracy"]:.3f} recall={m["macro_recall"]:.3f} loss={m["log_loss"]:.3f}',flush=True)
    vs=[dict(name=n,config=next(c for c in CONFIGS if name(c)==n),**{k:float(np.mean([r['metrics'][k] for r in vruns if r['name']==n])) for k in keys}) for n in selected]
    report=dict(protocol='Predeclared bounded search on grouped training CV; validation is development, test is excluded.',
                features=FEATURES,labels=labels.tolist(),display_names=baseline['display_names'],seeds=SEEDS,configs=CONFIGS,
                training_rows_sha256=digest,validation_rows_sha256=hashlib.sha256(json.dumps(val,sort_keys=True).encode()).hexdigest(),
                versions={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_hashes={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['forest_leaf_scores_predictions.csv','group_stopped_mlp_predictions.csv','location_points_segmented.csv','road_segments.geojson']},
                shortlist_rule='Within each family include best mean training-CV macro recall and lowest mean training-CV log loss, plus RF leaf 2 sqrt unweighted and MLP alpha 0.01 references; remove duplicates.',
                shortlist=selected,cv_summary=cv_summary,cv_runs=runs,validation_summary=vs,validation_runs=vruns,
                training_support=[int((y==label).sum()) for label in labels],validation_support=[int((vy==label).sum()) for label in labels],
                limitations=['Validation has been used for development and is not a fresh independent estimate.',
                  'The earlier test split and results are excluded from this work; a previously viewed test is not a pristine holdout.',
                  'Repeated comparisons on the same CV folds and validation risk overfitting model selection.',
                  'One vehicle and capture; grouped passes and repeated seeds do not establish independent-drive performance.',
                  'Log loss and Brier do not establish score calibration; no attack evaluation or decision threshold is tested.',
                  'Distance is point to predicted centerline, not road-network distance; wrong-only subsets differ.',
                  'The existing freshness input has unresolved provenance and availability.'])
    targets[0].write_text(json.dumps(report,indent=2)+'\n')
    with targets[1].open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(cv_rows[0]));writer.writeheader();writer.writerows(cv_rows);writer.writerows(vrows)
    render(report,targets[2])


def render(r,target):
    esc=html.escape
    parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Resumed model optimization</title><style>body{font:15px/1.5 system-ui;background:#f6f8fb;color:#18212b}main{max-width:1400px;margin:auto;padding:24px}table{border-collapse:collapse;background:white;width:100%;margin:14px 0}th,td{padding:7px;border:1px solid #dbe2eb;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}</style><main><h1>Bounded RF and MLP optimization</h1><p>Grouped training CV chooses a predeclared short list; validation is development data. No test rows or test results are used. Three seeds reuse the same observations.</p>''']
    for title,summary in [('Grouped training CV',r['cv_summary']),('Validation development comparison',r['validation_summary'])]:
        parts.append('<h2>'+title+'</h2><div class="scroll"><table><tr><th>Configuration</th><th>Accuracy</th><th>Macro recall</th><th>Macro F1</th><th>Top 3</th><th>Log loss</th><th>Brier</th><th>All median / P90 m</th><th>Wrong median / P90 m</th></tr>')
        for s in summary:
            parts.append('<tr><td>'+esc(s['name'])+'</td>'+''.join(f'<td>{s[k]:.1%}</td>' for k in ['accuracy','macro_recall','macro_f1','top3_accuracy'])+f'<td>{s["log_loss"]:.3f}</td><td>{s["multiclass_brier"]:.3f}</td><td>{s["all_distance_median"]:.1f} / {s["all_distance_p90"]:.1f}</td><td>{s["wrong_distance_median"]:.1f} / {s["wrong_distance_p90"]:.1f}</td></tr>')
        parts.append('</table></div>')
    parts.append('<p>'+esc(r['shortlist_rule'])+'</p><h2>Validation recall by segment</h2><p>Mean over three seeds; support counts each row once.</p><div class="scroll"><table><tr><th>Segment</th><th>Rows</th>'+''.join('<th>'+esc(s['name'])+'</th>' for s in r['validation_summary'])+'</tr>')
    for i,label in sorted(enumerate(r['labels']),key=lambda x:r['display_names'][x[1]]):
        parts.append(f'<tr><td>{esc(r["display_names"][label])}</td><td>{r["validation_support"][i]}</td>')
        for s in r['validation_summary']:
            v=np.mean([x['metrics']['per_segment_recall'][i] for x in r['validation_runs'] if x['name']==s['name']]);parts.append(f'<td>{v:.1%}</td>')
        parts.append('</tr>')
    parts.append('</table></div><h2>Limits</h2><ul>'+''.join('<li>'+esc(x)+'</li>' for x in r['limitations'])+'</ul><p>Full seed, fold, pass and stopping records are in <a href="resumed_optimization.json">JSON</a>; predictions and probability vectors are in <a href="resumed_optimization_predictions.csv">CSV</a>.</p></main></html>')
    target.write_text(''.join(parts))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
