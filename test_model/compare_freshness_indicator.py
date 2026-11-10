#!/usr/bin/env python3
"""Compare a freshness-missing indicator on grouped training and validation."""
import csv
import hashlib
import html
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from compare_models import FEATURES, OUT, SEEDS, model, read
from optimize_models import evaluate, feature_matrix, geography, load_split

BASES=[('RF leaf=2 features=sqrt weight=none',2,None),('RF leaf=10 features=sqrt weight=none',10,None),
       ('RF leaf=5 features=sqrt weight=balanced_subsample',5,'balanced_subsample')]


def add_indicator(X):
    return np.column_stack((X,np.isnan(X[:,FEATURES.index('message_freshness_ms')]).astype(float)))


def run():
    targets=[OUT/n for n in ('freshness_indicator.json','freshness_indicator_predictions.csv','freshness_indicator.html')]
    if any(p.exists() for p in targets):raise RuntimeError('Indicator outputs already exist; refusing to overwrite')
    previous=json.loads((OUT/'resumed_optimization.json').read_text())
    train=load_split('train');val=load_split('validation')
    assert len(train)==1613 and len(val)==514 and previous['seeds']==SEEDS and previous['features']==FEATURES
    assert hashlib.sha256(json.dumps(train,sort_keys=True).encode()).hexdigest()==previous['training_rows_sha256']
    assert hashlib.sha256(json.dumps(val,sort_keys=True).encode()).hexdigest()==previous['validation_rows_sha256']
    assert previous['versions']['sklearn']==sklearn.__version__
    ids=[r['timestamp_unix_ms'] for r in train];vid=[r['timestamp_unix_ms'] for r in val]
    assert len(ids)==len(set(ids)) and len(vid)==len(set(vid)) and not set(ids)&set(vid)
    X=feature_matrix(train);V=feature_matrix(val);XI=add_indicator(X);VI=add_indicator(V)
    assert XI.shape[1]==11 and XI[:,-1].sum()==98 and VI[:,-1].sum()==143
    y=np.array([r['segment_id'] for r in train]);vy=np.array([r['segment_id'] for r in val]);labels=np.unique(y)
    folds=np.array([int(r['cv_fold']) for r in train]);passes=np.array([int(r['pass_index']) for r in train]);vp=np.array([int(r['pass_index']) for r in val])
    assert set(folds)==set(range(1,6)) and set(passes)==set(range(1,11)) and set(vp)==set(range(11,14))
    assert set(vy)==set(labels) and labels.tolist()==previous['labels']
    relevant=set(ids)|set(vid)
    points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in relevant}
    roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    geo=geography(train,labels,ids,roads,points);vgeo=geography(val,labels,ids,roads,points)
    old={}
    with (OUT/'resumed_optimization_predictions.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            if r['model'] in {b[0] for b in BASES}:old[r['split'],r['model'],int(r['seed']),r['timestamp_unix_ms']]=r
    assert set(old)=={(split,b[0],seed,t) for split,keys in [('training_cv',ids),('validation_development',vid)] for b in BASES for seed in SEEDS for t in keys}
    runs=[];outputs=[];pairs=[]
    for base,leaf,cw in BASES:
        for split,rows,Z,truth,groups,distances in [('training_cv',train,XI,y,passes,geo),('validation_development',val,VI,vy,vp,vgeo)]:
            keys=[r['timestamp_unix_ms'] for r in rows]
            for seed in SEEDS:
                p=np.empty((len(rows),len(labels)));seen=np.zeros(len(rows),int)
                if split=='training_cv':
                    for f in range(1,6):
                        fit=folds!=f;held=~fit;assert not set(passes[fit])&set(passes[held])
                        pipe=model('RF',seed);pipe[-1].set_params(min_samples_leaf=leaf,class_weight=cw)
                        pipe.fit(XI[fit],y[fit]);np.testing.assert_allclose(pipe[0].statistics_,np.nanmedian(XI[fit],axis=0))
                        p[held]=pipe.predict_proba(XI[held]);seen[held]+=1
                else:
                    pipe=model('RF',seed);pipe[-1].set_params(min_samples_leaf=leaf,class_weight=cw)
                    pipe.fit(XI,y);np.testing.assert_allclose(pipe[0].statistics_,np.nanmedian(XI,axis=0))
                    p[:]=pipe.predict_proba(VI);seen+=1
                assert np.all(seen==1) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(axis=1),1)
                pred=labels[p.argmax(axis=1)];m,d=evaluate(truth,p,labels,distances)
                basep=np.array([json.loads(old[split,base,seed,t]['probabilities']) for t in keys])
                bm,bd=evaluate(truth,basep,labels,distances);bp=labels[basep.argmax(axis=1)]
                for i,t in enumerate(keys):
                    prior=old[split,base,seed,t]
                    assert prior['segment_id']==truth[i] and prior['predicted_segment_id']==bp[i] and int(prior['pass_index'])==groups[i]
                    if split=='training_cv':assert int(prior['cv_fold'])==folds[i]
                original=next(v for v in previous['cv_runs' if split=='training_cv' else 'validation_runs'] if v['name']==base and v['seed']==seed)
                for k in ['accuracy','macro_recall','macro_f1','top3_accuracy','log_loss','multiclass_brier']:assert np.isclose(bm[k],original['metrics'][k])
                subgroup=[]
                for missing in (False,True):
                    ix=Z[:,-1]==float(missing)
                    subgroup.append(dict(missing=missing,rows=int(ix.sum()),baseline_correct=int((bp[ix]==truth[ix]).sum()),indicator_correct=int((pred[ix]==truth[ix]).sum())))
                pair=dict(split=split,base=base,seed=seed,both_correct=int(((bp==truth)&(pred==truth)).sum()),
                          only_indicator_correct=int(((bp!=truth)&(pred==truth)).sum()),
                          only_baseline_correct=int(((bp==truth)&(pred!=truth)).sum()),both_wrong=int(((bp!=truth)&(pred!=truth)).sum()),subgroups=subgroup)
                assert sum(pair[k] for k in ['both_correct','only_indicator_correct','only_baseline_correct','both_wrong'])==len(rows)
                pairs.append(pair)
                grouped=folds if split=='training_cv' else groups
                details=[dict(group=int(g),rows=int((grouped==g).sum()),metrics=evaluate(truth[grouped==g],p[grouped==g],labels,distances[grouped==g])[0]) for g in sorted(set(grouped))]
                runs.extend([dict(split=split,base=base,variant='baseline',seed=seed,metrics=bm),dict(split=split,base=base,variant='indicator',seed=seed,metrics=m,groups=details)])
                for i,r in enumerate(rows):
                    outputs.append(dict(split=split,base=base,seed=seed,timestamp_unix_ms=keys[i],pass_index=int(groups[i]),
                                        cv_fold=int(folds[i]) if split=='training_cv' else '',segment_id=truth[i],predicted_segment_id=pred[i],
                                        freshness_missing=bool(Z[i,-1]),probabilities=json.dumps(p[i].tolist()),distance_m=float(d[i])))
                print(f'{split} {base} seed={seed}: baseline recall={bm["macro_recall"]:.3f}, indicator recall={m["macro_recall"]:.3f}',flush=True)
    keys=['accuracy','macro_recall','macro_f1','top3_accuracy','log_loss','multiclass_brier','all_distance_median','all_distance_p90','wrong_distance_median','wrong_distance_p90']
    summary=[dict(split=split,base=base,variant=variant,**{k:float(np.mean([r['metrics'][k] for r in runs if r['split']==split and r['base']==base and r['variant']==variant])) for k in keys})
             for split in ('training_cv','validation_development') for base,_,_ in BASES for variant in ('baseline','indicator')]
    report=dict(protocol='One missingness indicator; paired training CV and validation development, no test.',features=FEATURES+['freshness_missing_indicator'],
                labels=labels.tolist(),display_names=previous['display_names'],training_support=previous['training_support'],validation_support=previous['validation_support'],
                training_rows_sha256=previous['training_rows_sha256'],validation_rows_sha256=previous['validation_rows_sha256'],
                versions={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                previous_results_sha256=hashlib.sha256((OUT/'resumed_optimization_predictions.csv').read_bytes()).hexdigest(),
                missing_counts={'train':98,'validation':143},runs=runs,pairs=pairs,summary=summary,
                limitations=['Validation motivated the follow-up and remains development data, not independent evaluation.',
                  'The prior test split and results were excluded; repeated CV/validation comparison can overfit development.',
                  'Freshness derivation, decision-time availability and manipulation resistance remain unresolved.',
                  'Adding an input changes feature sampling despite the nominal sqrt candidate count staying at three.',
                  'One vehicle and capture; seeds reuse rows, not independent drives.',
                  'Wrong-only distance populations differ; distances are point to predicted centerline, not network distance.'])
    targets[0].write_text(json.dumps(report,indent=2)+'\n')
    with targets[1].open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(outputs[0]));w.writeheader();w.writerows(outputs)
    render(report,targets[2])


def render(r,target):
    esc=html.escape
    parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Freshness indicator</title><style>body{font:15px/1.5 system-ui;background:#f6f8fb;color:#18212b}main{max-width:1450px;margin:auto;padding:24px}table{border-collapse:collapse;background:white;width:100%;margin:14px 0}th,td{padding:7px;border:1px solid #dbe2eb;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}</style><main><h1>Freshness missingness indicator</h1><p>Training grouped CV and validation development; identical folds and seeds. The indicator is added to three existing forests. No test rows or test results are used.</p><div class="scroll"><table><tr><th>Split / forest / variant</th><th>Accuracy</th><th>Macro recall</th><th>Macro F1</th><th>Top 3</th><th>Log loss</th><th>Brier</th><th>All median / P90 m</th><th>Wrong median / P90 m</th></tr>''']
    for s in r['summary']:
        label=f"{s['split']} / {s['base']} / {s['variant']}"
        parts.append('<tr><td>'+esc(label)+'</td>'+''.join(f'<td>{s[k]:.1%}</td>' for k in ['accuracy','macro_recall','macro_f1','top3_accuracy'])+f'<td>{s["log_loss"]:.3f}</td><td>{s["multiclass_brier"]:.3f}</td><td>{s["all_distance_median"]:.1f} / {s["all_distance_p90"]:.1f}</td><td>{s["wrong_distance_median"]:.1f} / {s["wrong_distance_p90"]:.1f}</td></tr>')
    parts.append('</table></div><h2>Paired correctness and missingness</h2><p>Each row compares the same observations and seed. Missing/present changes are conditional on different subsets. Seeds reuse observations.</p><div class="scroll"><table><tr><th>Split / forest / seed</th><th>Both correct</th><th>Only indicator correct</th><th>Only baseline correct</th><th>Both wrong</th><th>Missing: baseline → indicator / rows</th><th>Present: baseline → indicator / rows</th></tr>')
    for p in r['pairs']:
        cells=[]
        for missing in (True,False):
            v=next(x for x in p['subgroups'] if x['missing']==missing);cells.append(f'{v["baseline_correct"]} → {v["indicator_correct"]} / {v["rows"]}')
        parts.append(f'<tr><td>{esc(p["split"])} / {esc(p["base"])} / {p["seed"]}</td>'+''.join(f'<td>{p[k]}</td>' for k in ['both_correct','only_indicator_correct','only_baseline_correct','both_wrong'])+''.join(f'<td>{v}</td>' for v in cells)+'</tr>')
    parts.append('</table></div><h2>Validation recall by segment</h2><p>Mean across seeds; support counts observations once. Small segments have unstable rates.</p><div class="scroll"><table><tr><th>Segment</th><th>Rows</th>'+''.join(f'<th>{esc(b[0])} / {v}</th>' for b in BASES for v in ('baseline','indicator'))+'</tr>')
    for i,label in sorted(enumerate(r['labels']),key=lambda x:r['display_names'][x[1]]):
        parts.append(f'<tr><td>{esc(r["display_names"][label])}</td><td>{r["validation_support"][i]}</td>')
        for base,_,_ in BASES:
            for variant in ('baseline','indicator'):
                v=np.mean([x['metrics']['per_segment_recall'][i] for x in r['runs'] if x['split']=='validation_development' and x['base']==base and x['variant']==variant]);parts.append(f'<td>{v:.1%}</td>')
        parts.append('</tr>')
    parts.append('</table></div><h2>Limits</h2><ul>'+''.join('<li>'+esc(x)+'</li>' for x in r['limitations'])+'</ul><p>Full seed and grouped results: <a href="freshness_indicator.json">JSON</a>. Indicator predictions: <a href="freshness_indicator_predictions.csv">CSV</a>. Baseline predictions remain in resumed_optimization_predictions.csv.</p></main></html>')
    target.write_text(''.join(parts))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
