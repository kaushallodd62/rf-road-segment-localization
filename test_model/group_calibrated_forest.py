#!/usr/bin/env python3
"""Nested pass-grouped sigmoid calibration, training and validation only."""
import csv
import hashlib
import html
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from threadpoolctl import threadpool_limits

from compare_models import FEATURES, OUT, SEEDS, model, read
from inspect_forest_leaf_scores import score_metrics
from optimize_models import evaluate, feature_matrix, geography, load_split

BASES=[('RF leaf=2 features=sqrt weight=none',2),('RF leaf=10 features=sqrt weight=none',10)]


def grouped_splits(y,folds,passes,labels):
    splits=[];metadata=[]
    for f in sorted(set(folds)):
        fit=np.flatnonzero(folds!=f);held=np.flatnonzero(folds==f)
        assert len(fit)>0 and len(held)>0 and not set(passes[fit])&set(passes[held])
        assert set(y[fit])==set(labels)
        splits.append((fit,held))
        metadata.append(dict(inner_fold=int(f),fit_passes=sorted(set(passes[fit].tolist())),held_passes=sorted(set(passes[held].tolist())),
                             fit_rows=len(fit),held_rows=len(held)))
    assert np.array_equal(np.sort(np.concatenate([v[1] for v in splits])),np.arange(len(y)))
    return splits,metadata


def fit_calibrated(X,y,folds,passes,labels,leaf,seed):
    assert not np.isnan(X).all(axis=0).any() and set(y)==set(labels)
    splits,metadata=grouped_splits(y,folds,passes,labels)
    base=model('RF',seed);base[-1].set_params(min_samples_leaf=leaf)
    pipe=CalibratedClassifierCV(estimator=base,method='sigmoid',cv=splits,ensemble=False,n_jobs=1)
    pipe.fit(X,y)
    assert np.array_equal(pipe.classes_,labels)
    # ensemble=False retains a base estimator refitted on the full fitting portion.
    final=pipe.calibrated_classifiers_[0].estimator
    np.testing.assert_allclose(final[0].statistics_,np.nanmedian(X,axis=0))
    return pipe,metadata


def run():
    targets=[OUT/n for n in ('group_calibrated_forest.json','group_calibrated_forest_predictions.csv','group_calibrated_forest.html')]
    if any(p.exists() for p in targets):raise RuntimeError('Calibration outputs already exist; refusing to overwrite')
    previous=json.loads((OUT/'resumed_optimization.json').read_text());train=load_split('train');val=load_split('validation')
    assert len(train)==1613 and len(val)==514 and previous['seeds']==SEEDS and previous['features']==FEATURES
    assert hashlib.sha256(json.dumps(train,sort_keys=True).encode()).hexdigest()==previous['training_rows_sha256']
    assert hashlib.sha256(json.dumps(val,sort_keys=True).encode()).hexdigest()==previous['validation_rows_sha256']
    assert previous['versions']['sklearn']==sklearn.__version__
    ids=[r['timestamp_unix_ms'] for r in train];vid=[r['timestamp_unix_ms'] for r in val]
    assert len(set(ids))==len(ids) and len(set(vid))==len(vid) and not set(ids)&set(vid)
    X=feature_matrix(train);V=feature_matrix(val);y=np.array([r['segment_id'] for r in train]);vy=np.array([r['segment_id'] for r in val]);labels=np.unique(y)
    folds=np.array([int(r['cv_fold']) for r in train]);passes=np.array([int(r['pass_index']) for r in train]);vp=np.array([int(r['pass_index']) for r in val])
    assert labels.tolist()==previous['labels'] and set(vy)==set(labels) and set(folds)==set(range(1,6))
    assert set(passes)==set(range(1,11)) and set(vp)==set(range(11,14)) and not set(passes)&set(vp)
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
    for base,leaf in BASES:
        for split,rows,Z,truth,groups,distances in [('training_cv',train,X,y,passes,geo),('validation_development',val,V,vy,vp,vgeo)]:
            keys=[r['timestamp_unix_ms'] for r in rows]
            for seed in SEEDS:
                prob=np.empty((len(rows),len(labels)));seen=np.zeros(len(rows),int);details=[]
                if split=='training_cv':
                    for f in range(1,6):
                        fit=folds!=f;held=~fit;assert not set(passes[fit])&set(passes[held])
                        pipe,inner=fit_calibrated(X[fit],y[fit],folds[fit],passes[fit],labels,leaf,seed)
                        prob[held]=pipe.predict_proba(X[held]);seen[held]+=1
                        details.append(dict(fold=f,outer_fit_passes=sorted(set(passes[fit].tolist())),outer_held_passes=sorted(set(passes[held].tolist())),inner_splits=inner))
                else:
                    pipe,inner=fit_calibrated(X,y,folds,passes,labels,leaf,seed)
                    prob[:]=pipe.predict_proba(V);seen+=1
                    details.append(dict(outer_fit_passes=sorted(set(passes.tolist())),outer_held_passes=sorted(set(vp.tolist())),inner_splits=inner))
                assert np.all(seen==1) and np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(axis=1),1)
                pred=labels[prob.argmax(axis=1)];m,d=evaluate(truth,prob,labels,distances);_,bins,_,_=score_metrics(truth,prob,labels)
                basep=np.array([json.loads(old[split,base,seed,t]['probabilities']) for t in keys])
                bm,bd=evaluate(truth,basep,labels,distances);_,bbins,_,_=score_metrics(truth,basep,labels);bp=labels[basep.argmax(axis=1)]
                for i,t in enumerate(keys):
                    prior=old[split,base,seed,t]
                    assert prior['segment_id']==truth[i] and prior['predicted_segment_id']==bp[i] and int(prior['pass_index'])==groups[i]
                    if split=='training_cv':assert int(prior['cv_fold'])==folds[i]
                prior=next(v for v in previous['cv_runs' if split=='training_cv' else 'validation_runs'] if v['name']==base and v['seed']==seed)
                for k in ['accuracy','macro_recall','macro_f1','top3_accuracy','log_loss','multiclass_brier']:assert np.isclose(bm[k],prior['metrics'][k])
                pair=dict(split=split,base=base,seed=seed,both_correct=int(((bp==truth)&(pred==truth)).sum()),
                          only_calibrated_correct=int(((bp!=truth)&(pred==truth)).sum()),
                          only_baseline_correct=int(((bp==truth)&(pred!=truth)).sum()),both_wrong=int(((bp!=truth)&(pred!=truth)).sum()))
                assert sum(pair[k] for k in ['both_correct','only_calibrated_correct','only_baseline_correct','both_wrong'])==len(rows)
                pairs.append(pair)
                grouped=folds if split=='training_cv' else groups
                details_metrics=[dict(group=int(g),rows=int((grouped==g).sum()),metrics=evaluate(truth[grouped==g],prob[grouped==g],labels,distances[grouped==g])[0]) for g in sorted(set(grouped))]
                runs.extend([dict(split=split,base=base,variant='baseline',seed=seed,metrics=bm,confidence_bins=bbins),
                             dict(split=split,base=base,variant='calibrated',seed=seed,metrics=m,confidence_bins=bins,groups=details_metrics,training=details)])
                for i,r in enumerate(rows):
                    outputs.append(dict(split=split,base=base,seed=seed,timestamp_unix_ms=keys[i],pass_index=int(groups[i]),
                                        cv_fold=int(folds[i]) if split=='training_cv' else '',segment_id=truth[i],predicted_segment_id=pred[i],
                                        probabilities=json.dumps(prob[i].tolist()),distance_m=float(d[i])))
                print(f'{split} {base} seed={seed}: baseline loss={bm["log_loss"]:.3f}, calibrated loss={m["log_loss"]:.3f}, recall={m["macro_recall"]:.3f}',flush=True)
    keys=['accuracy','macro_recall','macro_f1','top3_accuracy','log_loss','multiclass_brier','true_score_zero',
          'all_distance_median','all_distance_p90','wrong_distance_median','wrong_distance_p90']
    summary=[dict(split=split,base=base,variant=variant,**{k:float(np.mean([r['metrics'][k] for r in runs if r['split']==split and r['base']==base and r['variant']==variant])) for k in keys})
             for split in ('training_cv','validation_development') for base,_ in BASES for variant in ('baseline','calibrated')]
    report=dict(protocol='Pass-grouped inner sigmoid calibration, compared in grouped training CV and validation development; test excluded.',
                features=FEATURES,labels=labels.tolist(),display_names=previous['display_names'],training_support=previous['training_support'],validation_support=previous['validation_support'],
                training_rows_sha256=previous['training_rows_sha256'],validation_rows_sha256=previous['validation_rows_sha256'],seeds=SEEDS,
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),versions={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
                previous_predictions_sha256=hashlib.sha256((OUT/'resumed_optimization_predictions.csv').read_bytes()).hexdigest(),runs=runs,pairs=pairs,summary=summary,
                limitations=['Validation is development data and was used to motivate this experiment.',
                   'Test rows and results are excluded. Repeated comparisons can overfit CV and validation.',
                   'One vehicle and capture; seeds reuse observations, not independent drives.',
                   'Sigmoid calibration and top-label reliability bins do not prove classwise calibration or calibrated authenticity.',
                   'Distances are point to predicted centerline, not network distance; wrong-only subsets differ.',
                   'Existing freshness input provenance and decision-time availability remain unresolved.'])
    targets[0].write_text(json.dumps(report,indent=2)+'\n')
    with targets[1].open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(outputs[0]));w.writeheader();w.writerows(outputs)
    render(report,targets[2])


def render(r,target):
    esc=html.escape
    parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Grouped forest calibration</title><style>body{font:15px/1.5 system-ui;background:#f6f8fb;color:#18212b}main{max-width:1450px;margin:auto;padding:24px}table{border-collapse:collapse;background:white;width:100%;margin:14px 0}th,td{padding:7px;border:1px solid #dbe2eb;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}</style><main><h1>Group-aware sigmoid calibration</h1><p>Inner pass-group out-of-fold scores train the calibrator. The base forest is refitted on the whole fitting portion. Training grouped CV and validation are development data. Test rows and results are excluded.</p><div class="scroll"><table><tr><th>Split / forest / variant</th><th>Accuracy</th><th>Macro recall</th><th>Macro F1</th><th>Top 3</th><th>Log loss</th><th>Brier</th><th>True score zero</th><th>All median / P90 m</th><th>Wrong median / P90 m</th></tr>''']
    for s in r['summary']:
        label=f"{s['split']} / {s['base']} / {s['variant']}"
        parts.append('<tr><td>'+esc(label)+'</td>'+''.join(f'<td>{s[k]:.1%}</td>' for k in ['accuracy','macro_recall','macro_f1','top3_accuracy'])+f'<td>{s["log_loss"]:.3f}</td><td>{s["multiclass_brier"]:.3f}</td><td>{s["true_score_zero"]:.1f}</td><td>{s["all_distance_median"]:.1f} / {s["all_distance_p90"]:.1f}</td><td>{s["wrong_distance_median"]:.1f} / {s["wrong_distance_p90"]:.1f}</td></tr>')
    parts.append('</table></div><p>Cells average seed-specific results on the same rows. Median and P90 use actual point to predicted centerline. P90 uses nearest rank; wrong-only populations differ.</p><h2>Validation recall by segment</h2><div class="scroll"><table><tr><th>Segment</th><th>Rows</th>'+''.join(f'<th>{esc(b[0])} / {v}</th>' for b in BASES for v in ('baseline','calibrated'))+'</tr>')
    for i,label in sorted(enumerate(r['labels']),key=lambda x:r['display_names'][x[1]]):
        parts.append(f'<tr><td>{esc(r["display_names"][label])}</td><td>{r["validation_support"][i]}</td>')
        for base,_ in BASES:
            for variant in ('baseline','calibrated'):
                v=np.mean([x['metrics']['per_segment_recall'][i] for x in r['runs'] if x['split']=='validation_development' and x['base']==base and x['variant']==variant]);parts.append(f'<td>{v:.1%}</td>')
        parts.append('</tr>')
    parts.append('</table></div><h2>Top-label reliability</h2><p>Each row compares average winning score to observed top-one accuracy. The bins and groups may be small and correlated; this is not classwise calibration.</p><div class="scroll"><table><tr><th>Split / forest / variant / seed</th><th>Bin</th><th>Rows</th><th>Mean score</th><th>Accuracy</th></tr>')
    for run in r['runs']:
        for b in run['confidence_bins']:
            if b['count']:
                tag=f"{run['split']} / {run['base']} / {run['variant']} / {run['seed']}"
                parts.append(f'<tr><td>{esc(tag)}</td><td>{b["lower"]:.1f} to {b["upper"]:.1f}</td><td>{b["count"]}</td><td>{b["mean_confidence"]:.3f}</td><td>{b["accuracy"]:.3f}</td></tr>')
    parts.append('</table></div><h2>Limits</h2><ul>'+''.join('<li>'+esc(x)+'</li>' for x in r['limitations'])+'</ul><p>Detailed grouped splits, paired transitions and scores: <a href="group_calibrated_forest.json">JSON</a>. Calibrated predictions: <a href="group_calibrated_forest_predictions.csv">CSV</a>.</p></main></html>')
    target.write_text(''.join(parts))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
