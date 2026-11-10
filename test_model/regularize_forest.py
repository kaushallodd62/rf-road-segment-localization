#!/usr/bin/env python3
"""Controlled min_samples_leaf sweep; training grouped CV only.

Primary comparison: macro recall; also report accuracy, fitting gap, and geography.
This is development-set tuning, not a final generalization estimate.
"""
import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import sklearn
from sklearn.metrics import recall_score
from threadpoolctl import threadpool_limits
from compare_models import OUT, SEEDS, RF, FEATURES, model, read, scores
from segment_road import local_xy, point_to_line

LEAVES = [2, 5, 10, 20]


def run():
    rows=[r for r in read('modeling_dataset.csv') if r['split']=='train']
    baseline=json.loads((OUT/'model_comparison.json').read_text())
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    assert digest==baseline['training_rows_sha256'], 'Training data changed'
    assert FEATURES==baseline['features'] and RF==baseline['settings']['RF']
    assert sklearn.__version__==baseline['versions']['sklearn']
    ids=[r['timestamp_unix_ms'] for r in rows]
    assert len(set(ids))==len(ids)
    X=np.array([[float(r[f]) if r[f] else np.nan for f in FEATURES] for r in rows])
    y=np.array([r['segment_id'] for r in rows]);labels=np.unique(y)
    folds=np.array([int(r['cv_fold']) for r in rows]);passes=np.array([r['pass_index'] for r in rows])
    assert set(folds)==set(range(1,6)) and not np.isinf(X).any()
    previous={(int(r['seed']),r['timestamp_unix_ms']):r['predicted_segment_id'] for r in read('model_comparison_predictions.csv') if r['model']=='RF'}
    points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv')}
    roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    lon0=np.mean([float(points[t]['longitude']) for t in ids]);lat0=np.mean([float(points[t]['latitude']) for t in ids])
    geo=np.empty((len(rows),len(labels)))
    for i,t in enumerate(ids):
        assert points[t]['matched_segment_id']==y[i]
        xy=local_xy(float(points[t]['longitude']),float(points[t]['latitude']),lon0,lat0)
        for j,label in enumerate(labels):geo[i,j]=point_to_line(xy,roads[label],lon0,lat0)[0]
    assert np.isfinite(geo).all()
    runs=[];predictions=[]
    for leaf in LEAVES:
        for seed in SEEDS:
            pred=np.empty_like(y);seen=np.zeros(len(rows),int);details=[]
            for fold in np.unique(folds):
                fit=folds!=fold;held=~fit
                assert not set(passes[fit])&set(passes[held])
                assert set(y[fit])==set(labels) and not np.isnan(X[fit]).all(axis=0).any()
                pipe=model('RF',seed)
                pipe[-1].set_params(min_samples_leaf=leaf)
                pipe.fit(X[fit],y[fit])
                np.testing.assert_allclose(pipe[0].statistics_,np.nanmedian(X[fit],axis=0))
                pred[held]=pipe.predict(X[held]);seen[held]+=1
                details.append(dict(fold=int(fold),fit_rows=int(fit.sum()),held_rows=int(held.sum()),
                                    train=scores(y[fit],pipe.predict(X[fit]),labels),held=scores(y[held],pred[held],labels)))
            assert np.all(seen==1)
            if leaf==2:
                assert all(pred[i]==previous[seed,t] for i,t in enumerate(ids)), 'Baseline predictions changed'
            m=scores(y,pred,labels)
            distances=geo[np.arange(len(rows)),np.searchsorted(labels,pred)]
            wrong=pred!=y
            m.update(train_accuracy=float(np.mean([f['train']['accuracy'] for f in details])),
                     train_macro_recall=float(np.mean([f['train']['macro_recall'] for f in details])),
                     all_distance_p90=float(np.quantile(distances,.9,method='inverted_cdf')),
                     all_distance_median=float(np.median(distances)),
                     wrong_distance_median=float(np.median(distances[wrong])) if wrong.any() else None,
                     wrong_distance_p90=float(np.quantile(distances[wrong],.9,method='inverted_cdf')) if wrong.any() else None,
                     per_segment_recall=recall_score(y,pred,labels=labels,average=None,zero_division=0).tolist())
            m['accuracy_gap']=m['train_accuracy']-m['accuracy']
            runs.append(dict(leaf=leaf,seed=seed,metrics=m,folds=details))
            for i,r in enumerate(rows):
                predictions.append(dict(leaf=leaf,seed=seed,timestamp_unix_ms=ids[i],cv_fold=int(folds[i]),segment_id=y[i],predicted_segment_id=pred[i],distance_m=float(distances[i])))
            print(f'leaf={leaf:2} seed={seed:3}: fitting={m["train_accuracy"]:.3f} held={m["accuracy"]:.3f} macro_recall={m["macro_recall"]:.3f}',flush=True)
    summary=[]
    for leaf in LEAVES:
        rr=[r for r in runs if r['leaf']==leaf]
        entry={'leaf':leaf}
        for k in ['accuracy','macro_recall','macro_f1','train_accuracy','accuracy_gap','all_distance_p90','wrong_distance_median','wrong_distance_p90']:
            entry[k]=float(np.mean([r['metrics'][k] for r in rr]))
        entry['accuracy_seed_range']=[min(r['metrics']['accuracy'] for r in rr),max(r['metrics']['accuracy'] for r in rr)]
        entry['fold_accuracy']=[float(np.mean([r['folds'][f]['held']['accuracy'] for r in rr])) for f in range(5)]
        entry['fold_macro_recall']=[float(np.mean([r['folds'][f]['held']['macro_recall'] for r in rr])) for f in range(5)]
        summary.append(entry)
    report=dict(protocol='Four predeclared leaf sizes; five grouped training folds; three seeds. Only leaf size changes. Validation/test unscored. Macro recall primary; accuracy/geography guardrails.',
                features=FEATURES,seeds=SEEDS,leaf_sizes=LEAVES,forest_settings=RF,training_rows_sha256=digest,
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),versions={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
                labels=labels.tolist(),display_names=baseline['display_names'],support=[int(sum(y==l)) for l in labels],runs=runs,summary=summary,
                checks='Unique training-only coverage, disjoint pass groups, fold-local imputation, finite geography; leaf=2 exactly reproduces all saved RF predictions for all seeds.')
    (OUT/'forest_regularization.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'forest_regularization_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
    render(report)
    print(json.dumps(summary,indent=2))


def render(r):
    parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Forest regularization</title><style>body{font:15px/1.5 system-ui;color:#18212b;background:#f6f8fb}main{max-width:1100px;margin:auto;padding:24px}table{border-collapse:collapse;background:white;width:100%;margin:14px 0}th,td{padding:8px;border:1px solid #dbe2eb;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}svg{max-width:800px;width:100%;background:white}small{color:#526174}</style><main><h1>Does stronger forest regularization help?</h1><p>Only minimum leaf size changes: 2, 5, 10, 20. Same ten features, 300 trees, five grouped training folds and seeds 17/42/123. No feature experiment, validation scoring, or test scoring.</p><h2>Overall results</h2><p>Mean of three seed-specific pooled CV scores. Fitting accuracy is the mean across fitting folds. These seed repetitions are not independent drives. Gap = fitting accuracy minus held-out accuracy.</p><div class="scroll"><table><tr><th>Leaf size</th><th>Fitting accuracy</th><th>Held accuracy</th><th>Macro recall</th><th>Macro F1</th><th>Gap (pp)</th><th>All-row P90 distance (m)</th></tr>''']
    for s in r['summary']:
        parts.append(f'<tr><td>{s["leaf"]}</td>'+''.join(f'<td>{s[k]:.1%}</td>' for k in ['train_accuracy','accuracy','macro_recall','macro_f1'])+f'<td>{s["accuracy_gap"]*100:.1f}</td><td>{s["all_distance_p90"]:.1f}</td></tr>')
    parts.append('</table></div><h2>Fitting versus held-out accuracy</h2>')
    parts.append('<svg viewBox="0 0 800 300" role="img" aria-label="Fitting and held-out accuracy versus minimum leaf size"><title>Regularization and accuracy</title>')
    for i in range(6):
        yy=250-i*44;parts.append(f'<line x1="65" x2="760" y1="{yy}" y2="{yy}" stroke="#dbe2eb"/><text x="55" y="{yy+5}" text-anchor="end">{i*20}%</text>')
    for k,col in [('train_accuracy','#2472b8'),('accuracy','#16845b')]:
        points=' '.join(f'{100+i*200},{250-s[k]*220}' for i,s in enumerate(r['summary']))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{col}" stroke-width="3"/>')
    for i,s in enumerate(r['summary']):parts.append(f'<text x="{100+i*200}" y="275" text-anchor="middle">{s["leaf"]}</text>')
    parts.append('<text x="240" y="296">Minimum leaf size (categorical spacing)</text></svg><p>Blue: fitting accuracy. Green: held-out accuracy. A smaller gap alone is not an improvement.</p><h2>Paired fold results</h2><p>Each cell is accuracy / macro recall, averaged over three seeds for the same held-out passes.</p><table><tr><th>Leaf</th>'+''.join(f'<th>Fold {f}</th>' for f in range(1,6))+'</tr>')
    for s in r['summary']:parts.append(f'<tr><td>{s["leaf"]}</td>'+''.join(f'<td>{a:.1%} / {b:.1%}</td>' for a,b in zip(s['fold_accuracy'],s['fold_macro_recall']))+'</tr>')
    parts.append('</table><h2>Per-segment recall</h2><table><tr><th>Segment</th><th>Support</th>'+''.join(f'<th>Leaf {l}</th>' for l in LEAVES)+'</tr>')
    for j,label in sorted(enumerate(r['labels']),key=lambda v:r['display_names'][v[1]]):
        vals=[np.mean([v['metrics']['per_segment_recall'][j] for v in r['runs'] if v['leaf']==leaf]) for leaf in LEAVES]
        parts.append(f'<tr><td>{r["display_names"][label]}</td><td>{r["support"][j]}</td>'+''.join(f'<td>{v:.1%}</td>' for v in vals)+'</tr>')
    parts.append('</table><h2>Individual seed results</h2><table><tr><th>Leaf / seed</th><th>Accuracy</th><th>Macro recall</th><th>Wrong median m</th><th>Wrong P90 m</th></tr>')
    for v in r['runs']:
        m=v['metrics'];parts.append(f'<tr><td>{v["leaf"]} / {v["seed"]}</td><td>{m["accuracy"]:.1%}</td><td>{m["macro_recall"]:.1%}</td><td>{m["wrong_distance_median"]:.1f}</td><td>{m["wrong_distance_p90"]:.1f}</td></tr>')
    parts.append('</table><p>Distances are shortest reported-coordinate-to-predicted-centerline distances, not network distance. Wrong-only statistics use different subsets as predictions change. P90 uses nearest rank; overall summaries average seed-specific P90s.</p><p>Checks: '+r['checks']+'</p><p><small>This is development-set tuning from one capture. No significance claim or final test estimate. Source: regularize_forest.py; full results and row predictions are in forest_regularization.json and forest_regularization_predictions.csv.</small></p></main></html>')
    (OUT/'forest_regularization.html').write_text(''.join(parts))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
