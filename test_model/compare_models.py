#!/usr/bin/env python3
"""Fixed RF/MLP baselines: grouped training-only CV, three seeds, no tuning.

Run with test_model/.venv/bin/python test_model/compare_models.py.
MLP stopping uses fitting loss only (no random internal validation split).
"""
import csv
import hashlib
import html
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from evaluate_weighted_knn import FEATURES
from segment_road import local_xy, point_to_line

OUT = Path(__file__).resolve().parent / 'output'
SEEDS = [17, 42, 123]
RF = dict(n_estimators=300, min_samples_leaf=2, max_features='sqrt', n_jobs=1)
MLP = dict(hidden_layer_sizes=(64,32), activation='relu', solver='adam', alpha=.01,
           batch_size=64, learning_rate_init=.001, max_iter=3000,
           early_stopping=False, tol=1e-4, n_iter_no_change=30)


def read(name):
    with (OUT/name).open(newline='') as f:
        return list(csv.DictReader(f))


def model(kind, seed):
    if kind == 'RF':
        return make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(random_state=seed, **RF))
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), MLPClassifier(random_state=seed, **MLP))


def scores(y, pred, labels):
    return dict(accuracy=float(accuracy_score(y,pred)), macro_recall=float(recall_score(y,pred,labels=labels,average='macro',zero_division=0)),
                macro_f1=float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)))


def run():
    rows = [r for r in read('modeling_dataset.csv') if r['split']=='train']
    ids = [r['timestamp_unix_ms'] for r in rows]
    assert len(set(ids)) == len(rows)
    X = np.array([[float(r[f]) if r[f] else np.nan for f in FEATURES] for r in rows])
    assert not np.isinf(X).any()
    y = np.array([r['segment_id'] for r in rows]); labels = np.unique(y)
    folds = np.array([int(r['cv_fold']) for r in rows]); passes=np.array([int(r['pass_index']) for r in rows])
    assert set(folds)==set(range(1,6))
    for fold in np.unique(folds):
        assert not set(passes[folds==fold]) & set(passes[folds!=fold])
        assert set(y[folds!=fold])==set(labels)
    points = {r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv')}
    roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    ordered=sorted(roads,key=lambda s:(-np.mean([c[1] for c in roads[s]]),np.mean([c[0] for c in roads[s]])))
    names={s:f'S{i+1:02}' for i,s in enumerate(ordered)}
    lon0=np.mean([float(points[t]['longitude']) for t in ids]);lat0=np.mean([float(points[t]['latitude']) for t in ids])
    # Precompute geometric distances for scoring only; never model inputs.
    geo=np.empty((len(rows),len(labels)))
    for i,t in enumerate(ids):
        assert points[t]['matched_segment_id']==y[i]
        xy=local_xy(float(points[t]['longitude']),float(points[t]['latitude']),lon0,lat0)
        for j,label in enumerate(labels):geo[i,j]=point_to_line(xy,roads[label],lon0,lat0)[0]
    def evaluate(pred):
        m=scores(y,pred,labels)
        d=geo[np.arange(len(y)),np.searchsorted(labels,pred)];wrong=pred!=y
        m.update(wrong_distance_median=float(np.median(d[wrong])) if wrong.any() else None,
                 wrong_distance_p90=float(np.quantile(d[wrong],.9,method='inverted_cdf')) if wrong.any() else None,
                 all_distance_median=float(np.median(d)),all_distance_p90=float(np.quantile(d,.9,method='inverted_cdf')),
                 per_segment_recall=recall_score(y,pred,labels=labels,average=None,zero_division=0).tolist(),
                 confusion=confusion_matrix(y,pred,labels=labels).tolist())
        return m
    baseline={r['timestamp_unix_ms']:r for r in read('weighted_knn_cv_predictions.csv')}
    assert set(baseline)==set(ids)
    basepred=np.array([baseline[t]['predicted_segment_id'] for t in ids])
    base=evaluate(basepred)
    old=json.loads((OUT/'weighted_knn_cv_report.json').read_text())['overall']
    assert np.isclose(base['accuracy'],old['weighted_knn_accuracy'])
    assert np.isclose(base['macro_recall'],old['weighted_knn_macro_recall'])
    results=[];output_rows=[]
    for kind in ['RF','MLP']:
        for seed in SEEDS:
            pred=np.empty(len(y),dtype=y.dtype);probs=np.zeros((len(y),len(labels)));seen=np.zeros(len(y),int);details=[]
            for fold in np.unique(folds):
                fit=folds!=fold;held=~fit
                assert not np.isnan(X[fit]).all(axis=0).any()
                pipe=model(kind,seed);start=time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');pipe.fit(X[fit],y[fit])
                duration=time.perf_counter()-start
                assert np.array_equal(pipe.classes_,labels)
                assert np.allclose(pipe[0].statistics_,np.nanmedian(X[fit],axis=0))
                probs[held]=pipe.predict_proba(X[held]);pred[held]=pipe.predict(X[held]);seen[held]+=1
                assert np.isfinite(probs[held]).all() and np.allclose(probs[held].sum(axis=1),1)
                train_score=scores(y[fit],pipe.predict(X[fit]),labels)
                detail=dict(fold=int(fold),fit_rows=int(fit.sum()),held_rows=int(held.sum()),fit_seconds=duration,
                            train=train_score,held=scores(y[held],pred[held],labels),warnings=[str(w.message) for w in caught])
                if kind=='MLP':
                    detail.update(epochs=int(pipe[-1].n_iter_),loss_curve=[float(v) for v in pipe[-1].loss_curve_],hit_epoch_cap=pipe[-1].n_iter_>=MLP['max_iter'])
                details.append(detail)
                print(f'{kind} seed={seed} fold={fold}: train={train_score["accuracy"]:.3f} held={detail["held"]["accuracy"]:.3f} {duration:.1f}s',flush=True)
            assert np.all(seen==1)
            m=evaluate(pred)
            m.update(log_loss=float(log_loss(y,probs,labels=labels)),top3_accuracy=float(np.mean([np.searchsorted(labels,label) in ix for label,ix in zip(y,np.argsort(probs,axis=1)[:,-3:])])),
                     multiclass_brier=float(np.mean(np.sum((probs-(y[:,None]==labels[None,:]))**2,axis=1))))
            results.append(dict(model=kind,seed=seed,metrics=m,folds=details))
            for i,r in enumerate(rows):
                output_rows.append(dict(model=kind,seed=seed,timestamp_unix_ms=ids[i],cv_fold=int(folds[i]),pass_index=int(passes[i]),direction=r['direction'],segment_id=y[i],predicted_segment_id=pred[i],probabilities=json.dumps(probs[i].tolist())))
    # Tiny-set memorization diagnostic; not a held-out performance estimate.
    tiny=np.concatenate([np.where(y==label)[0][:2] for label in labels])
    checks={}
    for kind in ['RF','MLP']:
        p=model(kind,17)
        if kind=='RF':p[-1].set_params(min_samples_leaf=1)
        else:p[-1].set_params(solver='lbfgs',alpha=0,max_iter=3000,max_fun=30000)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always');p.fit(X[tiny],y[tiny])
        checks[kind]={'rows':len(tiny),'training_accuracy':float(p.score(X[tiny],y[tiny])),'warnings':[str(v.message) for v in w],
                      'scope':'Memorization only; less regularization, MLP uses L-BFGS; not CV configuration.'}
        assert checks[kind]['training_accuracy']>=.95,checks[kind]
    report=dict(protocol='Fixed configurations; 5 grouped training folds, 3 seeds. No tuning, validation/test scoring, or final deployment fit.',
                features=FEATURES,seeds=SEEDS,settings={'RF':RF,'MLP':MLP},labels=labels.tolist(),display_names=names,
                support=[int(sum(y==l)) for l in labels],knn=base,runs=results,memorization_checks=checks,
                versions={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
                training_rows_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limitations=['One capture and vehicle; seed spread is not uncertainty across independent drives.',
                             'No class weighting; settings are baselines, not an exhaustive search.',
                             'MLP uses median imputation then standard scaling; RF uses median imputation; kNN retains its original median/IQR scaling.',
                             'Probabilities are not calibrated; no spoof-detection threshold evaluated.',
                             'Geography is shortest point-to-predicted-centerline distance, not road-network distance.'])
    (OUT/'model_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'model_comparison_predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(output_rows[0]));writer.writeheader();writer.writerows(output_rows)
    render(report)
    for kind in ['RF','MLP']:
        rr=[r['metrics'] for r in results if r['model']==kind]
        print(kind, {k:round(float(np.mean([r[k] for r in rr])),4) for k in ['accuracy','macro_recall','macro_f1','wrong_distance_median','wrong_distance_p90']})


def render(r):
    esc=html.escape
    page=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RF and MLP comparison</title><style>body{font:15px/1.5 system-ui;background:#f5f7fb;color:#18212b;margin:0}main{max-width:1100px;margin:auto;padding:25px}table{border-collapse:collapse;background:white;width:100%;margin:15px 0}td,th{padding:8px;border:1px solid #d9e0ea;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}pre{white-space:pre-wrap}small{color:#526174}svg{max-width:800px;width:100%;background:white}</style><main><h1>Random forest and MLP: grouped CV comparison</h1><p>1,613 training rows, ten inputs, five grouped folds, three seeds per model. Validation and test remain unscored. Settings were fixed before the run; these are development results, not independent-drive estimates.</p><h2>Overall comparison</h2><p>One row per seed: repeated seeds use the same observations and are not independent test sets. Wrong-distance statistics include only classification mistakes; all-distance P90 includes every prediction.</p><div class="scroll"><table><tr><th>Model / seed</th><th>Accuracy</th><th>Macro recall</th><th>Macro F1</th><th>Wrong median m</th><th>Wrong P90 m</th><th>All P90 m</th></tr>''']
    for name,m in [('kNN / fixed',r['knn'])]+[(f"{v['model']} / {v['seed']}",v['metrics']) for v in r['runs']]:
        page.append(f'<tr><td>{name}</td>'+''.join(f'<td>{m[k]:.1%}</td>' for k in ['accuracy','macro_recall','macro_f1'])+''.join(f'<td>{m[k]:.1f}</td>' for k in ['wrong_distance_median','wrong_distance_p90','all_distance_p90'])+'</tr>')
    page.append('</table></div><h2>Probability scores</h2><p>Top-3 asks whether the true segment is among the three highest scores. Lower log loss and multiclass Brier score are better. These scores do not establish calibration or a usable spoof threshold. kNN lacks saved full probability vectors.</p><table><tr><th>Model / seed</th><th>Top-3 accuracy</th><th>Log loss</th><th>Brier (sum over classes)</th></tr>')
    for v in r['runs']:
        m=v['metrics']
        page.append(f'<tr><td>{v["model"]} / {v["seed"]}</td><td>{m["top3_accuracy"]:.1%}</td><td>{m["log_loss"]:.3f}</td><td>{m["multiclass_brier"]:.3f}</td></tr>')
    page.append('</table><h2>Per-segment recall</h2><p>RF/MLP values are arithmetic means over three seeds. Segment support is counted once.</p><table><tr><th>Segment</th><th>Support</th><th>kNN</th><th>RF</th><th>MLP</th></tr>')
    for j,l in sorted(enumerate(r['labels']),key=lambda v:r['display_names'][v[1]]):
        vals=[r['knn']['per_segment_recall'][j]]+[np.mean([v['metrics']['per_segment_recall'][j] for v in r['runs'] if v['model']==k]) for k in ['RF','MLP']]
        page.append(f'<tr><td title="{esc(l)}">{r["display_names"][l]}</td><td>{r["support"][j]}</td>'+''.join(f'<td>{v:.1%}</td>' for v in vals)+'</tr>')
    page.append('</table><h2>Fold performance and convergence</h2><table><tr><th>Model/seed/fold</th><th>Training accuracy</th><th>Held accuracy</th><th>Held macro recall</th><th>Epochs</th><th>Warnings</th></tr>')
    for v in r['runs']:
        for f in v['folds']:
            page.append(f'<tr><td>{v["model"]}/{v["seed"]}/{f["fold"]}</td><td>{f["train"]["accuracy"]:.1%}</td><td>{f["held"]["accuracy"]:.1%}</td><td>{f["held"]["macro_recall"]:.1%}</td><td>{f.get("epochs","—")}</td><td>{esc("; ".join(f["warnings"]) or "None")}</td></tr>')
    page.append('</table><h2>MLP fitting-loss curves</h2><p>One curve per fold, color-coded by seed (17 blue, 42 orange, 123 green). Fitting loss only; no held-out data used for stopping.</p>')
    curves=[(v['seed'],f['loss_curve']) for v in r['runs'] if v['model']=='MLP' for f in v['folds']]
    xmax=max(len(c) for _,c in curves);ymax=max(max(c) for _,c in curves)
    page.append('<svg viewBox="0 0 800 280" role="img" aria-label="MLP training loss against epochs"><title>MLP fitting loss</title>')
    for seed,c in curves:
        pts=' '.join(f'{50+720*i/max(1,xmax-1):.1f},{235-210*v/ymax:.1f}' for i,v in enumerate(c))
        color={17:'#2472b8',42:'#d97706',123:'#16845b'}[seed]
        page.append(f'<polyline points="{pts}" fill="none" stroke="{color}" opacity=".65"/>')
    page.append(f'<text x="5" y="30">{ymax:.1f}</text><text x="20" y="235">0</text><text x="50" y="258">1</text><text x="730" y="258">{xmax}</text><text x="350" y="277">Epoch (fitting loss on y-axis)</text></svg>')
    for kind in ['RF','MLP']:
        page.append(f'<h2>{kind} confusion matrix</h2><p>Counts pooled over three seeds: support is therefore tripled. Row percentages describe the average confusion pattern.</p><div class="scroll"><table><tr><th>True ↓ / predicted →</th>')
        order=sorted(range(len(r['labels'])),key=lambda j:r['display_names'][r['labels'][j]])
        matrix=np.sum([v['metrics']['confusion'] for v in r['runs'] if v['model']==kind],axis=0)
        page.extend(f'<th>{r["display_names"][r["labels"][j]]}</th>' for j in order);page.append('</tr>')
        for i in order:
            page.append(f'<tr><th>{r["display_names"][r["labels"][i]]}</th>')
            for j in order:
                frac=matrix[i,j]/sum(matrix[i]);page.append(f'<td title="{matrix[i,j]} / {sum(matrix[i])}" style="background:rgba(36,114,184,{frac*.7:.3f})">{frac:.0%}</td>')
            page.append('</tr>')
        page.append('</table></div>')
    page.append('<h2>Protocol, checks, and reproducibility</h2><p>Fitting-fold median imputation for both models; standard scaling for MLP. No random internal validation split. Seed, fold, software versions, full probabilities, loss curves, and source hashes are saved in the JSON/CSV alongside this page. No trained model is selected for deployment.</p>')
    page.append('<pre>'+esc(json.dumps({'settings':r['settings'],'memorization_checks':r['memorization_checks'],'versions':r['versions']},indent=2))+'</pre><ul>'+''.join('<li>'+esc(v)+'</li>' for v in r['limitations'])+'</ul></main></html>')
    (OUT/'model_comparison.html').write_text(''.join(page))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
