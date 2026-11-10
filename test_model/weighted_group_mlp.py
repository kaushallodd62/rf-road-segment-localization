#!/usr/bin/env python3
"""Compare one inverse-frequency MLP with saved grouped-CV references."""
import csv
import hashlib
import html
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import log_loss, recall_score
from threadpoolctl import threadpool_limits

from compare_models import FEATURES, MLP, OUT, SEEDS, read, scores
from group_stopped_mlp import MAX_EPOCHS, MIN_DELTA, PATIENCE, choose_epoch, make_mlp
from inspect_forest_leaf_scores import score_metrics
from segment_road import local_xy, point_to_line


def fitting_weights(y, labels):
    """Inverse frequency on exactly the fitting labels; mean weight is one."""
    counts = np.array([(y == label).sum() for label in labels])
    assert len(y) and np.all(counts > 0) and set(y) == set(labels)
    class_weights = len(y)/(len(labels)*counts)
    weights = class_weights[np.searchsorted(labels,y)]
    np.testing.assert_allclose(np.mean(weights),1)
    return weights, counts, class_weights


def train_weighted(X, y, fit, labels, seed, epochs=None, held=None):
    pipe=make_mlp(seed)
    scaled=pipe[:-1].fit_transform(X[fit])
    np.testing.assert_allclose(pipe[0].statistics_,np.nanmedian(X[fit],axis=0))
    np.testing.assert_allclose(pipe[1].mean_,np.mean(pipe[0].transform(X[fit]),axis=0))
    w,counts,class_weights=fitting_weights(y[fit],labels)
    metadata=dict(counts=counts.tolist(),class_weights=class_weights.tolist(),
                  median=pipe[0].statistics_.tolist(),scale_mean=pipe[1].mean_.tolist())
    assert (held is None) != (epochs is None)
    losses=[];best=float('inf');best_epoch=0;stale=0
    stop_scaled=pipe[:-1].transform(X[held]) if held is not None else None
    for epoch in range(1,(MAX_EPOCHS if epochs is None else epochs)+1):
        pipe[-1].partial_fit(scaled,y[fit],sample_weight=w,classes=labels)
        if held is not None:
            # Validation criterion deliberately stays unweighted, matching the reference.
            loss=float(log_loss(y[held],pipe[-1].predict_proba(stop_scaled),labels=labels))
            assert np.isfinite(loss)
            losses.append(loss)
            if loss < best-MIN_DELTA:
                best=loss;best_epoch=epoch;stale=0
            else:stale+=1
            if stale>=PATIENCE:break
    if held is not None:
        assert choose_epoch(losses)==(best_epoch,len(losses))
    return pipe,metadata,best_epoch,losses


def run():
    rows=[r for r in read('modeling_dataset.csv') if r['split']=='train']
    base=json.loads((OUT/'group_stopped_mlp.json').read_text())
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    assert digest==base['training_rows_sha256'] and base['features']==FEATURES and base['seeds']==SEEDS
    assert sklearn.__version__==base['versions']['sklearn'] and json.loads(json.dumps(MLP))==base['mlp_settings']
    assert hashlib.sha256((Path(__file__).parent/'group_stopped_mlp.py').read_bytes()).hexdigest()==base['code_sha256']
    for n in ['location_points_segmented.csv','road_segments.geojson']:
        assert hashlib.sha256((OUT/n).read_bytes()).hexdigest()==base['source_hashes'][n]
    ids=[r['timestamp_unix_ms'] for r in rows];assert len(ids)==len(set(ids))
    X=np.array([[float(r[f]) if r[f] else np.nan for f in FEATURES] for r in rows])
    y=np.array([r['segment_id'] for r in rows]);labels=np.unique(y)
    folds=np.array([int(r['cv_fold']) for r in rows]);passes=np.array([int(r['pass_index']) for r in rows])
    assert labels.tolist()==base['labels'] and set(folds)==set(range(1,6)) and not np.isinf(X).any()
    id_set=set(ids)
    points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in id_set}
    roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    lon0=np.mean([float(points[t]['longitude']) for t in ids]);lat0=np.mean([float(points[t]['latitude']) for t in ids])
    geo=np.empty((len(rows),len(labels)))
    for i,t in enumerate(ids):
        assert points[t]['matched_segment_id']==y[i]
        xy=local_xy(float(points[t]['longitude']),float(points[t]['latitude']),lon0,lat0)
        for j,label in enumerate(labels):geo[i,j]=point_to_line(xy,roads[label],lon0,lat0)[0]
    assert np.isfinite(geo).all()
    def evaluate(prob,mask=None):
        ix=np.ones(len(y),bool) if mask is None else mask
        yy=y[ix];pp=prob[ix];pred=labels[pp.argmax(axis=1)]
        m,bins,_,_=score_metrics(yy,pp,labels)
        d=geo[ix,np.searchsorted(labels,pred)];wrong=pred!=yy
        m.update(all_distance_median=float(np.median(d)),all_distance_p90=float(np.quantile(d,.9,method='inverted_cdf')),
                 wrong_distance_median=float(np.median(d[wrong])) if wrong.any() else None,
                 wrong_distance_p90=float(np.quantile(d[wrong],.9,method='inverted_cdf')) if wrong.any() else None,
                 per_segment_recall=recall_score(yy,pred,labels=labels,average=None,zero_division=0).tolist())
        return m,bins,d
    old={(int(r['seed']),r['timestamp_unix_ms']):r for r in read('group_stopped_mlp_predictions.csv')}
    assert set(old)=={(seed,t) for seed in SEEDS for t in ids}
    runs=[r for r in base['runs'] if r['model'] in ('Forest leaf 2','Forest leaf 10','MLP grouped stop')]
    assert len(runs)==9
    output=[];paired=[]
    for seed in SEEDS:
        prob=np.empty((len(rows),len(labels)));seen=np.zeros(len(rows),int);details=[];fit_scores=[]
        for outer in range(1,6):
            inner=outer%5+1
            inner_fit=(folds!=outer)&(folds!=inner);inner_held=folds==inner
            outer_fit=folds!=outer;outer_held=~outer_fit
            for a,b in [(inner_fit,inner_held),(inner_fit,outer_held),(inner_held,outer_held)]:assert not set(passes[a]) & set(passes[b])
            assert not np.isnan(X[inner_fit]).all(axis=0).any()
            _,inner_meta,epoch,history=train_weighted(X,y,inner_fit,labels,seed,held=inner_held)
            pipe,outer_meta,_,_=train_weighted(X,y,outer_fit,labels,seed,epochs=epoch)
            assert np.array_equal(pipe.classes_,labels)
            prob[outer_held]=pipe.predict_proba(X[outer_held]);seen[outer_held]+=1
            train_metric=scores(y[outer_fit],pipe.predict(X[outer_fit]),labels);fit_scores.append(train_metric)
            details.append(dict(fold=outer,inner_fold=inner,inner_fit_passes=sorted(set(passes[inner_fit].tolist())),
                                inner_held_passes=sorted(set(passes[inner_held].tolist())),outer_held_passes=sorted(set(passes[outer_held].tolist())),
                                inner_fit_rows=int(inner_fit.sum()),inner_held_rows=int(inner_held.sum()),outer_fit_rows=int(outer_fit.sum()),
                                outer_held_rows=int(outer_held.sum()),inner_preprocessing_and_weights=inner_meta,outer_preprocessing_and_weights=outer_meta,
                                selected_epoch=epoch,stopped_epoch=len(history),hit_epoch_cap=len(history)==MAX_EPOCHS,
                                inner_unweighted_log_loss=history,outer_fit=train_metric))
            print(f'weighted seed={seed} outer={outer} selected={epoch} stopped={len(history)}',flush=True)
        assert np.all(seen==1) and np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(axis=1),1)
        pred=labels[prob.argmax(axis=1)];m,bins,d=evaluate(prob)
        m['train_accuracy']=float(np.mean([v['accuracy'] for v in fit_scores]));m['accuracy_gap']=m['train_accuracy']-m['accuracy']
        fold_scores=[dict(fold=f,held_rows=int((folds==f).sum()),metrics=evaluate(prob,folds==f)[0]) for f in range(1,6)]
        segment_scores=[]
        for label in labels:
            ix=y==label;sm,_,_,_=score_metrics(y[ix],prob[ix],labels)
            segment_scores.append(dict(segment_id=label,support=int(ix.sum()),recall=float(np.mean(pred[ix]==label)),
                                       top3_accuracy=sm['top3_accuracy'],log_loss=sm['log_loss'],true_score_median=sm['true_score_median']))
        runs.append(dict(model='MLP weighted',seed=seed,metrics=m,confidence_bins=bins,folds=fold_scores,segments=segment_scores,stopping=details))
        oldpred=[];oldprob=[]
        for i,r in enumerate(rows):
            prior=old[seed,ids[i]]
            for k in ('segment_id','cv_fold','pass_index','direction'):assert prior[k]==r[k]
            oldpred.append(prior['predicted_segment_id']);oldprob.append(json.loads(prior['probabilities']))
            output.append(dict(seed=seed,timestamp_unix_ms=ids[i],cv_fold=int(folds[i]),pass_index=int(passes[i]),direction=r['direction'],
                               segment_id=y[i],predicted_segment_id=pred[i],probabilities=json.dumps(prob[i].tolist()),distance_m=float(d[i])))
        oldpred=np.array(oldpred);oldprob=np.array(oldprob)
        original=next(v for v in base['runs'] if v['model']=='MLP grouped stop' and v['seed']==seed)
        oldm,_,_=evaluate(oldprob)
        for k in ('accuracy','macro_recall','macro_f1','top3_accuracy','log_loss','multiclass_brier'):assert np.isclose(oldm[k],original['metrics'][k])
        assert np.array_equal(labels[oldprob.argmax(axis=1)],oldpred)
        c_old=oldpred==y;c_new=pred==y
        paired.append(dict(seed=seed,both_correct=int((c_old&c_new).sum()),only_weighted_correct=int((~c_old&c_new).sum()),
                           only_unweighted_correct=int((c_old&~c_new).sum()),both_wrong=int((~c_old&~c_new).sum())))
    names=['Forest leaf 2','Forest leaf 10','MLP grouped stop','MLP weighted']
    keys=['accuracy','macro_recall','macro_f1','top3_accuracy','log_loss','multiclass_brier','train_accuracy','accuracy_gap',
          'all_distance_median','all_distance_p90','wrong_distance_median','wrong_distance_p90']
    summary=[dict(model=name,**{k:float(np.mean([r['metrics'][k] for r in runs if r['model']==name])) for k in keys}) for name in names]
    result=dict(protocol='One inverse-frequency weighting scheme; paired grouped training CV; no validation/test or final fit.',
                features=FEATURES,labels=labels.tolist(),display_names=base['display_names'],support=base['support'],seeds=SEEDS,
                weighting='N/(K*n_class), fit rows only; separate inner and outer weights; inner stopping loss remains unweighted.',
                mlp_settings=MLP,stopping=base['stopping'],training_rows_sha256=digest,
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                versions={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
                source_hashes={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['group_stopped_mlp.json','group_stopped_mlp_predictions.csv','location_points_segmented.csv','road_segments.geojson']},
                summary=summary,runs=runs,paired=paired,
                limitations=['Each seed uses the same rows; inferred passes from one capture are not independent drives.',
                 'Rare-class weighting can magnify noise and alter score calibration; stopping on unweighted log loss may not favor this weighted objective.',
                 'Weighted minibatch normalization also changes the effective contribution of regularization; this compares a complete training procedure.',
                 'A small or inconsistent CV difference does not establish generalization; repeated selection can overfit these folds.',
                 'Geographic distances are point-to-predicted-centerline, not network distance; wrong-only subsets differ.',
                 'No calibrated authenticity probabilities, attack evaluation, validation/test scoring, or spoof threshold.'])
    (OUT/'weighted_group_mlp.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'weighted_group_mlp_predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    render(result)
    print(json.dumps(summary,indent=2))


def render(r):
    esc=html.escape
    parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Weighted MLP comparison</title><style>body{font:15px/1.5 system-ui;background:#f6f8fb;color:#18212b;margin:0}main{max-width:1450px;margin:auto;padding:24px}table{border-collapse:collapse;background:white;width:100%;margin:14px 0}th,td{padding:7px;border:1px solid #dbe2eb;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}small{color:#526174}</style><main><h1>Class-weighted versus unweighted MLP</h1><p>Same ten features, architecture, grouped early stopping, folds and seeds. The sole intentional training change is inverse segment-frequency weighting on fitting rows. Stop on unweighted inner log loss; separately refit preprocessing and weights on full outer fitting rows. Seeds reuse rows and are not independent drives.</p><h2>Overall results</h2><p>Three seed-specific pooled CV scores averaged. Lower log loss and Brier are better; neither establishes calibration. Geographic median/P90 are actual-point-to-predicted-centerline distances in meters; wrong-only populations differ. P90 uses nearest rank.</p><div class="scroll"><table><tr><th>Model</th><th>Fit accuracy</th><th>Held accuracy</th><th>Macro recall</th><th>Macro F1</th><th>Top 3</th><th>Log loss</th><th>Brier</th><th>All median / P90 m</th><th>Wrong median / P90 m</th></tr>''']
    for s in r['summary']:
        parts.append('<tr><td>'+esc(s['model'])+'</td>'+''.join(f'<td>{s[k]:.1%}</td>' for k in ['train_accuracy','accuracy','macro_recall','macro_f1','top3_accuracy'])+f'<td>{s["log_loss"]:.3f}</td><td>{s["multiclass_brier"]:.3f}</td><td>{s["all_distance_median"]:.1f} / {s["all_distance_p90"]:.1f}</td><td>{s["wrong_distance_median"]:.1f} / {s["wrong_distance_p90"]:.1f}</td></tr>')
    parts.append('</table></div><h2>Paired folds and seeds</h2><p>Each cell: accuracy / macro recall / log loss. Fold macro recall uses all training labels, as in earlier reports.</p><div class="scroll"><table><tr><th>Seed / fold (rows)</th>'+''.join('<th>'+esc(s['model'])+'</th>' for s in r['summary'])+'</tr>')
    for seed in r['seeds']:
        for f in range(1,6):
            ff=[next(x for x in next(v for v in r['runs'] if v['model']==s['model'] and v['seed']==seed)['folds'] if x['fold']==f) for s in r['summary']]
            parts.append(f'<tr><td>{seed} / {f} ({ff[0]["held_rows"]})</td>'+''.join(f'<td>{v["metrics"]["accuracy"]:.1%} / {v["metrics"]["macro_recall"]:.1%} / {v["metrics"]["log_loss"]:.3f}</td>' for v in ff)+'</tr>')
    parts.append('</table></div><h2>Per-segment results</h2><p>Each cell: recall / true-segment log loss / top-3 coverage, averaged over seeds. Support counts rows once. Small supports yield unstable rates.</p><div class="scroll"><table><tr><th>True segment</th><th>Rows</th>'+''.join('<th>'+esc(s['model'])+'</th>' for s in r['summary'])+'</tr>')
    for i,label in sorted(enumerate(r['labels']),key=lambda v:r['display_names'][v[1]]):
        parts.append(f'<tr><td>{esc(r["display_names"][label])}</td><td>{r["support"][i]}</td>')
        for s in r['summary']:
            v=[x['segments'][i] for x in r['runs'] if x['model']==s['model']]
            parts.append(f'<td>{np.mean([x["recall"] for x in v]):.1%} / {np.mean([x["log_loss"] for x in v]):.3f} / {np.mean([x["top3_accuracy"] for x in v]):.1%}</td>')
        parts.append('</tr>')
    parts.append('</table></div><h2>Paired prediction changes</h2><p>Rows and seeds matched; counts are not independent observations.</p><table><tr><th>Seed</th><th>Both correct</th><th>Only weighted correct</th><th>Only unweighted correct</th><th>Both wrong</th></tr>')
    for p in r['paired']:
        parts.append(f'<tr><td>{p["seed"]}</td><td>{p["both_correct"]}</td><td>{p["only_weighted_correct"]}</td><td>{p["only_unweighted_correct"]}</td><td>{p["both_wrong"]}</td></tr>')
    parts.append('</table><h2>Group-aware early stopping</h2><p>For each outer fold, a pair of inner passes chooses the epoch by unweighted log loss; fitting and stopping passes are disjoint. The model is then restarted and refit for that many epochs on all outer-fitting passes before predicting the outer-held passes. Median imputation, scaling, and class weights are fitted separately in the inner and refit stages. Complete group assignments, class weights, and inner loss histories appear in the JSON.</p><div class="scroll"><table><tr><th>Seed / outer</th><th>Inner fit passes</th><th>Inner stop passes</th><th>Outer held passes</th><th>Best / stopped epoch</th></tr>')
    for v in r['runs']:
        if v['model']=='MLP weighted':
            for f in v['stopping']:
                parts.append(f'<tr><td>{v["seed"]} / {f["fold"]}</td><td>{f["inner_fit_passes"]}</td><td>{f["inner_held_passes"]}</td><td>{f["outer_held_passes"]}</td><td>{f["selected_epoch"]} / {f["stopped_epoch"]}</td></tr>')
    parts.append('</table></div><h2>Limits and files</h2><ul>'+''.join('<li>'+esc(x)+'</li>' for x in r['limitations'])+'</ul><p>Full metrics and weights: <a href="weighted_group_mlp.json">JSON</a>. New held-out predictions: <a href="weighted_group_mlp_predictions.csv">CSV</a>.</p></main></html>')
    (OUT/'weighted_group_mlp.html').write_text(''.join(parts))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
