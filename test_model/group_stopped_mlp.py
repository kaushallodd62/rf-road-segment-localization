#!/usr/bin/env python3
"""Group-aware inner early stopping, refit on outer training, training-CV only."""
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

from compare_models import FEATURES, MLP, OUT, RF, SEEDS, model, read, scores
from inspect_forest_leaf_scores import score_metrics
from segment_road import local_xy, point_to_line

MAX_EPOCHS = 3000
PATIENCE = 30
MIN_DELTA = 1e-4


def make_mlp(seed):
    pipe = model('MLP', seed)
    # A persistent RNG avoids resetting minibatch shuffles on every partial_fit.
    pipe[-1].set_params(random_state=np.random.RandomState(seed), n_iter_no_change=MAX_EPOCHS+1, tol=0)
    return pipe


def choose_epoch(losses, patience=PATIENCE, delta=MIN_DELTA):
    best = float('inf'); best_epoch = 0; stale = 0
    for epoch, loss in enumerate(losses, 1):
        if loss < best - delta:
            best = loss; best_epoch = epoch; stale = 0
        else:
            stale += 1
        if stale >= patience:
            return best_epoch, epoch
    return best_epoch, len(losses)


def fit_inner(X, y, train, held, labels, seed):
    pipe = make_mlp(seed)
    train_scaled = pipe[:-1].fit_transform(X[train])
    held_scaled = pipe[:-1].transform(X[held])
    np.testing.assert_allclose(pipe[0].statistics_, np.nanmedian(X[train], axis=0))
    np.testing.assert_allclose(pipe[1].mean_, np.mean(pipe[0].transform(X[train]), axis=0))
    losses=[];best=float('inf');best_epoch=0;stale=0
    for epoch in range(1, MAX_EPOCHS+1):
        pipe[-1].partial_fit(train_scaled, y[train], classes=labels)
        loss=float(log_loss(y[held], pipe[-1].predict_proba(held_scaled), labels=labels))
        assert np.isfinite(loss)
        losses.append(loss)
        if loss < best-MIN_DELTA:
            best=loss;best_epoch=epoch;stale=0
        else:
            stale+=1
        if stale>=PATIENCE:
            break
    assert choose_epoch(losses)==(best_epoch,len(losses))
    return best_epoch, losses


def refit_outer(X, y, fit, labels, seed, epochs):
    pipe=make_mlp(seed)
    scaled=pipe[:-1].fit_transform(X[fit])
    np.testing.assert_allclose(pipe[0].statistics_,np.nanmedian(X[fit],axis=0))
    np.testing.assert_allclose(pipe[1].mean_,np.mean(pipe[0].transform(X[fit]),axis=0))
    for _ in range(epochs):
        pipe[-1].partial_fit(scaled,y[fit],classes=labels)
    assert np.array_equal(pipe.classes_,labels)
    return pipe


def run():
    rows=[r for r in read('modeling_dataset.csv') if r['split']=='train']
    baseline=json.loads((OUT/'model_comparison.json').read_text())
    forest=json.loads((OUT/'forest_leaf_scores.json').read_text())
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    assert digest==baseline['training_rows_sha256']==forest['training_rows_sha256']
    assert FEATURES==baseline['features']==forest['features'] and RF==baseline['settings']['RF']
    assert SEEDS==baseline['seeds']==forest['seeds'] and sklearn.__version__==baseline['versions']['sklearn']==forest['versions']['sklearn']
    assert json.loads(json.dumps(MLP))==baseline['settings']['MLP']
    ids=[r['timestamp_unix_ms'] for r in rows]
    assert len(ids)==len(set(ids))
    X=np.array([[float(r[f]) if r[f] else np.nan for f in FEATURES] for r in rows])
    y=np.array([r['segment_id'] for r in rows]);labels=np.unique(y)
    folds=np.array([int(r['cv_fold']) for r in rows]);passes=np.array([int(r['pass_index']) for r in rows])
    assert labels.tolist()==baseline['labels'] and set(folds)==set(range(1,6)) and not np.isinf(X).any()
    points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in set(ids)}
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
        yy=y[ix]; pp=prob[ix]; pred=labels[pp.argmax(axis=1)]
        m,bins,_,_=score_metrics(yy,pp,labels)
        d=geo[ix,np.searchsorted(labels,pred)]; wrong=pred!=yy
        m.update(all_distance_median=float(np.median(d)),all_distance_p90=float(np.quantile(d,.9,method='inverted_cdf')),
                 wrong_distance_median=float(np.median(d[wrong])) if wrong.any() else None,
                 wrong_distance_p90=float(np.quantile(d[wrong],.9,method='inverted_cdf')) if wrong.any() else None,
                 per_segment_recall=recall_score(yy,pred,labels=labels,average=None,zero_division=0).tolist())
        return m,bins,d
    refs={}
    for r in read('model_comparison_predictions.csv'):
        if r['model']=='MLP':refs['MLP original',int(r['seed']),r['timestamp_unix_ms']]=r
    for r in read('forest_leaf_scores_predictions.csv'):
        if int(r['leaf']) in (2,10):refs[f'Forest leaf {r["leaf"]}',int(r['seed']),r['timestamp_unix_ms']]=r
    names=['Forest leaf 2','Forest leaf 10','MLP original','MLP grouped stop']
    assert set(refs)=={(name,seed,t) for name in names[:-1] for seed in SEEDS for t in ids}
    runs=[];output=[]
    for name in names:
        for seed in SEEDS:
            detail=[];fit_scores=[]
            if name=='MLP grouped stop':
                prob=np.empty((len(rows),len(labels)));seen=np.zeros(len(rows),int)
                for outer in range(1,6):
                    inner=outer%5+1
                    inner_fit=(folds!=outer)&(folds!=inner);inner_held=folds==inner
                    outer_fit=folds!=outer;outer_held=~outer_fit
                    for m1,m2 in [(inner_fit,inner_held),(inner_fit,outer_held),(inner_held,outer_held)]:
                        assert not set(passes[m1]) & set(passes[m2])
                    assert set(y[inner_fit])==set(labels) and set(y[outer_fit])==set(labels)
                    assert not np.isnan(X[inner_fit]).all(axis=0).any()
                    epoch,history=fit_inner(X,y,inner_fit,inner_held,labels,seed)
                    pipe=refit_outer(X,y,outer_fit,labels,seed,epoch)
                    prob[outer_held]=pipe.predict_proba(X[outer_held]);seen[outer_held]+=1
                    train_metric=scores(y[outer_fit],pipe.predict(X[outer_fit]),labels)
                    fit_scores.append(train_metric)
                    detail.append(dict(fold=outer,inner_fold=inner,inner_fit_passes=sorted(set(passes[inner_fit].tolist())),
                                       inner_held_passes=sorted(set(passes[inner_held].tolist())),outer_held_passes=sorted(set(passes[outer_held].tolist())),
                                       inner_fit_rows=int(inner_fit.sum()),inner_held_rows=int(inner_held.sum()),outer_fit_rows=int(outer_fit.sum()),
                                       outer_held_rows=int(outer_held.sum()),selected_epoch=epoch,stopped_epoch=len(history),
                                       hit_epoch_cap=len(history)==MAX_EPOCHS,inner_log_loss=history,outer_fit=train_metric))
                    print(f'group-stop seed={seed} outer={outer} inner={inner} epoch={epoch} stop={len(history)}',flush=True)
                assert np.all(seen==1)
            else:
                for t in ids:
                    ref=refs[name,seed,t];original=rows[ids.index(t)]
                    for k in ('segment_id','cv_fold','pass_index','direction'):assert ref[k]==original[k]
                prob=np.array([json.loads(refs[name,seed,t]['probabilities']) for t in ids])
                assert np.array_equal(labels[prob.argmax(axis=1)],[refs[name,seed,t]['predicted_segment_id'] for t in ids])
                if name=='MLP original':
                    oldrun=next(x for x in baseline['runs'] if x['model']=='MLP' and x['seed']==seed)
                    fit_scores=[f['train'] for f in oldrun['folds']]
                else:
                    reg=json.loads((OUT/'forest_regularization.json').read_text())
                    oldrun=next(x for x in reg['runs'] if x['leaf']==int(name.rsplit(' ',1)[1]) and x['seed']==seed)
                    fit_scores=[f['train'] for f in oldrun['folds']]
            assert np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(axis=1),1)
            m,bins,d=evaluate(prob)
            segment_scores=[]
            for label in labels:
                ix=y==label
                sm,_,_,_=score_metrics(y[ix],prob[ix],labels)
                segment_scores.append(dict(segment_id=label,support=int(ix.sum()),recall=float(np.mean(labels[prob[ix].argmax(axis=1)]==label)),
                                           top3_accuracy=sm['top3_accuracy'],log_loss=sm['log_loss'],true_score_median=sm['true_score_median']))
            m['train_accuracy']=float(np.mean([v['accuracy'] for v in fit_scores]))
            m['accuracy_gap']=m['train_accuracy']-m['accuracy']
            fold_scores=[dict(fold=fold,held_rows=int((folds==fold).sum()),metrics=evaluate(prob,folds==fold)[0]) for fold in range(1,6)]
            runs.append(dict(model=name,seed=seed,metrics=m,confidence_bins=bins,folds=fold_scores,segments=segment_scores,stopping=detail))
            if name=='MLP grouped stop':
                pred=labels[prob.argmax(axis=1)]
                for i,r in enumerate(rows):
                    output.append(dict(seed=seed,timestamp_unix_ms=ids[i],cv_fold=int(folds[i]),pass_index=int(passes[i]),
                        direction=r['direction'],segment_id=y[i],predicted_segment_id=pred[i],probabilities=json.dumps(prob[i].tolist()),distance_m=float(d[i])))
    keys=['accuracy','macro_recall','macro_f1','top3_accuracy','log_loss','multiclass_brier','train_accuracy','accuracy_gap',
          'all_distance_median','all_distance_p90','wrong_distance_median','wrong_distance_p90']
    summary=[dict(model=name,**{k:float(np.mean([r['metrics'][k] for r in runs if r['model']==name])) for k in keys}) for name in names]
    report=dict(protocol='Paired grouped training CV, no validation/test or final fit. Macro recall and log loss compared jointly; no automatic cutoff.',
                features=FEATURES,labels=labels.tolist(),display_names=baseline['display_names'],support=[int((y==v).sum()) for v in labels],
                seeds=SEEDS,mlp_settings=MLP,stopping=dict(inner_fold_rule='outer % 5 + 1',min_delta=MIN_DELTA,patience=PATIENCE,max_epochs=MAX_EPOCHS,
                    refit='Reinitialize and fit full outer fitting rows for the selected epoch count; preprocessing fitted separately for inner and outer.'),
                training_rows_sha256=digest,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                versions={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
                source_hashes={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['model_comparison_predictions.csv','forest_leaf_scores_predictions.csv','location_points_segmented.csv','road_segments.geojson']},
                runs=runs,summary=summary,
                limitations=['Seeds reuse observations and pass groups are inferred within one capture, not independent drives.',
                 'Changing the stopping process includes an inner fit and a full-outer refit; refit has more minibatch updates per epoch.',
                 'The incremental partial_fit calls reshuffle fresh index order each epoch; the original fit reshuffles the previous order, so optimization paths also differ.',
                 'Score values are not calibrated probabilities of authenticity; top-label reliability is descriptive.',
                 'Point-to-predicted-centerline distance is not road-network distance; wrong-only subsets differ.',
                 'Further repeated selection on these same development folds can overfit them; validation and test remain reserved.'])
    (OUT/'group_stopped_mlp.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'group_stopped_mlp_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    render(report)
    print(json.dumps(summary,indent=2))


def render(r):
    esc=html.escape
    parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Group-stopped MLP comparison</title><style>body{font:15px/1.5 system-ui;background:#f6f8fb;color:#18212b}main{max-width:1400px;margin:auto;padding:24px}table{border-collapse:collapse;background:white;width:100%;margin:14px 0}th,td{padding:7px;border:1px solid #dbe2eb;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}small{color:#526174}</style><main><h1>Group-aware MLP stopping versus existing references</h1><p>Same ten features, five training folds and seeds. The grouped-stopping MLP selects epochs on two inner passes disjoint from the outer held-out passes, then refits on all outer-fitting rows. Inner and outer preprocessing are fitted separately. Validation and test remain unscored.</p><h2>Overall results</h2><p>Mean of three pooled seed-specific scores; seeds are not independent drives. Lower log loss and multiclass Brier are better, but neither alone establishes calibration. Distance is point-to-predicted-centerline; wrong-only subsets differ. P90 is nearest rank.</p><div class="scroll"><table><tr><th>Model</th><th>Fit accuracy</th><th>Held accuracy</th><th>Macro recall</th><th>Macro F1</th><th>Top 3</th><th>Log loss</th><th>Brier</th><th>All median / P90 m</th><th>Wrong median / P90 m</th></tr>''']
    for s in r['summary']:
        parts.append(f'<tr><td>{esc(s["model"])}</td>'+''.join(f'<td>{s[k]:.1%}</td>' for k in ['train_accuracy','accuracy','macro_recall','macro_f1','top3_accuracy'])+f'<td>{s["log_loss"]:.3f}</td><td>{s["multiclass_brier"]:.3f}</td><td>{s["all_distance_median"]:.1f} / {s["all_distance_p90"]:.1f}</td><td>{s["wrong_distance_median"]:.1f} / {s["wrong_distance_p90"]:.1f}</td></tr>')
    parts.append('</table></div><h2>Paired seed and fold comparisons</h2><p>Each cell gives accuracy / macro recall / log loss for the same held-out fold. Macro recall includes all training labels even if absent in a particular fold, matching earlier reports.</p><div class="scroll"><table><tr><th>Seed / fold (rows)</th>'+''.join('<th>'+esc(s['model'])+'</th>' for s in r['summary'])+'</tr>')
    for seed in r['seeds']:
        for fold in range(1,6):
            ff=[next(f for f in v['folds'] if f['fold']==fold) for v in r['runs'] if v['seed']==seed]
            parts.append(f'<tr><td>{seed} / {fold} ({ff[0]["held_rows"]})</td>'+''.join(f'<td>{x["metrics"]["accuracy"]:.1%} / {x["metrics"]["macro_recall"]:.1%} / {x["metrics"]["log_loss"]:.3f}</td>' for x in ff)+'</tr>')
    parts.append('</table></div><h2>Per-segment tradeoffs</h2><p>Each cell is recall / true-segment log loss / top-three coverage, averaged across seeds; support is counted once. High log loss means low scores on the authentic segment, not classwise calibration.</p><div class="scroll"><table><tr><th>Segment</th><th>Support</th>'+''.join('<th>'+esc(s['model'])+'</th>' for s in r['summary'])+'</tr>')
    for i,label in sorted(enumerate(r['labels']),key=lambda v:r['display_names'][v[1]]):
        parts.append(f'<tr><td>{esc(r["display_names"][label])}</td><td>{r["support"][i]}</td>')
        for s in r['summary']:
            vs=[run['segments'][i] for run in r['runs'] if run['model']==s['model']];rec=np.mean([v['recall'] for v in vs]);loss=np.mean([v['log_loss'] for v in vs]);top=np.mean([v['top3_accuracy'] for v in vs]);parts.append(f'<td>{rec:.1%} / {loss:.3f} / {top:.1%}</td>')
        parts.append('</tr>')
    parts.append('</table></div><h2>Stopping and group assignments</h2><p>Inner loss is multiclass log loss. A candidate must improve by 0.0001; stop after 30 epochs without sufficient improvement, or at 3,000. Refit starts from the seed on all outer-fitting rows for the selected number of epochs.</p><div class="scroll"><table><tr><th>Seed / outer fold</th><th>Outer held passes</th><th>Inner fit passes</th><th>Inner held passes</th><th>Best / stopped epoch</th><th>Epoch cap</th></tr>')
    for v in r['runs']:
        if v['model']=='MLP grouped stop':
            for f in v['stopping']:
                parts.append(f'<tr><td>{v["seed"]} / {f["fold"]}</td><td>{f["outer_held_passes"]}</td><td>{f["inner_fit_passes"]}</td><td>{f["inner_held_passes"]}</td><td>{f["selected_epoch"]} / {f["stopped_epoch"]}</td><td>{f["hit_epoch_cap"]}</td></tr>')
    parts.append('</table></div><h2>Limitations</h2><ul>'+''.join('<li>'+esc(x)+'</li>' for x in r['limitations'])+'</ul><p>Complete runs and stopping loss curves: <a href="group_stopped_mlp.json">JSON</a>; new model out-of-fold predictions: <a href="group_stopped_mlp_predictions.csv">CSV</a>.</p></main></html>')
    (OUT/'group_stopped_mlp.html').write_text(''.join(parts))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
