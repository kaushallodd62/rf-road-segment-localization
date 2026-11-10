#!/usr/bin/env python3
"""Reproduce the forest leaf sweep and inspect held-out localization scores."""
import csv
import hashlib
import html
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from compare_models import FEATURES, OUT, RF, SEEDS, model, read, scores
from regularize_forest import LEAVES


def score_metrics(y, p, labels):
    indices = np.searchsorted(labels, y)
    true = p[np.arange(len(y)), indices]
    # Preserve the original comparison's numpy argsort convention for top-three ties.
    order = np.argsort(p, axis=1)
    top3 = (order[:, -3:] == indices[:, None]).any(axis=1)
    greater = np.sum(p > true[:, None], axis=1)
    equal = np.sum(p == true[:, None], axis=1)
    possible = greater < 3
    guaranteed = greater + equal <= 3
    winner = p.argmax(axis=1)
    confidence = p[np.arange(len(y)), winner]
    m = scores(y, labels[winner], labels)
    m.update(top3_accuracy=float(top3.mean()), top3_possible=float(possible.mean()),
             top3_guaranteed=float(guaranteed.mean()), top3_boundary_ambiguous=int((possible & ~guaranteed).sum()),
             true_score_zero=int((true == 0).sum()), true_score_p10=float(np.quantile(true,.1,method='inverted_cdf')),
             true_score_median=float(np.median(true)), true_score_mean=float(np.mean(true)),
             outside_top3_count=int((~top3).sum()),
             outside_top3_score_median=float(np.median(true[~top3])) if (~top3).any() else None,
             log_loss=float(log_loss(y,p,labels=labels)),
             multiclass_brier=float(np.mean(np.sum((p-(y[:,None]==labels[None,:]))**2,axis=1))))
    bins = []
    # Fixed bins are descriptive; no calibration model or decision threshold is fitted.
    for k in range(10):
        subset = (confidence >= k/10) & (confidence < (k+1)/10 if k < 9 else confidence <= 1)
        bins.append(dict(lower=k/10,upper=(k+1)/10,count=int(subset.sum()),
                         mean_confidence=float(confidence[subset].mean()) if subset.any() else None,
                         accuracy=float((labels[winner[subset]]==y[subset]).mean()) if subset.any() else None))
    return m, bins, top3, true


def run():
    rows = [r for r in read('modeling_dataset.csv') if r['split']=='train']
    baseline = json.loads((OUT/'model_comparison.json').read_text())
    regularization = json.loads((OUT/'forest_regularization.json').read_text())
    digest = hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    assert digest == baseline['training_rows_sha256'] == regularization['training_rows_sha256']
    assert FEATURES == baseline['features'] == regularization['features']
    assert RF == baseline['settings']['RF'] == regularization['forest_settings']
    assert SEEDS == baseline['seeds'] == regularization['seeds'] and LEAVES == regularization['leaf_sizes']
    assert sklearn.__version__ == baseline['versions']['sklearn'] == regularization['versions']['sklearn']
    ids = [r['timestamp_unix_ms'] for r in rows]
    assert len(set(ids)) == len(ids)
    X = np.array([[float(r[f]) if r[f] else np.nan for f in FEATURES] for r in rows])
    y = np.array([r['segment_id'] for r in rows]);labels = np.unique(y)
    folds = np.array([int(r['cv_fold']) for r in rows]);passes = np.array([int(r['pass_index']) for r in rows])
    assert labels.tolist() == baseline['labels'] and set(folds) == set(range(1,6)) and not np.isinf(X).any()
    prev = {(int(r['leaf']),int(r['seed']),r['timestamp_unix_ms']):r for r in read('forest_regularization_predictions.csv')}
    assert set(prev) == {(leaf,seed,t) for leaf in LEAVES for seed in SEEDS for t in ids}
    old = {(int(r['seed']),r['timestamp_unix_ms']):r for r in read('model_comparison_predictions.csv') if r['model']=='RF'}
    assert set(old) == {(seed,t) for seed in SEEDS for t in ids}
    runs=[];prediction_rows=[]
    for leaf in LEAVES:
        for seed in SEEDS:
            prob=np.empty((len(rows),len(labels)));seen=np.zeros(len(rows),int)
            for fold in sorted(set(folds)):
                fit=folds!=fold;held=~fit
                assert not set(passes[fit]) & set(passes[held])
                assert set(y[fit])==set(labels) and not np.isnan(X[fit]).all(axis=0).any()
                pipe=model('RF',seed);pipe[-1].set_params(min_samples_leaf=leaf)
                pipe.fit(X[fit],y[fit])
                assert np.array_equal(pipe.classes_,labels)
                np.testing.assert_allclose(pipe[0].statistics_,np.nanmedian(X[fit],axis=0))
                prob[held]=pipe.predict_proba(X[held]);seen[held]+=1
            assert np.all(seen==1) and np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(axis=1),1)
            pred=labels[prob.argmax(axis=1)]
            assert all(pred[i] == prev[leaf,seed,t]['predicted_segment_id'] for i,t in enumerate(ids)), 'Leaf-sweep prediction mismatch'
            if leaf==2:
                np.testing.assert_allclose(prob,np.array([json.loads(old[seed,t]['probabilities']) for t in ids]),rtol=0,atol=1e-14)
            m,bins,top3,true=score_metrics(y,prob,labels)
            details=[]
            for fold in sorted(set(folds)):
                ix=folds==fold
                fm,_,_,_=score_metrics(y[ix],prob[ix],labels)
                details.append(dict(fold=int(fold),held_rows=int(ix.sum()),metrics=fm))
            segments=[]
            for label in labels:
                ix=y==label
                sm,_,_,_=score_metrics(y[ix],prob[ix],labels)
                segments.append(dict(segment_id=label,support=int(ix.sum()),top3_accuracy=sm['top3_accuracy'],
                                     true_score_zero=sm['true_score_zero'],true_score_median=sm['true_score_median']))
            pass_results=[]
            for number in sorted(set(passes)):
                ix=passes==number;pm,_,_,_=score_metrics(y[ix],prob[ix],labels)
                pass_results.append(dict(pass_index=int(number),support=int(ix.sum()),top3_accuracy=pm['top3_accuracy'],
                                         true_score_zero=pm['true_score_zero'],true_score_median=pm['true_score_median']))
            runs.append(dict(leaf=leaf,seed=seed,metrics=m,confidence_bins=bins,folds=details,segments=segments,passes=pass_results))
            for i,r in enumerate(rows):
                earlier=prev[leaf,seed,ids[i]]
                assert earlier['segment_id']==y[i] and int(earlier['cv_fold'])==folds[i]
                prediction_rows.append(dict(leaf=leaf,seed=seed,timestamp_unix_ms=ids[i],cv_fold=int(folds[i]),
                    pass_index=int(passes[i]),direction=r['direction'],segment_id=y[i],predicted_segment_id=pred[i],
                    probabilities=json.dumps(prob[i].tolist()),true_segment_score=float(true[i]),in_top3=bool(top3[i]),
                    distance_m=float(earlier['distance_m'])))
            print(f'leaf={leaf} seed={seed}: top3={m["top3_accuracy"]:.3f}, log_loss={m["log_loss"]:.3f}, zeros={m["true_score_zero"]}',flush=True)
    metric_keys=['accuracy','macro_recall','macro_f1','top3_accuracy','top3_possible','top3_guaranteed','true_score_zero',
                 'true_score_p10','true_score_median','true_score_mean','log_loss','multiclass_brier']
    summary=[]
    for leaf in LEAVES:
        rr=[r for r in runs if r['leaf']==leaf]
        summary.append(dict(leaf=leaf,**{k:float(np.mean([r['metrics'][k] for r in rr])) for k in metric_keys}))
    paired=[]
    for seed in SEEDS:
        subset={(v['leaf'],v['timestamp_unix_ms']):v for v in prediction_rows if v['seed']==seed}
        p2=[subset[2,t] for t in ids];p10=[subset[10,t] for t in ids]
        in2=np.array([v['in_top3'] for v in p2]);in10=np.array([v['in_top3'] for v in p10])
        z2=np.array([v['true_segment_score']==0 for v in p2])
        d2=np.array([v['distance_m'] for v in p2]);d10=np.array([v['distance_m'] for v in p10])
        paired.append(dict(seed=seed,two_out_ten_in=int((~in2 & in10).sum()),two_in_ten_out=int((in2 & ~in10).sum()),
                           both_out=int((~in2 & ~in10).sum()),two_zero_count=int(z2.sum()),
                           two_zero_ten_score_median=float(np.median(np.array([v['true_segment_score'] for v in p10])[z2])) if z2.any() else None,
                           leaf2_miss_distance_median=float(np.median(d2[~in2])),
                           leaf2_miss_distance_p90=float(np.quantile(d2[~in2],.9,method='inverted_cdf')),
                           leaf10_miss_distance_median=float(np.median(d10[~in10])),
                           leaf10_miss_distance_p90=float(np.quantile(d10[~in10],.9,method='inverted_cdf'))))
    report=dict(protocol='Previously tested leaf sizes, training-only grouped CV, three seeds. No calibration fit, threshold, or attack data.',
                features=FEATURES,forest_settings=RF,leaf_sizes=LEAVES,seeds=SEEDS,labels=labels.tolist(),
                display_names=baseline['display_names'],support=[int((y==l).sum()) for l in labels],runs=runs,summary=summary,paired_2_vs_10=paired,
                training_rows_sha256=digest,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                versions={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
                source_hashes={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['forest_regularization_predictions.csv','model_comparison_predictions.csv']},
                limits=['Scores are not calibrated; top-label reliability is not classwise calibration or a spoof-detection evaluation.',
                        'Top-3 uses the earlier numpy argsort convention; guaranteed/possible bounds disclose ties at rank 3.',
                        'Seeds reuse rows, not independent drives. Repeated selection on the same folds can overfit development.',
                        'Coordinates supply labels and precomputed centerline distances, never model inputs.',
                        'Log loss clips zero scores as implemented by the pinned scikit-learn version.',
                        'Validation and test are not scored; no performance on a new day or vehicle has been established.'])
    (OUT/'forest_leaf_scores.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'forest_leaf_scores_predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(prediction_rows[0]));writer.writeheader();writer.writerows(prediction_rows)
    render(report)


def render(r):
    esc=html.escape
    parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Forest leaf scores</title><style>body{font:15px/1.5 system-ui;background:#f6f8fb;color:#18212b;margin:0}main{max-width:1400px;margin:auto;padding:24px}table{border-collapse:collapse;background:white;width:100%;margin:14px 0}th,td{padding:7px;border:1px solid #dbe2eb;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}small{color:#526174}</style><main><h1>Forest leaf size and localization scores</h1><p>Same ten features, 300 trees, five grouped training folds, seeds 17/42/123; only minimum leaf size varies. Seed repetitions reuse the same rows. Validation and test are not scored; no calibration model or threshold is fitted.</p><h2>Overall grouped CV</h2><p>Mean of three pooled seed-specific scores. Zero counts count each training row once per seed then average. P10 uses nearest rank. Log loss and multiclass Brier are lower-is-better proper scores combining calibration and discrimination, not calibration alone.</p><div class="scroll"><table><tr><th>Leaf</th><th>Accuracy</th><th>Macro recall</th><th>Macro F1</th><th>Top 3</th><th>Top 3 guaranteed–possible</th><th>True score zero (rows)</th><th>True score P10</th><th>True score median</th><th>Log loss</th><th>Brier</th></tr>''']
    for s in r['summary']:
        parts.append(f'<tr><td>{s["leaf"]}</td>'+''.join(f'<td>{s[k]:.1%}</td>' for k in ['accuracy','macro_recall','macro_f1','top3_accuracy'])+f'<td>{s["top3_guaranteed"]:.1%}–{s["top3_possible"]:.1%}</td><td>{s["true_score_zero"]:.1f}</td><td>{s["true_score_p10"]:.4f}</td><td>{s["true_score_median"]:.3f}</td><td>{s["log_loss"]:.3f}</td><td>{s["multiclass_brier"]:.3f}</td></tr>')
    parts.append('</table></div><p>The top-3 convention matches the earlier baseline: numpy argsort selects its last three indices. At equal scores on the boundary, membership depends on tie ordering. Guaranteed–possible bounds show sensitivity: guaranteed includes the true segment only if every tied segment fits; possible includes it if some tie ordering can include it. Scores do not prove an authentic claim is plausible or flag an attack.</p><h2>Paired fold/seed scores</h2><div class="scroll"><table><tr><th>Seed / fold (rows)</th>'+''.join(f'<th>Leaf {leaf}: top 3 / log loss / Brier</th>' for leaf in LEAVES)+'</tr>')
    for seed in SEEDS:
        for fold in range(1,6):
            details=[next(x for x in next(v for v in r['runs'] if v['seed']==seed and v['leaf']==leaf)['folds'] if x['fold']==fold) for leaf in LEAVES]
            parts.append(f'<tr><td>{seed} / {fold} ({details[0]["held_rows"]})</td>'+''.join(f'<td>{x["metrics"]["top3_accuracy"]:.1%} / {x["metrics"]["log_loss"]:.3f} / {x["metrics"]["multiclass_brier"]:.3f}</td>' for x in details)+'</tr>')
    parts.append('</table></div><h2>Same-row comparison: leaf 2 versus 10</h2><p>Rows and seed are matched. Miss distances use each model’s different top-3-miss subset and are not a same-row distance comparison. Distances are actual-point-to-predicted-centerline, not network distances; P90 uses nearest rank. Scores for leaf-2 zero cases are the leaf-10 scores on the same rows and seed.</p><div class="scroll"><table><tr><th>Seed</th><th>Leaf 2 out / 10 in</th><th>Leaf 2 in / 10 out</th><th>Both out</th><th>Leaf 2 zero score</th><th>Leaf 10 score median on zeros</th><th>Leaf 2 miss distance median / P90 m</th><th>Leaf 10 miss distance median / P90 m</th></tr>')
    for x in r['paired_2_vs_10']:
        parts.append(f'<tr><td>{x["seed"]}</td><td>{x["two_out_ten_in"]}</td><td>{x["two_in_ten_out"]}</td><td>{x["both_out"]}</td><td>{x["two_zero_count"]}</td><td>{x["two_zero_ten_score_median"]:.4f}</td><td>{x["leaf2_miss_distance_median"]:.1f} / {x["leaf2_miss_distance_p90"]:.1f}</td><td>{x["leaf10_miss_distance_median"]:.1f} / {x["leaf10_miss_distance_p90"]:.1f}</td></tr>')
    parts.append('</table></div><h2>Authentic-segment score by segment</h2><p>Cells show top-3 fraction / median true-segment score averaged over seeds. Support is counted once; small support is uncertain.</p><div class="scroll"><table><tr><th>Segment</th><th>Support</th>'+''.join(f'<th>Leaf {leaf}</th>' for leaf in LEAVES)+'</tr>')
    for i,label in sorted(enumerate(r['labels']),key=lambda x:r['display_names'][x[1]]):
        parts.append(f'<tr><td>{esc(r["display_names"][label])}</td><td>{r["support"][i]}</td>')
        for leaf in LEAVES:
            vals=[next(s for s in v['segments'] if s['segment_id']==label) for v in r['runs'] if v['leaf']==leaf]
            parts.append(f'<td>{np.mean([s["top3_accuracy"] for s in vals]):.1%} / {np.mean([s["true_score_median"] for s in vals]):.3f}</td>')
        parts.append('</tr>')
    parts.append('</table></div><h2>Top-label reliability: each seed separately</h2><p>For each confidence bin, compare average winning score against fraction correct; score bins are fixed intervals. Small or empty bins are unstable. This does not measure classwise calibration or provide a decision threshold.</p><div class="scroll"><table><tr><th>Leaf / seed</th><th>Score bin</th><th>Rows</th><th>Mean winning score</th><th>Observed accuracy</th></tr>')
    for v in r['runs']:
        for b in v['confidence_bins']:
            if b['count']:
                parts.append(f'<tr><td>{v["leaf"]} / {v["seed"]}</td><td>{b["lower"]:.1f}–{b["upper"]:.1f}</td><td>{b["count"]}</td><td>{b["mean_confidence"]:.3f}</td><td>{b["accuracy"]:.3f}</td></tr>')
    parts.append('</table></div><h2>Limitations and reproducibility</h2><ul>'+''.join('<li>'+esc(x)+'</li>' for x in r['limits'])+'</ul><p>All seed, fold, pass, and segment metrics are in <a href="forest_leaf_scores.json">JSON</a>; full vectors, authentic scores, top-3 status and distances are in <a href="forest_leaf_scores_predictions.csv">CSV</a>. All predictions reproduce the prior leaf sweep and leaf-2 probabilities reproduce the original baseline.</p></main></html>')
    (OUT/'forest_leaf_scores.html').write_text(''.join(parts))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
