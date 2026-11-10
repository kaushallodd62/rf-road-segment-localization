#!/usr/bin/env python3
"""Predeclared train/validation selection and one-time held-out test evaluation."""
import argparse
import csv
import hashlib
import html
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import recall_score
from threadpoolctl import threadpool_limits

from compare_models import FEATURES, MLP, OUT, RF, SEEDS, model, read, scores
from group_stopped_mlp import fit_inner, refit_outer
from inspect_forest_leaf_scores import score_metrics
from segment_road import local_xy, point_to_line
from weighted_group_mlp import train_weighted

NAMES = ['Forest leaf 2', 'Forest leaf 10', 'MLP grouped stop', 'MLP full inverse']
SOURCES = ['compare_models.py', 'group_stopped_mlp.py', 'weighted_group_mlp.py',
           'inspect_forest_leaf_scores.py', 'segment_road.py', 'evaluate_weighted_knn.py',
           'requirements-modeling.txt', 'final_evaluate.py']
SELECTION_RULE = 'Highest mean seed-specific validation macro recall; exact tie: lower mean log loss, then higher mean accuracy, then candidate order.'


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(summary):
    assert [s['model'] for s in summary] == NAMES
    return min(summary, key=lambda s: (-s['macro_recall'], s['log_loss'], -s['accuracy'], NAMES.index(s['model'])))['model']


def features(rows):
    x = np.array([[float(r[f]) if r[f] else np.nan for f in FEATURES] for r in rows])
    assert not np.isinf(x).any()
    return x


def fit_candidate(name, seed, X, y, folds, passes, labels):
    assert set(y) == set(labels) and not np.isnan(X).all(axis=0).any()
    if name.startswith('Forest'):
        pipe = model('RF', seed)
        pipe[-1].set_params(min_samples_leaf=int(name.rsplit(' ', 1)[1]))
        pipe.fit(X, y)
        np.testing.assert_allclose(pipe[0].statistics_, np.nanmedian(X, axis=0))
        return pipe, {'fit_passes': sorted(set(passes.tolist()))}
    inner_held = folds == 5; inner_fit = ~inner_held
    assert inner_held.any() and inner_fit.any()
    assert not set(passes[inner_fit]) & set(passes[inner_held])
    assert set(y[inner_fit]) == set(labels) and not np.isnan(X[inner_fit]).all(axis=0).any()
    if name == 'MLP grouped stop':
        epoch, history = fit_inner(X, y, inner_fit, inner_held, labels, seed)
        pipe = refit_outer(X, y, np.ones(len(y), bool), labels, seed, epoch)
    else:
        assert name == 'MLP full inverse'
        _, _, epoch, history = train_weighted(X, y, inner_fit, labels, seed, held=inner_held)
        pipe, _, _, _ = train_weighted(X, y, np.ones(len(y), bool), labels, seed, epochs=epoch)
    assert np.array_equal(pipe.classes_, labels)
    return pipe, {'fit_passes': sorted(set(passes.tolist())),
                  'inner_fit_passes': sorted(set(passes[inner_fit].tolist())),
                  'inner_stopping_passes': sorted(set(passes[inner_held].tolist())),
                  'selected_epoch': epoch, 'stopped_epoch': len(history), 'inner_log_loss': history}


def metric_bundle(y, p, labels, distances):
    result, bins, top3, true = score_metrics(y, p, labels)
    pred = labels[p.argmax(axis=1)]; wrong = pred != y
    result.update(per_segment_recall=recall_score(y, pred, labels=labels, average=None, zero_division=0).tolist(),
                  all_distance_median=float(np.median(distances)),
                  all_distance_p90=float(np.quantile(distances, .9, method='inverted_cdf')),
                  wrong_distance_median=float(np.median(distances[wrong])) if wrong.any() else None,
                  wrong_distance_p90=float(np.quantile(distances[wrong], .9, method='inverted_cdf')) if wrong.any() else None)
    return result, bins, top3, true


def render(report, output):
    esc = html.escape
    parts = ['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Final grouped evaluation</title><style>body{font:15px/1.5 system-ui;background:#f6f8fb;color:#18212b}main{max-width:1350px;margin:auto;padding:24px}table{border-collapse:collapse;background:white;width:100%;margin:14px 0}th,td{padding:7px;border:1px solid #dbe2eb;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}</style><main>''']
    parts.append(f'<h1>{esc(report["split"].title())} results</h1><p>{esc(report["protocol"])}</p><p>Each summary is the mean of three seed-specific results on the same observations. The seeds are model repetitions, not independent drives. Lower log loss and Brier are better but do not establish calibration.</p><div class="scroll"><table><tr><th>Model</th><th>Fit accuracy</th><th>Accuracy</th><th>Macro recall</th><th>Macro F1</th><th>Top 3</th><th>Log loss</th><th>Brier</th><th>All median/P90 m</th><th>Wrong median/P90 m</th></tr>')
    for s in report['summary']:
        parts.append('<tr><td>'+esc(s['model'])+'</td>'+''.join(f'<td>{s[k]:.1%}</td>' for k in ['train_accuracy','accuracy','macro_recall','macro_f1','top3_accuracy'])+f'<td>{s["log_loss"]:.3f}</td><td>{s["multiclass_brier"]:.3f}</td><td>{s["all_distance_median"]:.1f} / {s["all_distance_p90"]:.1f}</td><td>{s["wrong_distance_median"]:.1f} / {s["wrong_distance_p90"]:.1f}</td></tr>')
    parts.append('</table></div><p>Distances are reported point to predicted centerline, not road-network distance. Wrong-only subsets differ between models. P90 uses nearest rank. Scores are not calibrated authenticity probabilities.</p><h2>Per-segment recall</h2><p>Mean of seed-specific recalls; segment support counts observations once. Small segment counts can make recall unstable.</p><div class="scroll"><table><tr><th>Segment</th><th>Support</th>'+''.join('<th>'+esc(s['model'])+'</th>' for s in report['summary'])+'</tr>')
    for i,label in sorted(enumerate(report['labels']),key=lambda x:report['display_names'][x[1]]):
        parts.append(f'<tr><td>{esc(report["display_names"][label])}</td><td>{report["support"][i]}</td>')
        for s in report['summary']:
            val=np.mean([v['metrics']['per_segment_recall'][i] for v in report['runs'] if v['model']==s['model']])
            parts.append(f'<td>{val:.1%}</td>')
        parts.append('</tr>')
    parts.append('</table></div><h2>Each seed</h2><div class="scroll"><table><tr><th>Model / seed</th><th>Accuracy</th><th>Macro recall</th><th>Macro F1</th><th>Top 3</th><th>Log loss</th><th>Brier</th></tr>')
    for v in report['runs']:
        m=v['metrics'];parts.append(f'<tr><td>{esc(v["model"])} / {v["seed"]}</td>'+''.join(f'<td>{m[k]:.1%}</td>' for k in ['accuracy','macro_recall','macro_f1','top3_accuracy'])+f'<td>{m["log_loss"]:.3f}</td><td>{m["multiclass_brier"]:.3f}</td></tr>')
    parts.append('</table></div><h2>Per-pass performance</h2><p>Macro recall uses all training labels, including zero for labels absent from that pass, consistent with earlier fold reports. It is useful for matched comparison, not as standalone pass accuracy.</p><div class="scroll"><table><tr><th>Model / seed / pass (rows)</th><th>Accuracy</th><th>Macro recall</th><th>Log loss</th></tr>')
    for v in report['runs']:
        for p in v['passes']:
            m=p['metrics'];parts.append(f'<tr><td>{esc(v["model"])} / {v["seed"]} / {p["pass_index"]} ({p["rows"]})</td><td>{m["accuracy"]:.1%}</td><td>{m["macro_recall"]:.1%}</td><td>{m["log_loss"]:.3f}</td></tr>')
    parts.append('</table></div><h2>Selection and limits</h2><p>'+esc(report['selection_rule'])+'</p><p>Selected configuration: '+esc(report['selected_model'])+'.</p><ul>'+''.join('<li>'+esc(v)+'</li>' for v in report['limitations'])+'</ul><p>Detailed metrics and predictions are in the JSON and CSV beside this page. No spoof-detection threshold is tested.</p></main></html>')
    output.write_text(''.join(parts))


def main(split):
    assert split in ('validation','test')
    result_path=OUT/f'final_{split}.json';pred_path=OUT/f'final_{split}_predictions.csv';page_path=OUT/f'final_{split}.html'
    selection_path=OUT/'final_selection.json'
    if result_path.exists() or pred_path.exists() or page_path.exists() or (split=='validation' and selection_path.exists()) or (split=='test' and (OUT/'final_reference_seed17.joblib').exists()):
        raise RuntimeError(f'{split} outputs already exist; refusing to rescore or overwrite')
    selection = json.loads(selection_path.read_text()) if split=='test' else None
    if selection:
        assert selection['rule']==SELECTION_RULE and selection['seeds']==SEEDS and selection['selected_model'] in NAMES
    with (OUT/'modeling_dataset.csv').open(newline='') as f:
        relevant=[r for r in csv.DictReader(f) if r['split'] in ('train',split)]
    train=[r for r in relevant if r['split']=='train'];held=[r for r in relevant if r['split']==split]
    assert len(train)==1613 and len(held)==(514 if split=='validation' else 516)
    assert len({r['timestamp_unix_ms'] for r in train+held})==len(train)+len(held)
    baseline=json.loads((OUT/'model_comparison.json').read_text())
    assert digest(train)==baseline['training_rows_sha256'] and FEATURES==baseline['features'] and SEEDS==baseline['seeds']
    assert RF==baseline['settings']['RF'] and json.loads(json.dumps(MLP))==baseline['settings']['MLP']
    assert sklearn.__version__==baseline['versions']['sklearn']
    labels=np.unique([r['segment_id'] for r in train]); assert labels.tolist()==baseline['labels']
    yt=np.array([r['segment_id'] for r in train]);y=np.array([r['segment_id'] for r in held])
    assert set(y)==set(labels)
    folds=np.array([int(r['cv_fold']) for r in train]);passes=np.array([int(r['pass_index']) for r in train])
    held_passes=np.array([int(r['pass_index']) for r in held])
    assert set(passes)==set(range(1,11)) and set(held_passes)==(set(range(11,14)) if split=='validation' else set(range(14,18)))
    assert not set(passes)&set(held_passes) and set(folds)==set(range(1,6))
    X=features(train);H=features(held)
    train_ids=[r['timestamp_unix_ms'] for r in train];ids=[r['timestamp_unix_ms'] for r in held]
    lookup=set(train_ids+ids)
    points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in lookup}
    assert set(points)==lookup
    roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    lon0=np.mean([float(points[t]['longitude']) for t in train_ids]);lat0=np.mean([float(points[t]['latitude']) for t in train_ids])
    geo=np.empty((len(held),len(labels)))
    for i,t in enumerate(ids):
        assert points[t]['matched_segment_id']==y[i]
        xy=local_xy(float(points[t]['longitude']),float(points[t]['latitude']),lon0,lat0)
        for j,label in enumerate(labels):geo[i,j]=point_to_line(xy,roads[label],lon0,lat0)[0]
    assert np.isfinite(geo).all()
    code_hashes={n:file_hash(Path(__file__).parent/n) for n in SOURCES}
    input_hashes={n:file_hash(OUT/n) for n in ['location_points_segmented.csv','road_segments.geojson']}
    if selection:
        assert selection['training_rows_sha256']==digest(train)
        assert selection['validation_result_sha256']==file_hash(OUT/'final_validation.json')
        assert selection['validation_rows_sha256']==json.loads((OUT/'final_validation.json').read_text())['held_rows_sha256']
        assert selection['validation_predictions_sha256']==file_hash(OUT/'final_validation_predictions.csv')
        assert selection['code_hashes']==code_hashes and selection['geometry_hashes']==input_hashes
        assert selection['versions']=={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__}
    names=NAMES if split=='validation' else [selection['selected_model']]
    runs=[];output=[];reference_pipe=None
    for name in names:
        for seed in SEEDS:
            pipe,detail=fit_candidate(name,seed,X,yt,folds,passes,labels)
            assert np.array_equal(pipe.classes_,labels)
            prob=pipe.predict_proba(H)
            assert np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(axis=1),1)
            pred=labels[prob.argmax(axis=1)];d=geo[np.arange(len(y)),np.searchsorted(labels,pred)]
            m,bins,top3,true=metric_bundle(y,prob,labels,d)
            m['train_accuracy']=float(pipe.score(X,yt));m['accuracy_gap']=m['train_accuracy']-m['accuracy']
            pass_results=[]
            for p in sorted(set(held_passes)):
                ix=held_passes==p;pm,_,_,_=metric_bundle(y[ix],prob[ix],labels,d[ix])
                pass_results.append(dict(pass_index=int(p),rows=int(ix.sum()),metrics=pm))
            runs.append(dict(model=name,seed=seed,metrics=m,confidence_bins=bins,passes=pass_results,training=detail))
            for i,row in enumerate(held):
                output.append(dict(model=name,seed=seed,timestamp_unix_ms=ids[i],pass_index=int(held_passes[i]),direction=row['direction'],
                    segment_id=y[i],predicted_segment_id=pred[i],probabilities=json.dumps(prob[i].tolist()),true_segment_score=float(true[i]),
                    in_top3=bool(top3[i]),distance_m=float(d[i])))
            if split=='test' and seed==17: reference_pipe=pipe
            print(f'{split}: {name} seed {seed}, accuracy={m["accuracy"]:.3f}, macro recall={m["macro_recall"]:.3f}',flush=True)
    keys=['train_accuracy','accuracy','macro_recall','macro_f1','top3_accuracy','log_loss','multiclass_brier',
          'all_distance_median','all_distance_p90','wrong_distance_median','wrong_distance_p90']
    summary=[dict(model=name,**{k:float(np.mean([v['metrics'][k] for v in runs if v['model']==name])) for k in keys}) for name in names]
    selected=select(summary) if split=='validation' else selection['selected_model']
    report=dict(split=split,protocol='Train on passes 1 through 10 only. Validation selects the configuration; test evaluates the frozen selection.',
                selection_rule=SELECTION_RULE,selected_model=selected,labels=labels.tolist(),display_names=baseline['display_names'],
                support=[int((y==label).sum()) for label in labels],seeds=SEEDS,features=FEATURES,
                training_rows_sha256=digest(train),held_rows_sha256=digest(held),
                versions={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
                code_hashes=code_hashes,geometry_hashes=input_hashes,runs=runs,summary=summary,
                limitations=['One vehicle and capture; held-out passes are not independent drives.',
                  'Model selection reuses training CV and a small validation set; no statistical significance or cross-drive claims.',
                  '16 equal class weights in macro recall can magnify small-support segment variation.',
                  'The source and decision-time availability of the existing message-freshness input remain unresolved.',
                  'Actual coordinates provide labels and evaluation geometry only, not model inputs.',
                  'No attack evaluation, calibration model or acceptance threshold has been tested.'])
    result_path.write_text(json.dumps(report,indent=2)+'\n')
    with pred_path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    render(report,page_path)
    if split=='validation':
        frozen=dict(rule=SELECTION_RULE,selected_model=selected,seeds=SEEDS,training_rows_sha256=digest(train),
                    validation_rows_sha256=digest(held),validation_result_sha256=file_hash(result_path),
                    validation_predictions_sha256=file_hash(pred_path),code_hashes=code_hashes,geometry_hashes=input_hashes,
                    versions=report['versions'],reference_seed=17,fit_split='train',candidate_order=NAMES)
        selection_path.write_text(json.dumps(frozen,indent=2)+'\n')
        print('Frozen selection:',selected,flush=True)
    else:
        assert reference_pipe is not None
        target=OUT/'final_reference_seed17.joblib'
        assert not target.exists()
        joblib.dump(reference_pipe,target)
        loaded=joblib.load(target)
        np.testing.assert_allclose(loaded.predict_proba(H),reference_pipe.predict_proba(H))
        print('Saved seed-17 reference model:',target,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('split',choices=['validation','test']);args=parser.parse_args()
    with threadpool_limits(limits=1):main(args.split)
