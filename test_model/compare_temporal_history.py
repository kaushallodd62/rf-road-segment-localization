#!/usr/bin/env python3
"""Three-window causal smoothing and matched RF-history refits, training only."""
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from compare_direction_feature import ADDED, INPUTS, MLP, OUT, RF, SEEDS, digest, encode_direction, fit_added, input_matrix, paired_changes, read
from compare_window_models import KEYS
from optimize_models import evaluate, geography

STEM = 'temporal_history_v1'
FILES = [STEM+s for s in ['.json', '_predictions.csv', '_lineage.csv', '.html']]
GAP_MS = 1500
ARMS = ['saved current', 'mean of 3 predictions', 'matched current', '3-window RF history']


def histories(rows, tracks):
    timestamps = np.array([int(r['timestamp_unix_ms']) for r in rows])
    if len(set(timestamps)) != len(rows): raise ValueError('Summary timestamps must be unique')
    history = np.full((len(rows),3),-1,int); runs = []; resets = Counter(); previous = None
    for i in np.argsort(timestamps,kind='stable'):
        r = rows[i]; key = (r['node_id'],r['session_id'],tracks[i],r['pass_index'],r['direction'])
        reason = 'start' if previous is None else 'group' if key != previous[0] else 'gap' if timestamps[i]-previous[1] > GAP_MS else None
        if reason:
            runs.append([]); resets[reason] += 1
        runs[-1].append(int(i))
        if len(runs[-1]) >= 3:
            history[i] = runs[-1][-3:]
            if len({rows[j]['cv_fold'] for j in history[i]}) != 1:
                raise ValueError('History crosses a CV fold')
        previous = key,timestamps[i]
    return history,runs,dict(resets)


def input_history(x, direction, history, queries):
    h = history[queries]
    if (h < 0).any(): raise ValueError('Full history required')
    return np.column_stack([x[h[:,2]],x[h[:,1]],x[h[:,0]],direction[queries]])


def transition_metrics(rows, runs, queries, pred):
    positions = {int(q):i for i,q in enumerate(queries)}
    events = []; excluded = 0; switches = 0; steps = 0
    for run in runs:
        for a,b in zip(run,run[1:]):
            if a in positions and b in positions:
                steps += 1; switches += int(pred[positions[a]] != pred[positions[b]])
        starts = [j for j in range(1,len(run)) if rows[run[j]]['segment_id'] != rows[run[j-1]]['segment_id']]
        for start in starts:
            first = run[start]
            end = next((j for j in range(start+1,len(run)) if rows[run[j]]['segment_id'] != rows[first]['segment_id']),len(run))
            if first not in positions:
                excluded += 1; continue
            span = run[start:end]
            correct = next((j for j,q in enumerate(span) if q in positions and pred[positions[q]] == rows[first]['segment_id']),None)
            events.append(dict(timestamp_unix_ms=rows[first]['timestamp_unix_ms'],
                               from_segment=rows[run[start-1]]['segment_id'],to_segment=rows[first]['segment_id'],
                               run_windows=len(span),first_correct_after_windows=correct,
                               first_correct_after_export_ms=int(rows[span[correct]]['timestamp_unix_ms'])-int(rows[first]['timestamp_unix_ms']) if correct is not None else None))
    detected = [e for e in events if e['first_correct_after_windows'] is not None]
    return dict(events=events,eligible_changes=len(events),excluded_warmup_changes=excluded,
                immediate_correct=sum(e['first_correct_after_windows']==0 for e in detected),
                never_correct_before_exit=len(events)-len(detected),
                median_first_correct_windows=float(np.median([e['first_correct_after_windows'] for e in detected])) if detected else None,
                median_first_correct_export_ms=float(np.median([e['first_correct_after_export_ms'] for e in detected])) if detected else None,
                prediction_switches=switches,consecutive_steps=steps,switch_rate=switches/steps if steps else None)


def render(report):
    cells = ''.join(f"<tr><td>{r['model']}</td><td>{r['accuracy']*100:.1f}%</td><td>{r['macro_recall']*100:.1f}%</td><td>{r['all_distance_p90']:.0f} m</td><td>{r['immediate_correct']:.1f}</td><td>{r['never_correct_before_exit']:.1f}</td><td>{r['prediction_switches']:.1f}</td></tr>" for r in report['summary'])
    names = report['display_names']; per_segment = []
    for kind in ['RF','MLP']:
        rows = []
        for i,s in sorted(enumerate(report['labels']),key=lambda pair:names[pair[1]]):
            values = [np.mean([r['metrics']['per_segment_recall'][i] for r in report['runs'] if r['kind']==kind and r['arm']==arm]) for arm in ARMS]
            rows.append(f"<tr><td>{names[s]}</td><td>{report['support'][s]}</td>"+''.join(f'<td>{v*100:.1f}%</td>' for v in values)+'</tr>')
        per_segment.append(f'<h2>{kind}: recall for each segment</h2><div class="scroll"><table><tr><th>Segment</th><th>Windows</th>'+''.join(f'<th>{html.escape(a)}</th>' for a in ARMS)+'</tr>'+''.join(rows)+'</table></div>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Short RF histories</title><style>body{font:16px/1.5 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#17283a}table{border-collapse:collapse}td,th{border:1px solid #ccd5df;padding:7px;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}</style><h1>Do recent RF measurements improve segment predictions?</h1><p>All scores use the same 1,335 held-pass observations with two recent preceding windows. There are five folds and three seeds. Histories restart after group changes or exported gaps longer than 1.5 seconds.</p><p>Compare “saved current” with “mean of 3 predictions” to assess averaging. Compare “matched current” with “3-window RF history” to assess model inputs: both were refitted on the same eligible training observations. Every model includes current reported direction.</p><h2>Overall results and segment changes</h2><p>Macro recall gives every segment equal importance. P90 is the error distance that 90% of predictions stay within. Entry-correct counts predictions correct on the first observation after a segment change. Missed counts changes where the model never predicts the new segment before the vehicle leaves it. Counts average seeds.</p><div class="scroll"><table><tr><th>Model</th><th>Accuracy</th><th>Macro recall</th><th>P90 error</th><th>Entry correct</th><th>Missed changes</th><th>Prediction switches</th></tr>'''+cells+'</table></div>'+''.join(per_segment)+'''<h2>Limits</h2><ul>'''+''.join('<li>'+html.escape(v)+'</li>' for v in report['limitations'])+'''</ul><p><a href="temporal_history_v1.json">All metrics, confusions and transition events</a> · <a href="temporal_history_v1_predictions.csv">Predictions</a> · <a href="temporal_history_v1_lineage.csv">History timestamps</a></p></html>'''


def run():
    if any((OUT/name).exists() for name in FILES): raise RuntimeError('Temporal artifacts exist; refusing overwrite')
    base = json.loads((OUT/'direction_feature_v1.json').read_text())
    for f,h in base['source_sha256'].items(): assert digest(OUT/f) == h
    for f,h in base['dependency_sha256'].items(): assert digest(Path(__file__).parent/f) == h
    assert digest(Path(__file__).parent/'compare_direction_feature.py') == base['code_sha256']
    assert base['features'] == INPUTS+['reported_direction_north'] and base['seeds'] == SEEDS
    assert base['rf_settings'] == dict(RF,max_features=2) and base['mlp_settings'] == json.loads(json.dumps(MLP))
    rows = read('training_windows_v1.csv'); ids = [r['timestamp_unix_ms'] for r in rows]; needed = set(ids)
    assert len(rows)==len(needed)==1393 and all(r['split']=='train' and 1<=int(r['pass_index'])<=10 for r in rows)
    assert len({tuple(r[k] for k in ['node_id','session_id','analyzer_window_id']) for r in rows})==len(rows)
    points = {r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in needed}
    assert set(points)==needed and len({r['track_id'] for r in points.values()})==1
    history,runs,resets = histories(rows,[points[t]['track_id'] for t in ids])
    queries = np.flatnonzero(history[:,0]>=0); assert len(queries)==1335
    direction = encode_direction(rows); x = input_matrix(rows)
    current = np.column_stack([x[queries],direction[queries]])
    past = input_history(x,direction,history,queries)
    selected = [rows[i] for i in queries]; y = np.array([r['segment_id'] for r in selected]); labels = np.unique(y)
    assert labels.tolist() == base['labels']
    folds = np.array([int(r['cv_fold']) for r in selected]); passes = np.array([int(r['pass_index']) for r in selected])
    for i in queries:
        assert all(rows[j]['cv_fold']==rows[i]['cv_fold'] for j in history[i])
        assert all(int(rows[j]['timestamp_unix_ms'])<=int(rows[i]['timestamp_unix_ms']) for j in history[i])
    coverage=[]
    for f in range(1,6):
        inner=f%5+1
        portions={name:dict(rows=int(mask.sum()),support={s:int((y[mask]==s).sum()) for s in labels}) for name,mask in
                  [('outer_fit',folds!=f),('outer_held',folds==f),('inner_fit',(folds!=f)&(folds!=inner)),('inner_stop',folds==inner)]}
        assert all(min(p['support'].values())>0 for p in portions.values())
        coverage.append(dict(fold=f,inner_fold=inner,portions=portions))
    roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    geo=geography(selected,labels,[ids[i] for i in queries],roads,points)
    saved={(r['model'],int(r['seed']),r['timestamp_unix_ms']):r for r in read('direction_feature_v1_predictions.csv')}
    results=[];predictions=[];paired=[]
    def record(kind,arm,seed,probabilities,training):
        assert np.isfinite(probabilities).all() and (probabilities>=0).all() and np.allclose(probabilities.sum(1),1)
        metrics,distances=evaluate(y,probabilities,labels,geo);pred=labels[probabilities.argmax(1)]
        transitions=transition_metrics(rows,runs,queries,pred)
        confusion=Counter((a,b) for a,b in zip(y,pred) if a!=b)
        result=dict(kind=kind,arm=arm,model=f'{kind}: {arm}',seed=seed,metrics=metrics,training=training,transitions=transitions,
                    folds=[dict(fold=f,metrics=evaluate(y[folds==f],probabilities[folds==f],labels,geo[folds==f])[0]) for f in range(1,6)],
                    confusions=[dict(actual=a,predicted=b,count=n) for (a,b),n in confusion.most_common()])
        results.append(result)
        for j,i in enumerate(queries):
            predictions.append(dict(model=result['model'],seed=seed,timestamp_unix_ms=ids[i],pass_index=int(passes[j]),cv_fold=int(folds[j]),
                                    segment_id=y[j],predicted_segment_id=pred[j],probabilities=json.dumps(probabilities[j].tolist()),distance_m=float(distances[j])))
    for kind in ['RF','MLP']:
        for seed in SEEDS:
            saved_rows=[saved[ADDED[kind],seed,t] for t in ids]
            for r,s in zip(rows,saved_rows):
                for k in ['segment_id','pass_index','cv_fold']: assert r[k]==s[k]
            allp=np.array([json.loads(r['probabilities']) for r in saved_rows])
            a=allp[queries];b=allp[history[queries]].mean(axis=1)
            record(kind,ARMS[0],seed,a,[]);record(kind,ARMS[1],seed,b,[])
            paired.append(dict(kind=kind,seed=seed,comparison='averaging',changes=paired_changes(y,a,b,labels,geo,folds)))
            fitted={}
            for arm,features in [(ARMS[2],current),(ARMS[3],past)]:
                pp=np.empty((len(y),len(labels)));details=[];seen=np.zeros(len(y),int)
                for outer in range(1,6):
                    pipe,detail=fit_added(kind,seed,features,y,labels,folds,passes,outer)
                    held=folds==outer;pp[held]=pipe.predict_proba(features[held]);seen[held]+=1
                    assert np.array_equal(pipe.classes_,labels) and pipe.n_features_in_==features.shape[1]
                    details.append(detail)
                    print(f'{kind} {arm} seed={seed} fold={outer}'+(f' epoch={detail["selected_epoch"]}' if kind=='MLP' else ''),flush=True)
                assert np.all(seen==1)
                record(kind,arm,seed,pp,details);fitted[arm]=pp
            paired.append(dict(kind=kind,seed=seed,comparison='input history',changes=paired_changes(y,fitted[ARMS[2]],fitted[ARMS[3]],labels,geo,folds)))
    summary=[]
    for kind in ['RF','MLP']:
        for arm in ARMS:
            rr=[r for r in results if r['kind']==kind and r['arm']==arm]
            summary.append(dict(model=f'{kind}: {arm}',**{key:float(np.mean([r['metrics'][key] for r in rr])) for key in KEYS},
                                **{key:float(np.mean([r['transitions'][key] for r in rr])) for key in ['immediate_correct','never_correct_before_exit','prediction_switches']}))
    lineage=[dict(timestamp_unix_ms=ids[i],previous_timestamp_unix_ms=ids[history[i,1]],previous_previous_timestamp_unix_ms=ids[history[i,0]],
                  pass_index=rows[i]['pass_index'],cv_fold=rows[i]['cv_fold']) for i in queries]
    report=dict(protocol='Fixed causal three-window comparison; training observations only.',history_length=3,gap_reset_ms=GAP_MS,
                rows=1393,eligible_rows=len(queries),excluded_rows=len(rows)-len(queries),resets=resets,coverage=coverage,
                current_features=INPUTS+['reported_direction_north'],history_features=[f'{lag}:{f}' for lag in ['current','previous','previous_previous'] for f in INPUTS]+['reported_direction_north'],
                labels=labels.tolist(),display_names=base['display_names'],support={s:int((y==s).sum()) for s in labels},seeds=SEEDS,
                rf_settings=base['rf_settings'],mlp_settings=MLP,stopping=base['stopping'],runs=results,summary=summary,paired=paired,
                source_sha256={f:digest(OUT/f) for f in ['training_windows_v1.csv','direction_feature_v1.json','direction_feature_v1_predictions.csv','road_segments.geojson']},
                training_points_sha256=hashlib.sha256(json.dumps([points[t] for t in ids],sort_keys=True).encode()).hexdigest(),
                code_sha256=digest(Path(__file__)),dependency_sha256={f:digest(Path(__file__).parent/f) for f in ['compare_direction_feature.py','compare_window_models.py','compare_models.py','group_stopped_mlp.py','optimize_models.py']},
                versions=dict(numpy=np.__version__,sklearn=sklearn.__version__),
                limitations=['Exported summary times order histories; physical RF/position timing remains unverified.',
                             'Reported direction remains an assumed available input; earlier true segments and coordinates never enter features.',
                             'Histories overlap within one capture; repeated seeds and reused folds do not establish independent-drive performance.',
                             'Some inner stopping classes have only one eligible observation, despite complete class coverage.',
                             'Averaging uses saved all-row models; current/history refits use identical eligible fitting rows. Compare within each pair.',
                             'Fewer prediction switches can mean delayed segment changes. First-correct delays describe only detected events; missed events are separate.',
                             'History dimensionality changes fitting and optimization. This fixed recipe cannot establish whether all temporal models help or fail.',
                             'No RNN, deployment latency measurement, attack evaluation, new capture or final model selection.'])
    (OUT/FILES[0]).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for filename,records in [(FILES[1],predictions),(FILES[2],lineage)]:
        with (OUT/filename).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (OUT/FILES[3]).write_text(render(report));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
