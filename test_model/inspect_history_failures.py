#!/usr/bin/env python3
"""Describe saved forest failures without fitting or selecting a model."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from compare_rnn_histories import OUT, SEEDS, common_histories, digest, read

STEM='history_forest_failures_v1'


def segment_visits(rows,runs):
    visits={}
    for run_id,run in enumerate(runs):
        start=0
        while start<len(run):
            end=start+1
            while end<len(run) and rows[run[end]]['segment_id']==rows[run[start]]['segment_id']:end+=1
            previous=rows[run[start-1]]['segment_id'] if start else None
            for j in range(start,end):
                visits[run[j]]=dict(run=run_id,age=j-start,visit_length=end-start,previous_segment=previous,
                    observed_entry=start>0)
            start=end
    return visits


def summarize(records):
    if not records:return dict(windows=0,recall=None)
    return dict(windows=len({r['timestamp_unix_ms'] for r in records}),
        predictions=len(records),recall=float(np.mean([r['correct'] for r in records])),
        top3=float(np.mean([r['true_in_top3'] for r in records])),
        mean_true_probability=float(np.mean([r['true_probability'] for r in records])),
        zero_true_probability=float(np.mean([r['true_probability']==0 for r in records])),
        mean_same_direction_fit_windows=float(np.mean([r['same_direction_fit_windows'] for r in records])),
        mean_same_direction_fit_passes=float(np.mean([r['same_direction_fit_passes'] for r in records])),
        mean_segment_windows_in_history=float(np.mean([r['segment_windows_in_history'] for r in records])),
        confusions=[dict(predicted=s,count=n,mean_count_per_seed=n/len(SEEDS)) for s,n in Counter(r['prediction'] for r in records if not r['correct']).most_common()])


def run():
    paths=[OUT/(STEM+s) for s in ['.json','_observations.csv','.html']]
    if any(p.exists() for p in paths):raise RuntimeError('Refusing overwrite')
    base=json.loads((OUT/'rnn_history_lengths_v1.json').read_text())
    for f,h in base['source_sha256'].items():assert digest(OUT/f)==h
    rows=read('training_windows_v1.csv');ids=[r['timestamp_unix_ms'] for r in rows];needed=set(ids)
    assert len(rows)==len(needed)==1393 and all(r['split']=='train' and 1<=int(r['pass_index'])<=10 for r in rows)
    points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in needed}
    q,w,runs,_=common_histories(rows,[points[t]['track_id'] for t in ids]);visits=segment_visits(rows,runs)
    lineage=read('rnn_history_lengths_v1_lineage.csv')
    assert [ids[i] for i in q]==[r['timestamp_unix_ms'] for r in lineage]
    assert [[ids[i] for i in h] for h in w]==[json.loads(r['history_timestamps']) for r in lineage]
    saved={(int(r['length']),int(r['seed']),r['timestamp_unix_ms']):r for r in read('rnn_history_lengths_v1_predictions.csv') if r['family']=='RF'}
    selected=[rows[i] for i in q];labels=base['labels'];names=base['display_names'];records=[]
    for seed in SEEDS:
        for j,i in enumerate(q):
            r=rows[i];s=r['segment_id'];visit=visits[i]
            old=saved[10,seed,ids[i]]
            for key in ['pass_index','cv_fold','segment_id']:assert r[key]==old[key]
            pp=np.array(json.loads(old['probabilities']));rank=np.argsort(-pp,kind='stable');label_index=labels.index(s)
            fit=[a for a in selected if a['cv_fold']!=r['cv_fold'] and a['segment_id']==s]
            same=[a for a in fit if a['direction']==r['direction']]
            records.append(dict(seed=seed,timestamp_unix_ms=ids[i],segment=s,direction=r['direction'],pass_index=int(r['pass_index']),cv_fold=int(r['cv_fold']),
                prediction=old['predicted_segment_id'],correct=old['predicted_segment_id']==s,true_probability=float(pp[label_index]),
                true_in_top3=label_index in rank[:3],fit_windows=len(fit),same_direction_fit_windows=len(same),same_direction_fit_passes=len({a['pass_index'] for a in same}),
                segment_windows_in_history=sum(rows[k]['segment_id']==s for k in w[j]),
                previous_segment=visit['previous_segment'] or '',predicted_previous=bool(visit['observed_entry'] and old['predicted_segment_id']==visit['previous_segment']),
                age=visit['age'],visit_length=visit['visit_length'],observed_entry=visit['observed_entry'],
                stage='first 3 observations' if visit['age']<3 else 'observations 4–9' if visit['age']<9 else 'observation 10 onward',
                short_visit=visit['visit_length']<10,old_summary_gap_ms=int(r['summary_to_ue_last_seen_ms']) if r['summary_to_ue_last_seen_ms'] else None,
                **{f'correct_length_{length}':saved[length,seed,ids[i]]['predicted_segment_id']==s for length in [1,3,5,10]}))
    assert len(records)==1217*3
    segment_reports=[]
    for s in sorted(labels,key=lambda s:names[s]):
        rr=[r for r in records if r['segment']==s];original=[r for r in rows if r['segment_id']==s]
        segment_reports.append(dict(segment=s,name=names[s],all_training_windows=len(original),query_windows=len(rr)//3,
            training_passes=sorted({int(r['pass_index']) for r in original}),summary=summarize(rr),
            by_direction={d:summarize([r for r in rr if r['direction']==d]) for d in ['N','S']},
            by_pass={str(p):summarize([r for r in rr if r['pass_index']==p]) for p in sorted({r['pass_index'] for r in rr})},
            by_stage={stage:summarize([r for r in rr if r['stage']==stage]) for stage in sorted({r['stage'] for r in rr})},
            recall_by_history={str(length):float(np.mean([r[f'correct_length_{length}'] for r in rr])) for length in [1,3,5,10]},
            mean_history_same_segment=float(np.mean([r['segment_windows_in_history'] for r in rr])),
            previous_segment_error_fraction=float(np.mean([r['predicted_previous'] for r in rr if not r['correct']])) if any(not r['correct'] for r in rr) else None))
    report=dict(protocol='Saved leaf-2 forest predictions only; no fits or model selection; training passes 1–10.',
        eligible_windows=1217,seeds=SEEDS,segments=segment_reports,
        by_stage={stage:summarize([r for r in records if r['stage']==stage]) for stage in ['first 3 observations','observations 4–9','observation 10 onward']},
        by_support={key:summarize([r for r in records if (r['same_direction_fit_windows']==0)==value]) for key,value in [('no_same_direction_examples',True),('some_same_direction_examples',False)]},
        by_summary_age={key:summarize([r for r in records if r['old_summary_gap_ms'] is not None and (r['old_summary_gap_ms']>1000)==value]) for key,value in [('over_1s',True),('at_most_1s',False)]},
        source_sha256={f:digest(OUT/f) for f in ['training_windows_v1.csv','rnn_history_lengths_v1.json','rnn_history_lengths_v1_predictions.csv','rnn_history_lengths_v1_lineage.csv']},
        code_sha256=digest(Path(__file__)),dependency_sha256={'compare_rnn_histories.py':digest(Path(__file__).parent/'compare_rnn_histories.py')},
        limitations=['Descriptive comparisons mix segment, direction and pass effects; they do not prove causes.',
            'Seeds repeat predictions for the same windows, not independent observations. Counts distinguish both.',
            'Visit age uses consecutive reported labels within a continuous run; run starts have no observed entry.',
            'A short visit may be truncated by capture or history reset. Physical RF/position alignment remains unverified.',
            'Top-three ties use saved label order; probabilities are not calibrated confidence.',
            'No validation/test predictions, new models or changes to the research reference.'])
    paths[0].write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with paths[1].open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    cells=''
    for r in segment_reports:
        z=r['summary'];conf=', '.join(f"{names[c['predicted']]} ({c['mean_count_per_seed']:.1f})" for c in z['confusions'][:2])
        cells+=f"<tr><td>{r['name']}</td><td>{r['query_windows']}</td><td>{100*z['recall']:.1f}%</td><td>{100*z['top3']:.1f}%</td><td>{r['recall_by_history']['1']*100:.1f}%</td><td>{r['mean_history_same_segment']:.1f}/10</td><td>{conf}</td></tr>"
    paths[2].write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Forest failures</title><style>body{font:16px/1.5 system-ui;max-width:1200px;margin:32px auto}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>Where the forest still fails</h1><p>Saved training-fold predictions, averaged across three seeds. All history lengths use the same 1,217 query windows. Recall is the share of a segment’s measurements correctly identified. Top 3 means the true segment was among the three highest scoring choices. Confusion counts average seeds.</p><table><tr><th>Segment</th><th>Windows</th><th>Recall: history 10</th><th>Top 3</th><th>Recall: history 1</th><th>History windows from current segment</th><th>Most common wrong predictions</th></tr>'+cells+'</table><p>Full JSON includes direction, pass, entry stage, fitting support and history comparisons. These describe patterns; they do not prove what caused the errors. Only training data were analyzed.</p><a href="'+STEM+'.json">Detailed results</a></html>')
    for r in segment_reports:
        print(r['name'],r['query_windows'],round(100*r['summary']['recall'],1),'history',round(r['mean_history_same_segment'],1),'fit',r['summary']['mean_same_direction_fit_windows'],flush=True)


if __name__=='__main__':run()
