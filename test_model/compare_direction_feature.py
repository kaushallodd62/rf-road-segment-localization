#!/usr/bin/env python3
"""Fixed eight-input versus eight-plus-reported-direction training comparison."""
import csv
import hashlib
import html
import itertools
import json
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from check_window_feature_dependency import INPUTS, NAMES, input_matrix
from compare_models import MLP, OUT, RF, SEEDS, model, read
from compare_window_models import KEYS, fit_one
from group_stopped_mlp import MAX_EPOCHS
from optimize_models import evaluate, geography
from prepare_evaluation_data import direction_from_heading

STEM = 'direction_feature_v1'
FILES = [STEM+s for s in ['.json', '_predictions.csv', '.html']]
ADDED = {'RF': 'Forest plus direction', 'MLP': 'MLP plus direction'}
FLIPPED = {'RF': 'Forest wrong direction', 'MLP': 'MLP wrong direction'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_direction(rows):
    if any(r['direction'] not in ['N', 'S'] for r in rows):
        raise ValueError('Direction must be N or S; unknown values require another protocol')
    return np.array([int(r['direction'] == 'N') for r in rows], float)


def direction_prior(y, directions, fit, held, labels):
    probabilities = np.empty((held.sum(), len(labels)))
    detail = {}
    for d in [0, 1]:
        counts = np.array([int((fit & (directions == d) & (y == s)).sum()) for s in labels])
        probabilities[directions[held] == d] = (counts+1)/(counts.sum()+len(labels))
        detail[str(d)] = counts.tolist()
    return probabilities, detail


def direction_effects(rows, x, along, features):
    y = np.array([r['segment_id'] for r in rows])
    passes = np.array([int(r['pass_index']) for r in rows])
    directions = {int(r['pass_index']):r['direction'] for r in rows}
    records = []
    for s in sorted(set(y)):
        bins = np.floor(along/50).astype(int)
        for bin_id in [None, *sorted(set(bins[y == s]))]:
            population = (y == s) & ((bins == bin_id) if bin_id is not None else True)
            for j, feature in enumerate(features):
                summaries = []
                for p in sorted(directions):
                    values = x[population & (passes == p), j]
                    values = values[np.isfinite(values)]
                    summaries.append(dict(pass_index=p, direction=directions[p], support=len(values),
                                          median=float(np.median(values)) if len(values) >= 5 else None))
                north = [r['median'] for r in summaries if r['direction'] == 'N' and r['median'] is not None]
                south = [r['median'] for r in summaries if r['direction'] == 'S' and r['median'] is not None]
                supported = min(len(north), len(south)) >= 2
                delta = float(np.median(north)-np.median(south)) if north and south else None
                drift = [abs(a-b) for group in [north, south] for a,b in itertools.combinations(group, 2)]
                typical = float(np.median(drift)) if supported else None
                records.append(dict(segment=s, bin_start_m=int(bin_id*50) if bin_id is not None else None,
                                    feature=feature, supported=supported, pass_summaries=summaries,
                                    north_median=float(np.median(north)) if north else None,
                                    south_median=float(np.median(south)) if south else None,
                                    north_minus_south=delta, within_direction_change=typical,
                                    gap_to_change=abs(delta)/typical if supported and typical > 0 else None))
    return records


def fit_added(kind, seed, x, y, labels, folds, passes, outer):
    if kind == 'MLP':
        return fit_one(kind, seed, x, y, labels, folds, passes, outer)
    fit = folds != outer; held = ~fit
    assert not set(passes[fit]) & set(passes[held]) and set(y[fit]) == set(labels)
    pipe = model('RF', seed)
    pipe[-1].set_params(max_features=2)
    pipe.fit(x[fit], y[fit])
    np.testing.assert_allclose(pipe[0].statistics_, np.nanmedian(x[fit], axis=0))
    return pipe, dict(outer_fold=outer, outer_fit_rows=int(fit.sum()), outer_held_rows=int(held.sum()),
                      outer_fit_passes=sorted(set(passes[fit].tolist())), outer_held_passes=sorted(set(passes[held].tolist())),
                      outer_median=pipe[0].statistics_.tolist(), candidate_features=2)


def paired_changes(y, reference, candidate, labels, geo, folds):
    result = []
    for f in range(6):
        mask = folds == f if f else np.ones(len(y), bool)
        a = evaluate(y[mask], reference[mask], labels, geo[mask])[0]
        b = evaluate(y[mask], candidate[mask], labels, geo[mask])[0]
        ac = labels[reference[mask].argmax(axis=1)] == y[mask]
        bc = labels[candidate[mask].argmax(axis=1)] == y[mask]
        result.append(dict(fold=f, support=int(mask.sum()), delta={k:b[k]-a[k] for k in KEYS},
                           reference_only_correct=int((ac & ~bc).sum()), candidate_only_correct=int((~ac & bc).sum())))
    return result


def make_html(report):
    cells = ''.join(f"<tr><td>{r['model']}</td><td>{r['accuracy']*100:.1f}%</td><td>{r['macro_recall']*100:.1f}%</td><td>{r['log_loss']:.3f}</td><td>{r['all_distance_p90']:.0f} m</td></tr>" for r in report['summary'])
    names = report['display_names']; per_segment = []
    for s in sorted(report['labels'], key=names.get):
        index = report['labels'].index(s)
        values = []
        for name in [NAMES['RF'], ADDED['RF'], NAMES['MLP'], ADDED['MLP']]:
            values.append(np.mean([r['metrics']['per_segment_recall'][index] for r in report['runs'] if r['model'] == name]))
        per_segment.append(f"<tr><td>{names[s]}</td><td>{report['direction_support'][s]['N']}</td><td>{report['direction_support'][s]['S']}</td>"+''.join(f'<td>{v*100:.1f}%</td>' for v in values)+'</tr>')
    effects = [r for r in report['direction_effects'] if r['feature']=='rsrp_dbm' and r['bin_start_m'] is None]
    effect_rows = ''.join(f"<tr><td>{names[r['segment']]}</td><td>{r['north_median']:.2f}</td><td>{r['south_median']:.2f}</td><td>{r['north_minus_south']:+.2f}</td><td>{'yes' if r['supported'] else 'no'}</td></tr>" for r in sorted(effects,key=lambda r:names[r['segment']]) if r['north_median'] is not None and r['south_median'] is not None)
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Travel direction experiment</title><style>body{font:16px/1.5 system-ui;color:#17283a;max-width:1100px;margin:32px auto;padding:0 20px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd5df;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}</style>
<h1>Does travel direction help predict the segment?</h1><p>1,393 distinct training windows, five held-pass folds, three seeds. Scores average the three seed-specific results. The direction-only control is deterministic.</p><p>North/south comes from the vehicle's reported heading. Coordinates and pass numbers remain outside the model inputs. These results assume that reported direction is available and correct.</p><h2>Overall results</h2><p>Macro recall gives every segment equal importance. P90 distance is the error distance that 90% of predictions stay within. Wrong-direction rows flip the added input at prediction time without retraining.</p><div class="scroll"><table><tr><th>Model</th><th>Accuracy</th><th>Macro recall</th><th>Log loss</th><th>P90 distance</th></tr>'''+cells+'''</table></div><h2>Recall for each segment</h2><div class="scroll"><table><tr><th>Segment</th><th>North windows</th><th>South windows</th><th>Forest RF only</th><th>Forest + direction</th><th>MLP RF only</th><th>MLP + direction</th></tr>'''+''.join(per_segment)+'''</table></div><h2>RSRP difference by travel direction</h2><p>For each pass, calculate its observed median. Give supported passes equal weight, then subtract the typical southbound value from the northbound value. At least two passes in each direction, each with five values, are required for a supported comparison. Positions can differ within a segment. S19 has no northbound windows and is omitted here.</p><div class="scroll"><table><tr><th>Segment</th><th>North dBm</th><th>South dBm</th><th>North minus south dB</th><th>Supported?</th></tr>'''+effect_rows+'''</table></div><h2>Limits</h2><ul>'''+''.join('<li>'+html.escape(v)+'</li>' for v in report['limitations'])+'''</ul><p><a href="direction_feature_v1.json">Full results and feature changes</a> · <a href="direction_feature_v1_predictions.csv">Saved predictions</a></p></html>'''


def run():
    if any((OUT/f).exists() for f in FILES): raise RuntimeError('Direction experiment outputs exist; refusing overwrite')
    base = json.loads((OUT/'window_feature_dependency.json').read_text())
    manifest = json.loads((OUT/'window_dataset_v1.json').read_text())
    assert digest(OUT/'training_windows_v1.csv') == base['window_csv_sha256']
    assert digest(OUT/'window_dataset_v1.json') == base['manifest_sha256']
    assert INPUTS == base['features'] and RF == base['rf_settings'] and SEEDS == base['seeds']
    assert json.loads(json.dumps(MLP)) == base['mlp_settings'] and sklearn.__version__ == base['versions']['sklearn']
    for f,h in manifest['artifact_sha256'].items(): assert digest(OUT/f) == h
    for f,h in base['dependency_sha256'].items(): assert digest(Path(__file__).parent/f) == h
    rows = read('training_windows_v1.csv'); ids = [r['timestamp_unix_ms'] for r in rows]; needed = set(ids)
    assert len(rows) == len(needed) == 1393 and all(r['split']=='train' and 1 <= int(r['pass_index']) <= 10 for r in rows)
    assert len({tuple(r[k] for k in manifest['observation_key']) for r in rows}) == len(rows)
    points = {r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in needed}
    assert set(points) == needed
    assert all(direction_from_heading(points[r['timestamp_unix_ms']]['reported_heading_deg']) == r['direction'] for r in rows)
    direction = encode_direction(rows); x = input_matrix(rows); augmented = np.column_stack([x, direction])
    flipped = augmented.copy(); flipped[:, -1] = 1-direction
    y = np.array([r['segment_id'] for r in rows]); labels = np.unique(y)
    folds = np.array([int(r['cv_fold']) for r in rows]); passes = np.array([int(r['pass_index']) for r in rows])
    assert all(int(r['cv_fold']) == (int(r['pass_index'])+1)//2 for r in rows)
    roads = {f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    geo = geography(rows, labels, ids, roads, points)
    saved = {(r['model'],int(r['seed']),r['timestamp_unix_ms']):r for r in read('window_feature_dependency_predictions.csv')}
    runs = []; predictions = []; paired = []
    def record(name, seed, probabilities, training):
        assert np.isfinite(probabilities).all() and (probabilities >= 0).all() and np.allclose(probabilities.sum(axis=1), 1)
        metrics, distances = evaluate(y, probabilities, labels, geo)
        by_direction = {d:evaluate(y[direction==value],probabilities[direction==value],labels,geo[direction==value])[0]
                        for d,value in [('N',1),('S',0)]}
        runs.append(dict(model=name,seed=seed,metrics=metrics,training=training,by_direction=by_direction,
                         folds=[dict(fold=f,metrics=evaluate(y[folds==f],probabilities[folds==f],labels,geo[folds==f])[0]) for f in range(1,6)]))
        for i,row in enumerate(rows):
            predictions.append(dict(model=name,seed=seed,timestamp_unix_ms=ids[i],pass_index=int(passes[i]),cv_fold=int(folds[i]),
                                    direction=row['direction'],segment_id=y[i],predicted_segment_id=labels[probabilities[i].argmax()],
                                    probabilities=json.dumps(probabilities[i].tolist()),distance_m=float(distances[i])))
    prior = np.empty((len(y),len(labels))); prior_details = []
    for outer in range(1,6):
        fit = folds != outer; held = ~fit
        assert not set(passes[fit]) & set(passes[held])
        prior[held], counts = direction_prior(y,direction,fit,held,labels)
        prior_details.append(dict(outer_fold=outer, counts=counts))
    record('Direction only',0,prior,prior_details)
    for kind in ['RF','MLP']:
        for seed in SEEDS:
            oldrows = [saved[NAMES[kind],seed,t] for t in ids]
            for row,old in zip(rows,oldrows):
                for key in ['segment_id','pass_index','cv_fold']: assert row[key] == old[key]
            oldp = np.array([json.loads(r['probabilities']) for r in oldrows])
            record(NAMES[kind],seed,oldp,[])
            oldrun = next(r for r in base['runs'] if r['model']==NAMES[kind] and r['seed']==seed)
            for key in KEYS: np.testing.assert_allclose(runs[-1]['metrics'][key],oldrun['metrics'][key])
            p = np.empty_like(oldp); wrong = np.empty_like(oldp); details = []; seen = np.zeros(len(y),int)
            for outer in range(1,6):
                pipe, detail = fit_added(kind,seed,augmented,y,labels,folds,passes,outer)
                held = folds == outer
                assert pipe.n_features_in_ == 9 and np.array_equal(pipe.classes_,labels)
                p[held] = pipe.predict_proba(augmented[held]); wrong[held] = pipe.predict_proba(flipped[held]); seen[held] += 1
                details.append(detail)
                print(f'{kind} + direction seed={seed} fold={outer}' + (f' epoch={detail["selected_epoch"]}' if kind=='MLP' else ''),flush=True)
            assert np.all(seen == 1)
            record(ADDED[kind],seed,p,details); record(FLIPPED[kind],seed,wrong,[])
            paired.append(dict(kind=kind,seed=seed,addition=paired_changes(y,oldp,p,labels,geo,folds),
                               flip=paired_changes(y,p,wrong,labels,geo,folds)))
    order = ['Direction only',NAMES['RF'],ADDED['RF'],FLIPPED['RF'],NAMES['MLP'],ADDED['MLP'],FLIPPED['MLP']]
    summary = [dict(model=name,**{key:float(np.mean([r['metrics'][key] for r in runs if r['model']==name])) for key in KEYS}) for name in order]
    along = np.array([float(points[t]['along_segment_m']) for t in ids])
    report = dict(protocol='Fixed paired eight-input plus reported-direction comparison; training windows only.',
                  features=INPUTS+['reported_direction_north'],direction_encoding={'N':1,'S':0},labels=labels.tolist(),display_names=base['display_names'],
                  direction_support={s:{d:sum(r['segment_id']==s and r['direction']==d for r in rows) for d in ['N','S']} for s in labels},
                  rf_settings=dict(RF,max_features=2),mlp_settings=MLP,seeds=SEEDS,stopping=base['stopping'],
                  runs=runs,summary=summary,paired=paired,direction_effects=direction_effects(rows,x,along,INPUTS),
                  source_sha256={f:digest(OUT/f) for f in ['training_windows_v1.csv','window_dataset_v1.json','window_feature_dependency.json','window_feature_dependency_predictions.csv','road_segments.geojson']},
                  training_points_sha256=hashlib.sha256(json.dumps([points[t] for t in ids],sort_keys=True).encode()).hexdigest(),
                  code_sha256=digest(Path(__file__)),dependency_sha256={f:digest(Path(__file__).parent/f) for f in ['check_window_feature_dependency.py','compare_window_models.py','compare_models.py','group_stopped_mlp.py','optimize_models.py','prepare_evaluation_data.py']},
                  versions=dict(numpy=np.__version__,sklearn=sklearn.__version__),
                  limitations=['Direction is derived from reported vehicle heading, not an independent RF measurement; trust and live availability remain unresolved.',
                               'One capture with reused development folds; seeds repeat observations and do not provide independent-drive uncertainty.',
                               'Direction, position, time, receiver gain and route support are entangled; this experiment cannot identify a physical cause.',
                               'Per-direction macro recall includes all 16 labels, including S19 with zero northbound support; inspect class supports.',
                               'Feature summaries give supported passes equal weight; 50 m bins only approximately match position.',
                               'Wrong-direction predictions are an input sensitivity check, not measured spoof detection or an attack success rate.',
                               'No new dataset contract, final model, sequence model, calibration or detector threshold is selected.'])
    (OUT/FILES[0]).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (OUT/FILES[1]).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    (OUT/FILES[2]).write_text(make_html(report))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1): run()
