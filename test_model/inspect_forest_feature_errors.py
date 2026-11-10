#!/usr/bin/env python3
"""Inspect paired, saved forest CV predictions; never fit a model."""
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
from compare_models import OUT, SEEDS, read

FOUR, TEN = 'Power + gain', 'All ten'


def summarize(rows):
    """Rows cover the same observations for each seed; average seed summaries."""
    result = []
    for seed in SEEDS:
        rr = [r for r in rows if r['seed'] == seed]
        assert rr
        y = np.array([r['segment_id'] for r in rr])
        p4 = np.array([r['pred4'] for r in rr]); p10 = np.array([r['pred10'] for r in rr])
        d4 = np.array([r['distance4'] for r in rr]); d10 = np.array([r['distance10'] for r in rr])
        c4 = p4 == y; c10 = p10 == y
        assert np.isfinite(d4).all() and np.isfinite(d10).all()
        counts = dict(both_correct=int((c4 & c10).sum()), only_ten_correct=int((~c4 & c10).sum()),
                      only_four_correct=int((c4 & ~c10).sum()), both_wrong=int((~c4 & ~c10).sum()))
        assert sum(counts.values()) == len(rr)
        metrics = dict(seed=seed, accuracy4=float(c4.mean()), accuracy10=float(c10.mean()), **counts,
                       both_wrong_different_predictions=int((~c4 & ~c10 & (p4 != p10)).sum()),
                       closer10=int((d10 < d4 - 1e-6).sum()), closer4=int((d4 < d10 - 1e-6).sum()),
                       equal_distance=int((np.abs(d10-d4) <= 1e-6).sum()))
        for name, d, correct in [('four', d4, c4), ('ten', d10, c10)]:
            for subset, values in [('all', d), ('wrong', d[~correct])]:
                metrics[f'{name}_{subset}_median'] = float(np.median(values)) if len(values) else None
                metrics[f'{name}_{subset}_p90'] = float(np.quantile(values, .9, method='inverted_cdf')) if len(values) else None
        result.append(metrics)
    assert len({len([r for r in rows if r['seed'] == s]) for s in SEEDS}) == 1
    means = {k: float(np.mean([v[k] for v in result])) if all(v[k] is not None for v in result) else None
             for k in result[0] if k != 'seed'}
    return dict(support=len(rows)//len(SEEDS), mean=means, seeds=result)


def run():
    report = json.loads((OUT/'forest_feature_groups.json').read_text())
    source = [r for r in read('modeling_dataset.csv') if r['split'] == 'train']
    assert hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest() == report['training_rows_sha256']
    train = {r['timestamp_unix_ms']: r for r in source}
    assert len(train) == len(source) and report['seeds'] == SEEDS
    predictions = [r for r in read('forest_feature_groups_predictions.csv') if r['group'] in (FOUR, TEN)]
    indexed = {(r['group'], int(r['seed']), r['timestamp_unix_ms']): r for r in predictions}
    assert len(indexed) == len(predictions) == 2*len(SEEDS)*len(train)
    assert set(indexed) == {(g, s, t) for g in (FOUR, TEN) for s in SEEDS for t in train}
    rows = []
    for seed in SEEDS:
        for t, original in train.items():
            a, b = (indexed[g, seed, t] for g in (FOUR, TEN))
            for r in (a, b):
                for key in ('segment_id', 'cv_fold', 'pass_index'):
                    assert r[key] == original[key]
            rows.append(dict(seed=seed, timestamp=t, segment_id=original['segment_id'],
                             pass_index=int(original['pass_index']), fold=int(original['cv_fold']), direction=original['direction'],
                             pred4=a['predicted_segment_id'], pred10=b['predicted_segment_id'],
                             distance4=float(a['distance_m']), distance10=float(b['distance_m'])))
    for run in report['runs']:
        if run['group'] in (FOUR, TEN):
            rr = [r for r in rows if r['seed'] == run['seed']]
            key = 'pred4' if run['group'] == FOUR else 'pred10'
            assert np.isclose(np.mean([r[key] == r['segment_id'] for r in rr]), run['metrics']['accuracy'])
    names = report['display_names']
    overall = summarize(rows)
    passes = []
    for p in sorted({r['pass_index'] for r in rows}):
        rr = [r for r in rows if r['pass_index'] == p]
        folds = {r['fold'] for r in rr}; directions = {r['direction'] for r in rr}
        assert len(folds) == len(directions) == 1
        passes.append(dict(pass_index=p, fold=folds.pop(), direction=directions.pop(), **summarize(rr)))
    directions = [dict(direction=d, **summarize([r for r in rows if r['direction'] == d])) for d in sorted({r['direction'] for r in rows})]
    segments = []
    cells = []
    for label in sorted({r['segment_id'] for r in rows}, key=lambda l: names[l]):
        rr = [r for r in rows if r['segment_id'] == label]
        segments.append(dict(segment_id=label, segment=names[label], **summarize(rr)))
        for p in sorted({r['pass_index'] for r in rr}):
            subset = [r for r in rr if r['pass_index'] == p]
            cells.append(dict(segment=names[label], pass_index=p, fold=subset[0]['fold'], direction=subset[0]['direction'], **summarize(subset)))
    assert sum(x['support'] for x in passes) == sum(x['support'] for x in segments) == len(train)
    assert sum(x['support'] for x in cells) == len(train)
    confusions = []
    for key in ('pred4', 'pred10'):
        counts = Counter((r['segment_id'], r[key]) for r in rows if r['segment_id'] != r[key])
        confusions.append({pair: n/len(SEEDS) for pair, n in counts.items()})
    pairs = sorted(set(confusions[0]) | set(confusions[1]), key=lambda p: -(confusions[0].get(p, 0)+confusions[1].get(p, 0)))
    confusion_rows = [dict(actual=names[a], predicted=names[b], support=sum(r['segment_id']==a for r in source),
                           four=confusions[0].get((a,b),0), ten=confusions[1].get((a,b),0)) for a,b in pairs]
    result = dict(protocol='Paired saved training-CV predictions only; no refit or validation/test scoring.',
                  seeds=SEEDS, feature_groups={g: report['feature_groups'][g] for g in (FOUR,TEN)},
                  training_rows_sha256=report['training_rows_sha256'],
                  predictions_sha256=hashlib.sha256((OUT/'forest_feature_groups_predictions.csv').read_bytes()).hexdigest(),
                  overall=overall, passes=passes, directions=directions, segments=segments, pass_segments=cells,
                  confusion_pairs=confusion_rows,
                  notes=['Counts are averages of three seed-specific counts, not pooled independent observations.',
                         'Passes are inferred heading runs; fold, direction, time, support, and fitting data are entangled.',
                         'Distances are point-to-predicted-centerline, not network distance. Wrong-only subsets differ.',
                         'S-label numbers are names only. A sparse cell below 10 rows is descriptive, not a reliable rate.',
                         'Fixed sqrt changes nominal candidates from 2 to 3; no feature effect or mechanism is isolated.'])
    (OUT/'forest_feature_error_inspection.json').write_text(json.dumps(result, indent=2)+'\n')
    render(result)
    print('Wrote paired inspection for', len(train), 'training rows and', len(cells), 'observed pass/segment cells')


def render(r):
    esc = html.escape
    parts = ['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Forest paired error inspection</title><style>body{font:15px/1.5 system-ui;background:#f6f8fb;color:#18212b;margin:0}main{max-width:1500px;margin:auto;padding:24px}table{border-collapse:collapse;background:white;width:100%;margin:14px 0}th,td{padding:7px;border:1px solid #dbe2eb;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}small{color:#526174}</style><main><h1>Four versus ten forest features: paired errors</h1><p>Same held-out training rows and same seed; no refitting. Four means RSRP, RSSI, burst power, and receiver gain. Ten is the existing reference. Each fold holds out two passes, and each fold has different fitting rows. Validation and test are unscored.</p><p>Counts are mean seed-specific counts, so may be fractional; support counts each row once. The seeds reuse observations, not independent drives. Direction is derived from BSM heading for analysis only. Pass boundaries are inferred. Cells marked * have fewer than 10 rows and should be interpreted cautiously. Accuracy within a single true segment is segment recall.</p>''']
    def table(title, groups, label):
        parts.append('<h2>'+esc(title)+'</h2><div class="scroll"><table><tr><th>Group</th><th>Rows</th><th>Four correct %</th><th>Ten correct %</th><th>Δ pp</th><th>Seed Δ range pp</th><th>Both correct</th><th>Only ten correct</th><th>Only four correct</th><th>Both wrong</th><th>Both wrong, different predictions</th></tr>')
        for x in groups:
            m=x['mean']; diffs=[(v['accuracy10']-v['accuracy4'])*100 for v in x['seeds']]
            parts.append(f'<tr><td>{esc(label(x))}{" *" if x["support"] < 10 else ""}</td><td>{x["support"]}</td><td>{m["accuracy4"]:.1%}</td><td>{m["accuracy10"]:.1%}</td><td>{(m["accuracy10"]-m["accuracy4"])*100:+.1f}</td><td>{min(diffs):+.1f} to {max(diffs):+.1f}</td>'+''.join(f'<td>{m[k]:.1f}</td>' for k in ['both_correct','only_ten_correct','only_four_correct','both_wrong','both_wrong_different_predictions'])+'</tr>')
        parts.append('</table></div>')
    table('Overall', [r['overall']], lambda x:'All training rows')
    table('By pass and fold', r['passes'], lambda x:f'Pass {x["pass_index"]} · {x["direction"]} · fold {x["fold"]}')
    table('By direction', r['directions'], lambda x:x['direction'])
    table('By true segment', r['segments'], lambda x:x['segment'])
    table('By true segment and pass', r['pass_segments'], lambda x:f'{x["segment"]} · pass {x["pass_index"]} · {x["direction"]} · fold {x["fold"]}')
    parts.append('''<h2>Geographic error by pass</h2><p>Median and nearest-rank P90 meters to the predicted centerline, calculated within each seed and then averaged. All rows and wrong-only populations are separate; wrong-only sets differ between configurations. Closer counts compare distances on the same rows; ties are within 0.000001 m. None of these measures are road-network distances.</p><div class="scroll"><table><tr><th>Pass · direction</th><th>Rows</th><th>Four all median / P90</th><th>Ten all median / P90</th><th>Four wrong median / P90</th><th>Ten wrong median / P90</th><th>Ten closer</th><th>Four closer</th><th>Equal</th></tr>''')
    for x in [dict(pass_index='All',direction='',**r['overall'])]+r['passes']:
        m=x['mean']
        def fmt(a,b):
            return '—' if m[a] is None or m[b] is None else f'{m[a]:.1f} / {m[b]:.1f}'
        parts.append(f'<tr><td>{x["pass_index"]} · {esc(x["direction"])}</td><td>{x["support"]}</td>'+''.join('<td>'+fmt(f'{model}_{sub}_median',f'{model}_{sub}_p90')+'</td>' for sub in ('all','wrong') for model in ('four','ten'))+''.join(f'<td>{m[k]:.1f}</td>' for k in ('closer10','closer4','equal_distance'))+'</tr>')
    parts.append('</table></div><h2>Confusion pairs</h2><p>Observed pairs sorted by combined frequency; counts are averaged across seeds, not independent examples. Segment numbers do not represent distance or adjacency.</p><div class="scroll"><table><tr><th>True → predicted</th><th>True support</th><th>Four wrong</th><th>Ten wrong</th><th>Ten minus four</th></tr>')
    for x in r['confusion_pairs']:
        parts.append(f'<tr><td>{esc(x["actual"])} → {esc(x["predicted"])}</td><td>{x["support"]}</td><td>{x["four"]:.1f}</td><td>{x["ten"]:.1f}</td><td>{x["ten"]-x["four"]:+.1f}</td></tr>')
    parts.append('</table></div><h2>Limits</h2><ul>'+''.join('<li>'+esc(x)+'</li>' for x in r['notes'])+'</ul><p>Full seed-specific summaries and paired cells: <a href="forest_feature_error_inspection.json">JSON</a>. The source prediction rows are in <a href="forest_feature_groups_predictions.csv">forest_feature_groups_predictions.csv</a>.</p></main></html>')
    (OUT/'forest_feature_error_inspection.html').write_text(''.join(parts))


if __name__ == '__main__':
    run()
