#!/usr/bin/env python3
"""Training-only provenance, feature distributions and saved-prediction audit."""
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from compare_models import FEATURES, OUT, SEEDS, read
from optimize_models import feature_matrix, geography, load_split
from prepare_evaluation_data import STRUCTURED, read_records
from segment_road import distance, local_xy, point_to_line

BANDS = ['0-10m', '10-25m', '25-50m', '>50m']


def distribution(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return dict(n=len(v), median=float(np.median(v)) if len(v) else None,
                q25=float(np.quantile(v, .25)) if len(v) else None,
                q75=float(np.quantile(v, .75)) if len(v) else None,
                p90=float(np.quantile(v, .9, method='inverted_cdf')) if len(v) else None,
                min=float(v.min()) if len(v) else None, max=float(v.max()) if len(v) else None)


def compare_distributions(a, b, scale):
    a = np.asarray(a); b = np.asarray(b); a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    aa = distribution(a); bb = distribution(b)
    assert len(a) and len(b)
    delta = bb['median'] - aa['median']
    return dict(a=aa, b=bb, median_delta=delta,
                median_gap_training_iqr=abs(delta)/scale if scale > 0 else None,
                wasserstein_training_iqr=float(wasserstein_distance(a, b)/scale) if scale > 0 else None,
                central_intervals_overlap=max(aa['q25'], bb['q25']) <= min(aa['q75'], bb['q75']))


def segment_geometry(point_rows, roads, labels):
    lon0 = float(np.mean([float(r['longitude']) for r in point_rows]))
    lat0 = float(np.mean([float(r['latitude']) for r in point_rows]))
    lengths = {s: sum(distance(local_xy(*a, lon0, lat0), local_xy(*b, lon0, lat0)) for a, b in zip(c, c[1:])) for s, c in roads.items()}
    shared = {s: [] for s in roads}; neighbors = {s: set() for s in roads}
    for s, coords in roads.items():
        for end, along in [(coords[0], 0), (coords[-1], lengths[s])]:
            xy = local_xy(*end, lon0, lat0)
            others = [t for t, c in roads.items() if t != s and any(distance(xy, local_xy(*e, lon0, lat0)) <= .1 for e in [c[0], c[-1]])]
            if others:
                shared[s].append(along); neighbors[s].update(others)
    records = []
    for r in point_rows:
        s = r['matched_segment_id']; xy = local_xy(float(r['longitude']), float(r['latitude']), lon0, lat0)
        d, along = point_to_line(xy, roads[s], lon0, lat0)
        all_d = {t: point_to_line(xy, coords, lon0, lat0)[0] for t, coords in roads.items()}
        alt = min((t for t in roads if t != s), key=all_d.get)
        boundary = min((abs(along-e) for e in shared[s]), default=float('inf'))
        band = BANDS[int(boundary > 10) + int(boundary > 25) + int(boundary > 50)]
        assert abs(d - float(r['centerline_offset_m'])) < .05
        assert abs(along - float(r['along_segment_m'])) < .05
        records.append(dict(timestamp_unix_ms=r['timestamp_unix_ms'], segment_id=s, along_m=along,
                            boundary_distance_m=boundary if np.isfinite(boundary) else None,
                            band=band, centerline_offset_m=d, alternative_segment=alt,
                            alternative_margin_m=all_d[alt]-d))
    return records, neighbors, lengths


def provenance(rows, points):
    wanted = {r['timestamp_unix_ms']: r for r in rows}; raw = {}; notes = set()
    for path in sorted(STRUCTURED.glob('*.ndjson')):
        for r in read_records(path):
            t = str(r.get('timestamp_unix_ms'))
            if r.get('record_type') != 'channel_summary' or t not in wanted:
                continue
            assert t not in raw; raw[t] = r['payload']
    assert set(raw) == set(wanted)
    checks = Counter(); arrays = {k: [] for k in ['freshness', 'receive_delay', 'processing_delay', 'location_age', 'duration_ms', 'freshness_count']}
    ages = {}; window_ids = []; analyzer_groups = defaultdict(list); location_groups = defaultdict(list)
    for row in rows:
        t = row['timestamp_unix_ms']; p = raw[t]
        for f in FEATURES:
            actual = p.get(f); saved = row[f]
            assert (actual is None) == (saved == '')
            if actual is not None: np.testing.assert_allclose(float(saved), actual)
        for line in p.get('analyzer_detail_lines', []):
            if line.strip().startswith('note:') and any(k in line for k in ['freshness', 'generation', 'processing_complete', 'timestamp_interarrival']): notes.add(line.strip())
        a = p.get('message_freshness_samples_ms', []); b = p.get('bsm_generation_to_initial_iq_receive_samples_ms', []); c = p.get('initial_iq_receive_to_processing_complete_samples_ms', [])
        checks['rows'] += 1; checks['freshness_missing'] += p.get('message_freshness_ms') is None
        if a:
            checks['freshness_mean_matches'] += bool(np.isclose(np.mean(a), p['message_freshness_ms']))
            checks['component_lengths_equal'] += len(a) == len(b) == len(c)
            if len(a) == len(b) == len(c): checks['sample_component_sums_match'] += bool(np.allclose(a, np.asarray(b)+np.asarray(c), atol=1e-6))
            checks['mean_component_sum_matches'] += bool(np.isclose(p['message_freshness_ms'], p['bsm_generation_to_initial_iq_receive_ms'] + p['initial_iq_receive_to_processing_complete_ms']))
            arrays['freshness'].append(p['message_freshness_ms']); arrays['receive_delay'].append(p['bsm_generation_to_initial_iq_receive_ms']); arrays['processing_delay'].append(p['initial_iq_receive_to_processing_complete_ms'])
        duration = p['analyzer_duration_ms']; assert duration > 0
        checks['rate_count_duration_matches'] += bool(np.isclose(p['bsm_rate_per_sec'], p['bsm_message_count']/(duration/1000)))
        checks['partial_windows'] += bool(p.get('analyzer_is_partial')); checks['late_windows'] += bool(p.get('analyzer_is_late'))
        checks['rows_with_dropped_events'] += p.get('analyzer_dropped_event_count', 0) > 0
        checks['rows_with_late_events'] += p.get('analyzer_late_event_count', 0) > 0
        tracks = [u for u in p.get('ue_locations', []) if u.get('track_id') == points[t]['track_id']]
        assert len(tracks) == 1
        u = tracks[0]; assert np.isclose(u['lat'], float(points[t]['latitude']), rtol=0, atol=1e-8) and np.isclose(u['lng'], float(points[t]['longitude']), rtol=0, atol=1e-8)
        age = int(t) - u['last_seen_unix_ms']; ages[t] = age
        arrays['location_age'].append(age); arrays['duration_ms'].append(duration); arrays['freshness_count'].append(len(a))
        checks['zero_bsm_rows'] += p['bsm_message_count'] == 0
        checks['location_age_over_1000ms'] += age > 1000
        checks['location_age_negative'] += age < 0
        window_ids.append(p.get('observation_window'))
        analyzer_groups[p['analyzer_window_id']].append(t)
        location_groups[(u['track_id'], u['last_seen_unix_ms'])].append((u['lat'], u['lng']))
        if age > 1000:
            checks['older_location_with_bsm'] += p['bsm_message_count'] > 0
            checks['older_location_with_freshness'] += p['message_freshness_ms'] is not None
    repeated = []
    representatives = []
    for window, times in analyzer_groups.items():
        times.sort(key=int); representatives.append(times[0])
        first = raw[times[0]]
        for t in times:
            assert all(raw[t].get(f) == first.get(f) for f in FEATURES)
            assert raw[t]['ue_locations'] == first['ue_locations']
            assert wanted[t]['segment_id'] == wanted[times[0]]['segment_id']
        if len(times) > 1:
            repeated.append(dict(window=window, timestamps=times, rows=len(times),
                                 pass_indices=sorted({int(wanted[t]['pass_index']) for t in times}),
                                 cv_folds=sorted({int(wanted[t]['cv_fold']) for t in times}),
                                 segment_id=wanted[times[0]]['segment_id']))
    checks['distinct_analyzer_windows'] = len(analyzer_groups)
    checks['repeated_window_groups'] = len(repeated)
    checks['extra_repeated_rows'] = sum(g['rows']-1 for g in repeated)
    checks['repeated_windows_cross_pass'] = sum(len(g['pass_indices']) > 1 for g in repeated)
    checks['repeated_windows_cross_fold'] = sum(len(g['cv_folds']) > 1 for g in repeated)
    checks['same_lastseen_changed_coordinate_groups'] = sum(len(set(v)) > 1 for v in location_groups.values())
    return dict(checks=dict(checks), repeated_windows=repeated, representative_timestamps=representatives,
                distinct_window_support_by_segment=dict(Counter(wanted[t]['segment_id'] for t in representatives)),
                distinct_window_support_by_pass=dict(Counter(wanted[t]['pass_index'] for t in representatives)),
                distributions={k: distribution(v) for k, v in arrays.items()},
                observation_windows=dict(Counter(window_ids)), analyzer_notes=sorted(notes), location_ages_ms=ages,
                raw_training_summary_sha256=hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest())


def error_summary(mask, y, pred, distances, labels, neighbors):
    indices = np.flatnonzero(mask); wrong = pred[mask] != y[mask]; d = distances[mask]
    return dict(support=len(indices), wrong=int(wrong.sum()), error_rate=float(wrong.mean()) if len(indices) else None,
                adjacent_wrong=sum(pred[i] != y[i] and pred[i] in neighbors[y[i]] for i in indices),
                all_distance=distribution(d), wrong_distance=distribution(d[wrong]))


def run():
    target = OUT/'training_diagnostic_audit.json'
    if target.exists(): raise RuntimeError('Audit output exists; refusing to overwrite')
    rows = load_split('train'); assert len(rows) == 1613
    ids = [r['timestamp_unix_ms'] for r in rows]; assert len(set(ids)) == len(ids)
    base = json.loads((OUT/'group_stopped_mlp.json').read_text())
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    assert digest == base['training_rows_sha256'] and FEATURES == base['features']
    X = feature_matrix(rows); y = np.array([r['segment_id'] for r in rows]); labels = np.unique(y)
    passes = np.array([int(r['pass_index']) for r in rows]); directions = np.array([r['direction'] for r in rows])
    assert set(passes) == set(range(1, 11)) and labels.tolist() == base['labels']
    points = {r['timestamp_unix_ms']: r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in set(ids)}
    assert set(points) == set(ids)
    roads = {f['properties']['segment_id']: f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
    geometry, neighbors, lengths = segment_geometry([points[t] for t in ids], roads, labels)
    geo = geography(rows, labels, ids, roads, points)
    bands = np.array([r['band'] for r in geometry]); along = np.array([r['along_m'] for r in geometry])
    margin = np.array([r['alternative_margin_m'] for r in geometry])
    prov = provenance(rows, points)
    predictions = {'Forest leaf 2': {(int(r['seed']), r['timestamp_unix_ms']): r for r in read('forest_leaf_scores_predictions.csv') if int(r['leaf']) == 2},
                   'MLP grouped stop': {(int(r['seed']), r['timestamp_unix_ms']): r for r in read('group_stopped_mlp_predictions.csv')}}
    runs = []; pair_counts = Counter()
    for name, saved in predictions.items():
        assert set(saved) == {(s, t) for s in SEEDS for t in ids}
        for seed in SEEDS:
            pp = [saved[seed, t] for t in ids]
            for row, p in zip(rows, pp):
                for k in ['segment_id', 'cv_fold', 'pass_index']: assert row[k] == p[k]
            pred = np.array([p['predicted_segment_id'] for p in pp]); prob = np.array([json.loads(p['probabilities']) for p in pp])
            assert np.array_equal(labels[prob.argmax(axis=1)], pred)
            d = geo[np.arange(len(rows)), np.searchsorted(labels, pred)]
            np.testing.assert_allclose(d, [float(p['distance_m']) for p in pp])
            wrong = pred != y
            old = next(r for r in base['runs'] if r['model'] == name and r['seed'] == seed)
            assert np.isclose(1-wrong.mean(), old['metrics']['accuracy'])
            if name == 'Forest leaf 2': pair_counts.update(tuple(sorted((a, b))) for a, b in zip(y, pred) if a != b)
            by_band = {b: error_summary(bands == b, y, pred, d, labels, neighbors) for b in BANDS}
            by_segment = {s: {b: error_summary((y == s) & (bands == b), y, pred, d, labels, neighbors) for b in BANDS} for s in labels}
            by_pass = {str(p): {b: error_summary((passes == p) & (bands == b), y, pred, d, labels, neighbors) for b in BANDS} for p in range(1, 11)}
            interior = bands == '>50m'
            runs.append(dict(model=name, seed=seed, overall=error_summary(np.ones(len(rows), bool), y, pred, d, labels, neighbors),
                             by_boundary_band=by_band, by_segment_boundary_band=by_segment, by_pass_boundary_band=by_pass,
                             interior_wrong=int((interior & wrong).sum()), interior_wrong_over100m=int((interior & wrong & (d > 100)).sum()),
                             low_margin=error_summary(margin <= 2, y, pred, d, labels, neighbors),
                             older_location=error_summary(np.array([prov['location_ages_ms'][t] > 1000 for t in ids]), y, pred, d, labels, neighbors)))
    iqrs = np.nanquantile(X, .75, axis=0) - np.nanquantile(X, .25, axis=0)
    confused = [dict(a=a, b=b, wrong_per_seed=n/len(SEEDS), support_a=int((y == a).sum()), support_b=int((y == b).sum()))
                for (a, b), n in pair_counts.most_common(5)]
    def comparisons(eligible):
        within = []; shifts = []
        for pair in confused:
            a, b = pair['a'], pair['b']
            for p in range(1, 11):
                ma = (y == a) & (passes == p) & eligible; mb = (y == b) & (passes == p) & eligible
                if min(ma.sum(), mb.sum()) < 10: continue
                fields = {}
                for j, f in enumerate(FEATURES):
                    if min(np.isfinite(X[ma, j]).sum(), np.isfinite(X[mb, j]).sum()) >= 10:
                        fields[f] = compare_distributions(X[ma, j], X[mb, j], iqrs[j])
                within.append(dict(a=a, b=b, pass_index=p, direction=rows[np.flatnonzero(ma)[0]]['direction'],
                                   support_a=int(ma.sum()), support_b=int(mb.sum()), features=fields))
        bins = np.floor(along/50).astype(int)
        for s in labels:
            for direction in ['N', 'S']:
                for bin_id in sorted(set(bins[y == s])):
                    masks = {p: (y == s) & (directions == direction) & (passes == p) & (bins == bin_id) & eligible for p in range(1, 11)}
                    supported = [p for p, m in masks.items() if m.sum() >= 5]
                    for a, b in itertools.combinations(supported, 2):
                        fields = {}
                        for j, f in enumerate(FEATURES):
                            if min(np.isfinite(X[masks[a], j]).sum(), np.isfinite(X[masks[b], j]).sum()) >= 5:
                                fields[f] = compare_distributions(X[masks[a], j], X[masks[b], j], iqrs[j])
                        shifts.append(dict(segment=s, direction=direction, bin_start_m=int(bin_id*50), pass_a=a, pass_b=b,
                                           along_a=distribution(along[masks[a]]), along_b=distribution(along[masks[b]]), features=fields))
        return within, shifts
    within, shifts = comparisons(np.ones(len(rows), bool))
    representative_ids = set(prov['representative_timestamps'])
    unique_within, unique_shifts = comparisons(np.array([t in representative_ids for t in ids]))
    output = dict(protocol='Training-only descriptive audit; no classifier fit or validation/test results.',
                  training_rows_sha256=digest, features=FEATURES, labels=labels.tolist(), display_names=base['display_names'],
                  seeds=SEEDS, feature_training_iqr=dict(zip(FEATURES, iqrs.tolist())), provenance=prov,
                  geometry=geometry, neighbors={s: sorted(v) for s, v in neighbors.items()}, segment_lengths_m=lengths,
                  runs=runs, top_forest_confusions=confused, within_pass_comparisons=within, same_direction_position_pass_comparisons=shifts,
                  unique_window_within_pass_comparisons=unique_within, unique_window_same_direction_position_pass_comparisons=unique_shifts,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_hashes={f: hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in ['group_stopped_mlp.json', 'forest_leaf_scores_predictions.csv', 'group_stopped_mlp_predictions.csv', 'road_segments.geojson']},
                  limitations=['Marginal overlap does not prove multivariate ambiguity or a performance ceiling.',
                               '50 m position bins and direction do not isolate a physical cause of between-pass differences.',
                               'Boundary bands and 2 m margin are diagnostics, not known coordinate uncertainty.',
                               'Same capture, inferred passes, sparse and dependent observations; repeated seeds add no data.',
                               'Export documents semantics but does not supply the producer source or an end-to-end availability trace.'])
    target.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print('Wrote', target, 'with', len(within), 'within-pass comparisons and', len(shifts), 'matched-position pass comparisons')
    print(json.dumps(prov['checks'], indent=2))


if __name__ == '__main__':
    run()
