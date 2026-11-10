#!/usr/bin/env python3
"""Create an additive, training-only window dataset and publication lineage."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from prepare_evaluation_data import FEATURES, OUTPUT_DIR, STRUCTURED, read_records

VERSION = 'rf-localization.window-dataset.v1'
FILES = ['training_windows_v1.csv', 'training_publications_v1.csv', 'window_dataset_v1.json']


def row_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def build_tables(rows, raw, points):
    """Keep earliest seen snapshot per scoped window; never infer capture time."""
    if not rows or any(r['split'] != 'train' or not 1 <= int(r['pass_index']) <= 10 for r in rows):
        raise ValueError('Only existing training passes 1 through 10 are allowed')
    ids = [r['timestamp_unix_ms'] for r in rows]
    if len(set(ids)) != len(ids) or set(raw) != set(ids) or set(points) != set(ids):
        raise ValueError('Training timestamps must be unique and exactly covered')
    groups = defaultdict(list)
    for r in sorted(rows, key=lambda r: int(r['timestamp_unix_ms'])):
        t = r['timestamp_unix_ms']; p = raw[t]
        key = tuple(p.get(k) for k in ['node_id', 'session_id', 'analyzer_window_id'])
        if not all(key) or p.get('analyzer_duration_ms', 0) <= 0:
            raise ValueError('Missing scoped window identity or valid duration')
        for f in FEATURES:
            expected = p.get(f)
            if (expected is None) != (r[f] == '') or (expected is not None and float(r[f]) != expected):
                raise ValueError(f'Feature mismatch at {t}: {f}')
        point = points[t]
        tracks = [u for u in p.get('ue_locations', []) if u.get('track_id') == point['track_id']]
        if len(tracks) != 1:
            raise ValueError(f'Ambiguous label track at {t}')
        u = tracks[0]
        if (point['matched_segment_id'] != r['segment_id'] or point['match_status'] != 'matched'
                or abs(float(point['latitude']) - u['lat']) > 1e-8
                or abs(float(point['longitude']) - u['lng']) > 1e-8):
            raise ValueError(f'Existing coordinate-derived label does not match snapshot at {t}')
        if int(t) < u['last_seen_unix_ms']:
            raise ValueError('Negative summary-to-last-seen gap requires investigation')
        groups[key].append((r, p, u))
    windows = []; publications = []
    for (node, session, window), group in groups.items():
        first, payload, track = group[0]
        t0 = int(first['timestamp_unix_ms'])
        for row, p, u in group:
            if any(row[k] != first[k] for k in ['pass_index', 'cv_fold', 'direction', 'segment_id', *FEATURES]):
                raise ValueError(f'Window changes features, label or grouping: {window}')
            if p['ue_locations'] != payload['ue_locations'] or p['analyzer_duration_ms'] != payload['analyzer_duration_ms']:
                raise ValueError(f'Window changes location snapshot or duration: {window}')
            t = int(row['timestamp_unix_ms'])
            publications.append(dict(source_summary_timestamp_unix_ms=t, node_id=node, session_id=session,
                                     analyzer_window_id=window, representative_timestamp_unix_ms=t0,
                                     pass_index=int(row['pass_index']), cv_fold=int(row['cv_fold']),
                                     is_first_seen=int(t == t0), elapsed_since_first_seen_ms=t-t0,
                                     summary_to_ue_last_seen_ms=t-u['last_seen_unix_ms']))
        # Label/provenance fields are metadata; FEATURES is the sole input allowlist.
        windows.append(dict(first, node_id=node, session_id=session, analyzer_window_id=window,
                            analyzer_duration_ms=payload['analyzer_duration_ms'],
                            ue_last_seen_unix_ms=track['last_seen_unix_ms'],
                            summary_to_ue_last_seen_ms=t0-track['last_seen_unix_ms'],
                            label_alignment_status='same_exported_snapshot_unverified',
                            physical_window_end_unix_ms='', decision_available_unix_ms=''))
    return windows, sorted(publications, key=lambda r: r['source_summary_timestamp_unix_ms'])


def coverage(rows, labels):
    def counts(part):
        return {label: sum(r['segment_id'] == label for r in part) for label in labels}
    folds = []
    for outer in range(1, 6):
        inner = outer % 5 + 1
        masks = {
            'outer_fit': [r for r in rows if int(r['cv_fold']) != outer],
            'outer_held': [r for r in rows if int(r['cv_fold']) == outer],
            'inner_fit': [r for r in rows if int(r['cv_fold']) not in (outer, inner)],
            'inner_stop': [r for r in rows if int(r['cv_fold']) == inner],
        }
        folds.append(dict(outer_fold=outer, inner_fold=inner,
                          portions={k: dict(rows=len(v), support=counts(v),
                                           missing_labels=[s for s, n in counts(v).items() if not n],
                                           passes=sorted({int(r['pass_index']) for r in v})) for k, v in masks.items()}))
    return dict(rows=len(rows), support=counts(rows),
                by_pass=dict(sorted(Counter(int(r['pass_index']) for r in rows).items())),
                by_fold=dict(sorted(Counter(int(r['cv_fold']) for r in rows).items())), folds=folds)


def run():
    targets = [OUTPUT_DIR/name for name in FILES]
    if any(p.exists() for p in targets):
        raise RuntimeError('Versioned dataset artifacts exist; refusing to overwrite')
    with (OUTPUT_DIR/'modeling_dataset.csv').open(newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == 'train']
    audit = json.loads((OUTPUT_DIR/'training_diagnostic_audit.json').read_text())
    if len(rows) != 1613 or row_hash(rows) != audit['training_rows_sha256']:
        raise ValueError('Training data differs from the audited snapshot')
    ids = {r['timestamp_unix_ms'] for r in rows}; raw = {}
    for path in sorted(STRUCTURED.glob('*.ndjson')):
        for record in read_records(path):
            t = str(record.get('timestamp_unix_ms'))
            if record.get('record_type') == 'channel_summary' and t in ids:
                if t in raw: raise ValueError(f'Duplicate source summary timestamp {t}')
                raw[t] = record['payload']
    with (OUTPUT_DIR/'location_points_segmented.csv').open(newline='') as f:
        points = {r['timestamp_unix_ms']: r for r in csv.DictReader(f) if r['timestamp_unix_ms'] in ids}
    if row_hash(raw) != audit['provenance']['raw_training_summary_sha256']:
        raise ValueError('Raw training summaries differ from the audited source')
    windows, publications = build_tables(rows, raw, points)
    if len(windows) != 1393 or {r['timestamp_unix_ms'] for r in windows} != set(audit['provenance']['representative_timestamps']):
        raise ValueError('Window representatives disagree with the prior audit')
    labels = sorted({r['segment_id'] for r in rows})
    supports = coverage(windows, labels)
    if any(f['portions'][k]['missing_labels'] for f in supports['folds'] for k in ['inner_fit', 'outer_fit']):
        raise ValueError('A fitting portion loses class coverage; review before modeling')
    for path, data in zip(targets[:2], [windows, publications]):
        with path.open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    manifest = dict(schema_version=VERSION, scope='Existing training passes 1-10 only; no validation/test or fitting.',
                    observation_key=['node_id', 'session_id', 'analyzer_window_id'],
                    representative_rule='Earliest summary timestamp observed for each scoped window in the training subset.',
                    label_rule='Reuse coordinate-derived label from the same earliest exported snapshot; require agreement across repeats.',
                    classifier_features=FEATURES, target='segment_id',
                    non_input_columns=[k for k in windows[0] if k not in FEATURES],
                    publication_columns_are_metadata_only=True,
                    timing_contract={
                        'timestamp_unix_ms': 'Earliest exported summary timestamp, not verified capture/window-end/publication time.',
                        'summary_to_ue_last_seen_ms': 'Difference between two exported timestamps; not verified physical RF age.',
                        'elapsed_since_first_seen_ms': 'Causal elapsed summary time since this window was first seen in this subset; not physical RF age.',
                        'is_first_seen': 'First seen in the training export subset, not a producer guarantee of freshness.',
                        'physical_window_end_unix_ms': 'Unknown; blank.',
                        'decision_available_unix_ms': 'Unknown; blank.',
                        'future_history': 'Publication counts, last appearances and full lineage must never become classifier inputs.'},
                    feature_availability={
                        'message_freshness_ms': 'BSM generation-to-processing-complete latency; availability and manipulation resistance remain unverified.',
                        'bsm_rate_per_sec': 'Received BSM count per represented analyzer duration; may be retained; live availability unverified.',
                        'all_ten': 'Preserved for controlled development comparison; not a verified deployment feature contract.'},
                    counts=dict(original_rows=len(rows), windows=len(windows), publications=len(publications),
                                repeated_publications=sum(not r['is_first_seen'] for r in publications),
                                first_seen_label_gap_over_1000ms=sum(r['summary_to_ue_last_seen_ms'] > 1000 for r in windows)),
                    original_support=coverage(rows, labels), window_support=supports,
                    training_rows_sha256=row_hash(rows), raw_training_summaries_sha256=row_hash(raw),
                    window_rows_sha256=row_hash(windows), publication_rows_sha256=row_hash(publications),
                    artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in targets[:2]},
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    dependency_sha256={'prepare_evaluation_data.py': hashlib.sha256((Path(__file__).parent/'prepare_evaluation_data.py').read_bytes()).hexdigest()},
                    audit_sha256=hashlib.sha256((OUTPUT_DIR/'training_diagnostic_audit.json').read_bytes()).hexdigest(),
                    limitations=['Distinct windows are correlated observations within one capture, not independent drives.',
                                 'Removing repeated publications changes the observation distribution; existing model scores do not transfer.',
                                 'No time-based relabeling, interpolation, inferred physical capture times, or freshness-based row exclusion.',
                                 'Repeated snapshots may still belong in a per-second live decision evaluation; that requires a separate contract.',
                                 'No model fit, saved-prediction rescoring, calibration, spoof detection, or final model selection.'])
    targets[2].write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest['counts'], indent=2))
    print('All 16 labels remain in each outer and inner fitting portion.')


if __name__ == '__main__':
    run()
