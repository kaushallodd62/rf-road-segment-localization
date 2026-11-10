#!/usr/bin/env python3
"""Training-only audit of freshness missingness and saved forest predictions."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from compare_models import OUT, SEEDS, read
from prepare_evaluation_data import STRUCTURED, read_records


def run():
    rows = [r for r in read('modeling_dataset.csv') if r['split'] == 'train']
    train = {r['timestamp_unix_ms']: r for r in rows}
    assert len(train) == len(rows)
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    assert digest == json.loads((OUT / 'forest_regularization.json').read_text())['training_rows_sha256']
    raw = {}
    for path in STRUCTURED.glob('*.ndjson'):
        for record in read_records(path):
            t = str(record.get('timestamp_unix_ms'))
            if record.get('record_type') != 'channel_summary' or t not in train:
                continue
            assert t not in raw
            raw[t] = record.get('payload', {})
    assert set(raw) == set(train)
    for t, row in train.items():
        raw_value = raw[t].get('message_freshness_ms')
        assert (raw_value is None) == (row['message_freshness_ms'] == '')
        if raw_value is not None:
            assert float(raw_value) == float(row['message_freshness_ms'])
    previous = {(int(r['leaf']), int(r['seed']), r['timestamp_unix_ms']): r
                for r in read('forest_regularization_predictions.csv') if int(r['leaf']) in (2, 10)}
    assert set(previous) == {(leaf, seed, t) for leaf in (2, 10) for seed in SEEDS for t in train}
    for (leaf, seed, t), prediction in previous.items():
        assert prediction['segment_id'] == train[t]['segment_id'] and prediction['cv_fold'] == train[t]['cv_fold']
    names = json.loads((OUT / 'group_stopped_mlp.json').read_text())['display_names']

    def measure(timestamps):
        result = {}
        for missing in (True, False):
            subset = [t for t in timestamps if (train[t]['message_freshness_ms'] == '') == missing]
            result['missing' if missing else 'present'] = {
                'rows': len(subset),
                'correct_by_leaf_and_seed': {
                    str(leaf): {str(seed): sum(previous[leaf, seed, t]['segment_id'] == previous[leaf, seed, t]['predicted_segment_id'] for t in subset)
                                for seed in SEEDS} for leaf in (2, 10)}
            }
        return result

    segments = sorted({r['segment_id'] for r in train.values()}, key=lambda label: names[label])
    overall = measure(list(train))
    by_pass = {str(p): measure([t for t, r in train.items() if int(r['pass_index']) == p]) for p in range(1, 11)}
    by_segment = {names[seg]: measure([t for t, r in train.items() if r['segment_id'] == seg]) for seg in segments}
    by_segment_pass = {names[seg]: {str(p): measure([t for t, r in train.items() if r['segment_id'] == seg and int(r['pass_index']) == p])
                                   for p in range(1, 11) if any(r['segment_id'] == seg and int(r['pass_index']) == p for r in train.values())}
                       for seg in segments}
    for group in ('missing', 'present'):
        for part in (by_pass, by_segment):
            assert sum(value[group]['rows'] for value in part.values()) == overall[group]['rows']
            for leaf in ('2', '10'):
                for seed in map(str, SEEDS):
                    assert sum(value[group]['correct_by_leaf_and_seed'][leaf][seed] for value in part.values()) == overall[group]['correct_by_leaf_and_seed'][leaf][seed]
    raw_counts = {}
    for missing in (True, False):
        ids = [t for t in train if (train[t]['message_freshness_ms'] == '') == missing]
        raw_counts['missing' if missing else 'present'] = {
            'explicit_null': sum('message_freshness_ms' in raw[t] and raw[t]['message_freshness_ms'] is None for t in ids),
            'absent_key': sum('message_freshness_ms' not in raw[t] for t in ids),
            'freshness_sample_lengths': dict(Counter(str(len(raw[t]['message_freshness_samples_ms'])) if isinstance(raw[t].get('message_freshness_samples_ms'), list) else 'not_a_list' for t in ids)),
            'bsm_message_count': dict(Counter(str(raw[t].get('bsm_message_count')) for t in ids)),
            'active_ue_tracks': dict(Counter(str(raw[t].get('active_ue_tracks')) for t in ids)),
        }
    report = {'scope': 'Existing training rows and saved out-of-fold predictions only. No fitting; validation and test are not scored.',
              'definition': 'Freshness is copied from channel_summary.payload.message_freshness_ms. Local export does not define derivation or decision-time availability.',
              'overall': overall, 'by_pass': by_pass, 'by_segment': by_segment, 'by_segment_pass': by_segment_pass,
              'raw_field_counts': raw_counts, 'seeds': SEEDS,
              'training_rows_sha256': digest,
              'forest_predictions_sha256': hashlib.sha256((OUT / 'forest_regularization_predictions.csv').read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limits': ['Pass and segment composition confound observed rates; seed predictions reuse the same rows.',
                         'The capture contains one vehicle and decoder run; timing and provenance require confirmation.',
                         'A missingness indicator could encode collection or decoding behavior; its deployment and attacker robustness are unverified.']}
    target = OUT / 'freshness_missingness_audit.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print('Audited', len(train), 'training rows:', overall['missing']['rows'], 'missing;', overall['present']['rows'], 'present')
    print('Wrote', target)


if __name__ == '__main__':
    run()
