"""Development-only checks for observation units, lineage and input boundaries."""
import copy
import csv
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import prepare_window_dataset as builder
from prepare_evaluation_data import FEATURES, OUTPUT_DIR


def read_csv(name):
    with (OUTPUT_DIR/name).open(newline='') as f:
        return list(csv.DictReader(f))


class WindowDatasetTests(unittest.TestCase):
    def example(self):
        rows = [dict(timestamp_unix_ms=str(t), split='train', pass_index='1', cv_fold='1',
                     direction='S', segment_id='segment-a', **{f:'1.0' for f in FEATURES}) for t in [2000, 1000]]
        track = dict(track_id='ue', lat=45.0, lng=-93.0, last_seen_unix_ms=900)
        payload = dict(node_id='node', session_id='session', analyzer_window_id='window',
                       analyzer_duration_ms=1000, ue_locations=[track], **{f:1.0 for f in FEATURES})
        raw = {r['timestamp_unix_ms']: copy.deepcopy(payload) for r in rows}
        points = {r['timestamp_unix_ms']: dict(track_id='ue', latitude='45.0', longitude='-93.0',
                                              matched_segment_id='segment-a', match_status='matched') for r in rows}
        return rows, raw, points

    def test_earliest_snapshot_and_causal_publication_lineage(self):
        rows, raw, points = self.example()
        windows, publications = builder.build_tables(rows, raw, points)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]['timestamp_unix_ms'], '1000')
        self.assertEqual(windows[0]['summary_to_ue_last_seen_ms'], 100)
        self.assertEqual([r['is_first_seen'] for r in publications], [1, 0])
        self.assertEqual([r['elapsed_since_first_seen_ms'] for r in publications], [0, 1000])
        self.assertEqual(windows[0]['physical_window_end_unix_ms'], '')
        self.assertEqual(windows[0]['decision_available_unix_ms'], '')
        raw['2000']['session_id'] = 'another-session'
        self.assertEqual(len(builder.build_tables(rows, raw, points)[0]), 2)

    def test_reject_conflicting_snapshot_and_nontraining_rows(self):
        for field, value in [('cv_fold', '2'), ('segment_id', 'segment-b'), ('rsrp_dbm', '2.0')]:
            rows, raw, points = self.example(); rows[0][field] = value
            with self.assertRaises(ValueError): builder.build_tables(rows, raw, points)
        rows, raw, points = self.example(); raw['2000']['ue_locations'][0]['last_seen_unix_ms'] = 950
        with self.assertRaises(ValueError): builder.build_tables(rows, raw, points)
        for split, p in [('validation', '11'), ('test', '14')]:
            rows, raw, points = self.example(); rows[0]['split'] = split; rows[0]['pass_index'] = p
            with self.assertRaises(ValueError): builder.build_tables(rows, raw, points)

    def test_existing_artifacts_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp); existing = output/builder.FILES[1]; existing.write_text('preserve')
            with patch.object(builder, 'OUTPUT_DIR', output), self.assertRaises(RuntimeError): builder.run()
            self.assertEqual(existing.read_text(), 'preserve')
            self.assertEqual(len(list(output.iterdir())), 1)

    def test_saved_training_lineage_support_and_input_allowlist(self):
        original = [r for r in read_csv('modeling_dataset.csv') if r['split'] == 'train']
        source = {r['timestamp_unix_ms']: r for r in original}
        windows = read_csv(builder.FILES[0]); publications = read_csv(builder.FILES[1])
        manifest = json.loads((OUTPUT_DIR/builder.FILES[2]).read_text())
        audit = json.loads((OUTPUT_DIR/'training_diagnostic_audit.json').read_text())
        self.assertEqual(manifest['training_rows_sha256'], builder.row_hash(original))
        self.assertEqual(manifest['classifier_features'], FEATURES)
        self.assertEqual(set(windows[0])-set(FEATURES), set(manifest['non_input_columns']))
        self.assertFalse(set(FEATURES) & {'segment_id', 'analyzer_window_id', 'summary_to_ue_last_seen_ms', 'pass_index', 'cv_fold'})
        self.assertEqual(len(windows), 1393); self.assertEqual(len(publications), 1613)
        self.assertEqual(len({r['source_summary_timestamp_unix_ms'] for r in publications}), len(source))
        self.assertEqual({r['source_summary_timestamp_unix_ms'] for r in publications}, set(source))
        keys = ['node_id', 'session_id', 'analyzer_window_id']
        scoped = {tuple(r[k] for k in keys): r for r in windows}
        self.assertEqual(len(scoped), len(windows))
        self.assertEqual({r['timestamp_unix_ms'] for r in windows}, set(audit['provenance']['representative_timestamps']))
        self.assertEqual(Counter(r['segment_id'] for r in windows), Counter(audit['provenance']['distinct_window_support_by_segment']))
        for r in windows:
            old = source[r['timestamp_unix_ms']]
            self.assertEqual({k:r[k] for k in old}, old)
            self.assertEqual(r['label_alignment_status'], 'same_exported_snapshot_unverified')
        for pub in publications:
            row = scoped[tuple(pub[k] for k in keys)]; t = pub['source_summary_timestamp_unix_ms']; old = source[t]
            self.assertEqual(int(pub['elapsed_since_first_seen_ms']), int(t)-int(row['timestamp_unix_ms']))
            self.assertEqual(int(pub['is_first_seen']), int(t == row['timestamp_unix_ms']))
            for k in ['pass_index', 'cv_fold']: self.assertEqual(pub[k], row[k])
            for k in ['segment_id', *FEATURES]: self.assertEqual(old[k], row[k])
        self.assertEqual(sum(int(p['is_first_seen']) for p in publications), len(windows))
        for outer in manifest['window_support']['folds']:
            f, inner = outer['outer_fold'], outer['inner_fold']
            selections = dict(outer_fit=[r for r in windows if int(r['cv_fold']) != f],
                              outer_held=[r for r in windows if int(r['cv_fold']) == f],
                              inner_fit=[r for r in windows if int(r['cv_fold']) not in [f, inner]],
                              inner_stop=[r for r in windows if int(r['cv_fold']) == inner])
            for name, part in selections.items():
                self.assertEqual(len(part), outer['portions'][name]['rows'])
                self.assertEqual(Counter(r['segment_id'] for r in part), Counter(outer['portions'][name]['support']))
                self.assertFalse(outer['portions'][name]['missing_labels'])
            for a, b in [('inner_fit','inner_stop'), ('inner_fit','outer_held'), ('inner_stop','outer_held')]:
                self.assertFalse({r['pass_index'] for r in selections[a]} & {r['pass_index'] for r in selections[b]})
        for name, digest in manifest['artifact_sha256'].items():
            self.assertEqual(hashlib.sha256((OUTPUT_DIR/name).read_bytes()).hexdigest(), digest)
        self.assertEqual(hashlib.sha256((Path(__file__).parent/'prepare_window_dataset.py').read_bytes()).hexdigest(), manifest['code_sha256'])


if __name__ == '__main__':
    unittest.main()
