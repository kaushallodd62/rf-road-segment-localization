"""Explicit training-only audit checks; no model fitting or holdout evaluation."""
import hashlib
import json
import math
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from audit_training_diagnostics import BANDS, compare_distributions, segment_geometry
from compare_models import FEATURES, OUT, SEEDS, read
from optimize_models import load_split
from prepare_evaluation_data import STRUCTURED, read_records
from segment_road import EARTH_RADIUS_M, local_xy, point_to_line


class TrainingDiagnosticTests(unittest.TestCase):
    def test_shared_boundaries_exclude_outer_road_end(self):
        degree = 180/(math.pi*EARTH_RADIUS_M)
        roads = {'a': [[0, 0], [100*degree, 0]], 'b': [[100*degree, 0], [200*degree, 0]]}
        points = [dict(timestamp_unix_ms='1', longitude=95*degree, latitude=0,
                       matched_segment_id='a', centerline_offset_m=0, along_segment_m=95),
                  dict(timestamp_unix_ms='2', longitude=0, latitude=0,
                       matched_segment_id='a', centerline_offset_m=0, along_segment_m=0)]
        records, neighbors, _ = segment_geometry(points, roads, ['a', 'b'])
        self.assertEqual(neighbors, {'a': {'b'}, 'b': {'a'}})
        self.assertAlmostEqual(records[0]['boundary_distance_m'], 5)
        self.assertEqual(records[0]['band'], '0-10m')
        self.assertAlmostEqual(records[1]['boundary_distance_m'], 100)
        self.assertEqual(records[1]['band'], '>50m')
        self.assertAlmostEqual(records[0]['alternative_margin_m'], 5)

    def test_distribution_overlap_is_marginal_and_handles_constant_scale(self):
        v = compare_distributions([0, 1, 2, np.nan], [1, 2, 3], 2)
        self.assertEqual(v['a']['n'], 3)
        self.assertTrue(v['central_intervals_overlap'])
        self.assertAlmostEqual(v['median_gap_training_iqr'], .5)
        self.assertAlmostEqual(v['wasserstein_training_iqr'], .5)
        w = compare_distributions([0, 0], [1, 1], 0)
        self.assertFalse(w['central_intervals_overlap'])
        self.assertIsNone(w['median_gap_training_iqr'])

    def test_training_provenance_and_repeated_windows(self):
        r = json.loads((OUT/'training_diagnostic_audit.json').read_text())
        rows = load_split('train'); wanted = {x['timestamp_unix_ms']: x for x in rows}
        self.assertEqual(len(rows), 1613)
        self.assertEqual({int(x['pass_index']) for x in rows}, set(range(1, 11)))
        self.assertEqual(hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(), r['training_rows_sha256'])
        self.assertEqual(hashlib.sha256((Path(__file__).parent/'audit_training_diagnostics.py').read_bytes()).hexdigest(), r['code_sha256'])
        raw = {}
        for p in STRUCTURED.glob('*.ndjson'):
            for rec in read_records(p):
                t = str(rec.get('timestamp_unix_ms'))
                if rec.get('record_type') == 'channel_summary' and t in wanted:
                    self.assertNotIn(t, raw); raw[t] = rec['payload']
        self.assertEqual(set(raw), set(wanted))
        groups = defaultdict(list); present = 0; older = 0
        for t, q in raw.items():
            groups[q['analyzer_window_id']].append(t)
            self.assertAlmostEqual(q['bsm_rate_per_sec'], 1000*q['bsm_message_count']/q['analyzer_duration_ms'])
            a = q['message_freshness_samples_ms']
            if a:
                present += 1
                self.assertAlmostEqual(q['message_freshness_ms'], sum(a)/len(a), places=6)
                for aa, bb, cc in zip(a, q['bsm_generation_to_initial_iq_receive_samples_ms'], q['initial_iq_receive_to_processing_complete_samples_ms']):
                    self.assertAlmostEqual(aa, bb+cc, places=5)
            age = int(t)-q['ue_locations'][0]['last_seen_unix_ms']
            self.assertEqual(age, r['provenance']['location_ages_ms'][t]); older += age > 1000
        checks = r['provenance']['checks']
        self.assertEqual(present, checks['freshness_mean_matches'])
        self.assertEqual(older, checks['location_age_over_1000ms'])
        self.assertEqual(len(groups), checks['distinct_analyzer_windows'])
        self.assertEqual(sum(len(g)-1 for g in groups.values()), checks['extra_repeated_rows'])
        representatives = set(r['provenance']['representative_timestamps'])
        self.assertEqual(len(representatives), len(groups))
        for times in groups.values():
            self.assertEqual(len(set(times) & representatives), 1)
            self.assertEqual(len({wanted[t]['cv_fold'] for t in times}), 1)
            self.assertEqual(len({wanted[t]['pass_index'] for t in times}), 1)
            for t in times:
                self.assertEqual([raw[t].get(f) for f in FEATURES], [raw[times[0]].get(f) for f in FEATURES])
                self.assertEqual(raw[t]['ue_locations'], raw[times[0]]['ue_locations'])
        self.assertEqual(sum(r['provenance']['distinct_window_support_by_segment'].values()), len(groups))
        # Ensure the unique-window comparisons actually use unique supports.
        for c in r['unique_window_same_direction_position_pass_comparisons']:
            for side, p in [('a', c['pass_a']), ('b', c['pass_b'])]:
                selected = [g for g in r['geometry'] if g['timestamp_unix_ms'] in representatives
                            and g['segment_id'] == c['segment']
                            and int(wanted[g['timestamp_unix_ms']]['pass_index']) == p
                            and int(g['along_m']//50)*50 == c['bin_start_m']]
                self.assertEqual(len(selected), c['along_'+side]['n'])
                self.assertGreaterEqual(len(selected), 5)

    def test_saved_error_partitions_and_point_distances(self):
        r = json.loads((OUT/'training_diagnostic_audit.json').read_text())
        rows = load_split('train'); ids = [v['timestamp_unix_ms'] for v in rows]; wanted = set(ids)
        points = {v['timestamp_unix_ms']: v for v in read('location_points_segmented.csv') if v['timestamp_unix_ms'] in wanted}
        roads = {v['properties']['segment_id']: v['geometry']['coordinates'] for v in json.loads((OUT/'road_segments.geojson').read_text())['features']}
        lon0 = np.mean([float(points[t]['longitude']) for t in ids]); lat0 = np.mean([float(points[t]['latitude']) for t in ids])
        source = {'Forest leaf 2': [v for v in read('forest_leaf_scores_predictions.csv') if int(v['leaf']) == 2],
                  'MLP grouped stop': read('group_stopped_mlp_predictions.csv')}
        for name, predictions in source.items():
            for seed in SEEDS:
                pp = {v['timestamp_unix_ms']: v for v in predictions if int(v['seed']) == seed}
                self.assertEqual(set(pp), wanted)
                run = next(v for v in r['runs'] if v['model'] == name and v['seed'] == seed)
                wrong = [pp[t]['predicted_segment_id'] != row['segment_id'] for t, row in zip(ids, rows)]
                self.assertEqual(sum(wrong), run['overall']['wrong'])
                self.assertEqual(sum(run['by_boundary_band'][b]['support'] for b in BANDS), len(rows))
                self.assertEqual(sum(run['by_boundary_band'][b]['wrong'] for b in BANDS), sum(wrong))
                for band in BANDS:
                    mask = [i for i, g in enumerate(r['geometry']) if g['band'] == band]
                    self.assertEqual(sum(wrong[i] for i in mask), run['by_boundary_band'][band]['wrong'])
                for i in [0, 100, 500, 1000, 1612]:
                    t = ids[i]; q = points[t]
                    xy = local_xy(float(q['longitude']), float(q['latitude']), lon0, lat0)
                    pred = pp[t]['predicted_segment_id']
                    d, _ = point_to_line(xy, roads[pred], lon0, lat0)
                    self.assertAlmostEqual(d, float(pp[t]['distance_m']), places=5)
                    self.assertEqual(q['matched_segment_id'], r['geometry'][i]['segment_id'])


if __name__ == '__main__':
    unittest.main()
