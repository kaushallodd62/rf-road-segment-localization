#!/usr/bin/env python3
"""Exercise the frozen methods on invented data; no field results are reproduced."""
from pathlib import Path
import sys
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'test_model'))
from compare_partial_history import partial_histories, partial_inputs, fit_fold
from compare_rnn_histories import common_histories
from test_partial_history import PartialHistoryTests


def main():
    rng = np.random.RandomState(7)
    rows, tracks, values = [], [], []
    for pass_id in range(1, 11):
        for step in range(24):
            label = step // 8
            rows.append(dict(timestamp_unix_ms=str((pass_id * 100 + step) * 1000),
                node_id='synthetic_receiver', session_id='synthetic_session',
                pass_index=str(pass_id), direction='N' if pass_id % 2 == 0 else 'S',
                cv_fold=str((pass_id + 1) // 2), segment_id=f'segment_{label}'))
            tracks.append('synthetic_vehicle')
            values.append(rng.normal(label * 3, .4, size=8))
    raw = np.array(values)
    raw[rng.uniform(size=raw.shape) < .04] = np.nan
    direction = np.array([int(r['direction'] == 'N') for r in rows])
    y = np.array([r['segment_id'] for r in rows])
    folds = np.array([int(r['cv_fold']) for r in rows])
    passes = np.array([int(r['pass_index']) for r in rows])
    windows, _, _ = partial_histories(rows, tracks)
    x = partial_inputs(raw, direction, windows)
    complete = (windows >= 0).all(1)
    q, old, _, _ = common_histories(rows, tracks)
    np.testing.assert_array_equal(np.flatnonzero(complete), q)
    np.testing.assert_array_equal(windows[q], old[:, ::-1])
    assert x.shape == (240, 82) and complete.sum() == 150
    assert np.all((x[:, -1] >= 1) & (x[:, -1] <= 10))
    seen = np.zeros(len(rows), int)
    for fold in range(1, 6):
        pipe, detail = fit_fold(x, y, folds, passes, np.ones(len(rows), bool), fold, 17)
        held = folds == fold
        probabilities = pipe.predict_proba(x[held])
        assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1)
        assert not set(detail['fit_passes']) & set(detail['held_passes'])
        seen[held] += 1
    assert np.all(seen == 1)
    checks = unittest.TestSuite(PartialHistoryTests(name) for name in [
        'test_partial_order_padding_labels_and_resets',
        'test_features_do_not_fill_from_future',
        'test_fitting_medians_exclude_held_and_ineligible_rows',
    ])
    result = unittest.TextTestRunner(verbosity=1).run(checks)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print('PASS: 240 invented observations; 150 complete histories and 90 retained early observations.')
    print('PASS: five whole-pass folds; fitting-only medians; causal input construction.')
    print('No private data were read and no field-study scores were reproduced.')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
