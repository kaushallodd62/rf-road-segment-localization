# Code guide

The 60 Python files and two requirement files in `test_model/` are unchanged copies
of the research code. Their relative paths are preserved so sibling imports keep
working. `CODE_SHA256SUMS` records the publication snapshot. The public documentation
and `demo/` are separate additions.

## Entry points

| Question or stage | Main files in `test_model/` |
|---|---|
| Data audit, map extraction and road labels | `audit_field_data.py`, `explore_spatial_data.py`, `segment_road.py` |
| Pass splits and first baseline | `prepare_evaluation_data.py`, `evaluate_weighted_knn.py` |
| Forest/MLP references and tuning | `compare_models.py`, `optimize_models.py`, `group_stopped_mlp.py` |
| Repeated publications and distinct windows | `prepare_window_dataset.py`, `compare_window_models.py` |
| Segment distinguishability and direction | `audit_segment_fingerprints.py`, `compare_direction_feature.py` |
| History and sequence models | `compare_temporal_history.py`, `compare_rnn_histories.py`, `compare_gru_cnn.py` |
| Remaining failures and coverage | `inspect_history_failures.py`, `compare_partial_history.py` |
| Leaf-size comparison with history | `compare_history_leaf_sizes.py` |

Several earlier experiments remain alongside these entry points. The flat layout
reflects the experiment history; reorganizing it into a package would change
imports and recorded source hashes. The table provides navigation without that
refactor.

## What runs without private data

Run `python demo/synthetic_demo.py` from the repository root after installing
`test_model/requirements-sequences.txt`. It calls the original history builders,
input construction and fold-fitting function on invented observations. It also
runs selected existing synthetic checks, explicitly by name.

Do not use unrestricted test discovery as the public demo. Many `test_*.py` files
load private saved predictions or replay historical evaluation workflows. Those
tests remain frozen for provenance, not as a promise of a fully runnable public
suite. No private dataset is downloaded by the demo.

## Input and artifact contracts

The history methods use dictionaries with timestamps, node/session/track grouping,
pass, direction and fold metadata. Each numeric RF vector has the eight fields
listed in [Methods](METHODS.md). Segment IDs are targets only. A partial-history
matrix has 80 RF columns, reported direction and available-history count.

Most experiment runners expect CSV/JSON intermediates under `test_model/output/`.
Some verify source and data hashes and refuse to overwrite previous artifacts.
Those files are excluded. A future adaptation to another dataset would be new
work, with its own schema and evaluation; this snapshot does not silently adapt
or refit the original models.

The old serialized ten-feature reference described in historical code is a
different model from the later history-aware forest. No serialized model is
included in this repository.
