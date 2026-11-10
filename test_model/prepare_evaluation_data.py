#!/usr/bin/env python3
"""Create a chronological, whole-pass split for one-second segment modeling."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STRUCTURED = ROOT / "field_test_files" / "segments" / "structured"
POINTS = HERE / "output" / "location_points_segmented.csv"
OUTPUT_DIR = HERE / "output"
FEATURES = [
    "rsrp_dbm",
    "rssi_dbm",
    "burst_power_dbm",
    "estimated_cbr_percent",
    "frame_busy_ratio_percent",
    "message_freshness_ms",
    "rx_gain_db",
    "sci_crc_failure_percent",
    "tb_crc_failure_percent",
    "bsm_rate_per_sec",
]


def read_records(path: Path):
    decoder = json.JSONDecoder()
    source = path.read_text(encoding="utf-8")
    pos = 0
    while pos < len(source):
        while pos < len(source) and source[pos].isspace():
            pos += 1
        if pos == len(source):
            break
        record, pos = decoder.raw_decode(source, pos)
        yield record


def direction_from_heading(value: str) -> str:
    heading = float(value) % 360
    return "N" if min(heading, 360 - heading) < 90 else "S"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-dir", type=Path, default=STRUCTURED)
    parser.add_argument("--points-csv", type=Path, default=POINTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--train-passes", type=int, default=10)
    parser.add_argument("--validation-passes", type=int, default=3)
    args = parser.parse_args()
    if args.train_passes < 2 or args.train_passes % 2 or args.validation_passes < 1:
        parser.error("train passes must be a positive even number; validation passes must be positive")

    summaries = [
        row
        for path in args.structured_dir.glob("*.ndjson")
        for row in read_records(path)
        if row.get("record_type") == "channel_summary"
    ]
    summaries.sort(key=lambda row: int(row["timestamp_unix_ms"]))
    if not summaries:
        raise SystemExit(f"No channel_summary records found in {args.structured_dir}")

    with args.points_csv.open(newline="", encoding="utf-8") as stream:
        points = {int(row["timestamp_unix_ms"]): row for row in csv.DictReader(stream)}

    # Match the audit's north/south heading rule and keep each contiguous run intact.
    runs: list[list[dict]] = []
    run_directions: list[str] = []
    for row in summaries:
        timestamp = int(row["timestamp_unix_ms"])
        point = points.get(timestamp)
        if point is None:
            raise SystemExit(f"No coordinate/label row matches timestamp {timestamp}")
        raw_heading = point.get("reported_heading_deg", "")
        if raw_heading == "":
            raise SystemExit(f"Missing heading at timestamp {timestamp}; pass boundary is ambiguous")
        direction = direction_from_heading(raw_heading)
        if not runs or direction != run_directions[-1]:
            runs.append([])
            run_directions.append(direction)
        runs[-1].append(row)

    test_start = args.train_passes + args.validation_passes
    if len(runs) <= test_start:
        raise SystemExit(
            f"Found {len(runs)} heading runs; need more than {test_start} "
            "to leave at least one complete run for test"
        )

    rows = []
    split_counts = Counter()
    split_labels = defaultdict(Counter)
    split_directions = defaultdict(Counter)
    cv_fold_counts = Counter()
    unmatched = 0
    for pass_index, (direction, run) in enumerate(zip(run_directions, runs), start=1):
        if pass_index <= args.train_passes:
            split = "train"
        elif pass_index <= test_start:
            split = "validation"
        else:
            split = "test"
        for record in run:
            timestamp = int(record["timestamp_unix_ms"])
            point = points[timestamp]
            label = point.get("matched_segment_id", "")
            if point.get("match_status") != "matched" or not label:
                unmatched += 1
                continue
            out = {
                "timestamp_unix_ms": timestamp,
                "timestamp_utc": datetime.fromtimestamp(
                    timestamp / 1000, timezone.utc
                ).isoformat(),
                "pass_index": pass_index,
                "direction": direction,
                "split": split,
                "cv_fold": (pass_index + 1) // 2 if split == "train" else "",
                "segment_id": label,
            }
            payload = record.get("payload", {})
            out.update({name: payload.get(name) for name in FEATURES})
            rows.append(out)
            split_counts[split] += 1
            split_labels[split][label] += 1
            split_directions[split][direction] += 1
            if split == "train":
                cv_fold_counts[(pass_index + 1) // 2] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "modeling_dataset.csv"
    fields = [
        "timestamp_unix_ms", "timestamp_utc", "pass_index", "direction",
        "split", "cv_fold", "segment_id", *FEATURES,
    ]
    with dataset_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    split_summary = {
        "policy": "chronological whole-pass split using contiguous heading runs",
        "pass_index_definition": "1-based order of contiguous direction runs after sorting one-second summaries by timestamp",
        "pass_boundary_rule": "start a new pass_index when heading classification changes between north and south",
        "heading_rule": "north if within 90 degrees of 0; otherwise south",
        "rationale": "Keep each heading run intact so adjacent one-second rows from the same run cannot appear in different splits; chronological holdout approximates evaluation on later runs.",
        "cross_validation": {
            "scope": "training passes only",
            "policy": f"{args.train_passes // 2} folds; each fold holds out two adjacent complete training passes, one northbound and one southbound",
            "folds": [
                {
                    "fold": fold,
                    "held_out_passes": [2 * fold - 1, 2 * fold],
                    "directions": {
                        str(2 * fold - 1): run_directions[2 * fold - 2],
                        str(2 * fold): run_directions[2 * fold - 1],
                    },
                    "rows": cv_fold_counts[fold],
                }
                for fold in range(1, args.train_passes // 2 + 1)
            ],
        },
        "pass_counts": {
            "train": args.train_passes,
            "validation": args.validation_passes,
            "test": len(runs) - test_start,
        },
        "split_passes": {
            "train": list(range(1, args.train_passes + 1)),
            "validation": list(range(args.train_passes + 1, test_start + 1)),
            "test": list(range(test_start + 1, len(runs) + 1)),
        },
        "rows_per_split": dict(split_counts),
        "unique_segments_per_split": {
            split: len(split_labels[split]) for split in ("train", "validation", "test")
        },
        "directions_per_split": {
            split: dict(split_directions[split]) for split in ("train", "validation", "test")
        },
        "segment_rows_per_split": {
            split: dict(split_labels[split]) for split in ("train", "validation", "test")
        },
        "excluded_unmatched_rows": unmatched,
        "features": FEATURES,
        "target": "segment_id assigned from the OBU-reported coordinate",
        "limitations": [
            "All splits come from one capture, one run, and one UE track.",
            "Validation and test measure later passes from this capture, not generalization to a new drive.",
            "Passes are inferred from heading reversals; this is a heuristic grouping.",
        ],
    }
    summary_path = args.output_dir / "evaluation_split_summary.json"
    summary_path.write_text(json.dumps(split_summary, indent=2) + "\n", encoding="utf-8")

    print(f"Heading runs: {len(runs)}; matched modeling rows: {len(rows):,}; excluded: {unmatched:,}")
    for split in ("train", "validation", "test"):
        print(
            f"{split}: {split_counts[split]:,} rows, "
            f"{len(split_labels[split])} segments, "
            f"directions {dict(split_directions[split])}"
        )
    print(f"Wrote {dataset_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
