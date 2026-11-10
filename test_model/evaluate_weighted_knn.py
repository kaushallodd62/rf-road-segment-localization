#!/usr/bin/env python3
"""Evaluate a first weighted k-NN segment fingerprint baseline by CV fold.

Only rows marked split=train are used. Validation and test rows are intentionally
not scored here so they remain available for later stages.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "output" / "modeling_dataset.csv"
DEFAULT_REPORT = HERE / "output" / "weighted_knn_cv_report.json"
DEFAULT_PREDICTIONS = HERE / "output" / "weighted_knn_cv_predictions.csv"
FEATURES = [
    "rsrp_dbm", "rssi_dbm", "burst_power_dbm", "estimated_cbr_percent",
    "frame_busy_ratio_percent", "message_freshness_ms", "rx_gain_db",
    "sci_crc_failure_percent", "tb_crc_failure_percent", "bsm_rate_per_sec",
]


def number(value: str) -> float | None:
    return float(value) if value != "" else None


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def fit_transformer(rows: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    medians, scales = {}, {}
    for feature in FEATURES:
        values = [row[feature] for row in rows if row[feature] is not None]
        if not values:
            raise ValueError(f"{feature} has no observed value in this training fold")
        medians[feature] = statistics.median(values)
        iqr = quantile(values, 0.75) - quantile(values, 0.25)
        scales[feature] = iqr if iqr > 0 else 1.0
    return medians, scales


def vector(row: dict, medians: dict[str, float], scales: dict[str, float]) -> tuple[float, ...]:
    return tuple(
        ((row[feature] if row[feature] is not None else medians[feature]) - medians[feature]) / scales[feature]
        for feature in FEATURES
    )


def predict(train: list[dict], query: dict, medians: dict[str, float], scales: dict[str, float], neighbors: int) -> tuple[str, float, float]:
    q = vector(query, medians, scales)
    ranked = []
    for row in train:
        r = vector(row, medians, scales)
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(q, r)))
        ranked.append((distance, row["segment_id"]))
    nearest = sorted(ranked)[:neighbors]
    weights = defaultdict(float)
    for distance, segment in nearest:
        weights[segment] += 1.0 / max(distance, 1e-12)
    predicted, score = max(weights.items(), key=lambda item: (item[1], item[0]))
    total_weight = sum(weights.values())
    return predicted, score / total_weight, nearest[0][0]


def macro_recall(rows: list[dict], prediction_field: str) -> float:
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["segment_id"]].append(row[prediction_field] == row["segment_id"])
    return statistics.mean(statistics.mean(values) for values in by_label.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    args = parser.parse_args()
    if args.neighbors < 1:
        parser.error("--neighbors must be positive")

    with args.dataset.open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    required = {"split", "cv_fold", "segment_id", *FEATURES}
    if not raw_rows or not required.issubset(raw_rows[0]):
        raise SystemExit("Dataset is missing required modeling columns; regenerate it first.")
    train_rows = [{**row, **{feature: number(row[feature]) for feature in FEATURES}} for row in raw_rows if row["split"] == "train"]
    if not train_rows or any(row["cv_fold"] == "" for row in train_rows):
        raise SystemExit("Training rows require a cv_fold; regenerate the evaluation dataset first.")

    folds = sorted({row["cv_fold"] for row in train_rows}, key=int)
    predictions = []
    fold_reports = []
    for fold in folds:
        held_out = [row for row in train_rows if row["cv_fold"] == fold]
        fit_rows = [row for row in train_rows if row["cv_fold"] != fold]
        if len(fit_rows) < args.neighbors:
            raise SystemExit(f"Fold {fold} has fewer than {args.neighbors} fitting rows")
        medians, scales = fit_transformer(fit_rows)
        majority = Counter(row["segment_id"] for row in fit_rows).most_common(1)[0][0]
        fold_predictions = []
        for row in held_out:
            predicted, confidence, nearest_distance = predict(fit_rows, row, medians, scales, args.neighbors)
            result = {
                "timestamp_unix_ms": row["timestamp_unix_ms"],
                "pass_index": row["pass_index"],
                "direction": row["direction"],
                "cv_fold": fold,
                "segment_id": row["segment_id"],
                "predicted_segment_id": predicted,
                "predicted_weight_share": confidence,
                "nearest_distance": nearest_distance,
                "majority_segment_id": majority,
            }
            predictions.append(result)
            fold_predictions.append(result)
        fold_reports.append({
            "fold": int(fold),
            "held_out_rows": len(fold_predictions),
            "weighted_knn_accuracy": sum(row["predicted_segment_id"] == row["segment_id"] for row in fold_predictions) / len(fold_predictions),
            "majority_baseline_accuracy": sum(row["majority_segment_id"] == row["segment_id"] for row in fold_predictions) / len(fold_predictions),
            "weighted_knn_macro_recall": macro_recall(fold_predictions, "predicted_segment_id"),
        })

    assert len(predictions) == len(train_rows), "Each training row must be held out exactly once."
    report = {
        "purpose": "First segment-fingerprint baseline; grouped CV on training passes only.",
        "method": {
            "model": "distance-weighted k-nearest neighbors",
            "neighbors": args.neighbors,
            "features": FEATURES,
            "missing_values": "per-feature training-fold median",
            "scaling": "per-feature training-fold median/IQR",
            "weight": "inverse Euclidean distance after robust scaling",
        },
        "scope_guardrail": "Validation and test rows were not scored.",
        "folds": fold_reports,
        "overall": {
            "held_out_rows": len(predictions),
            "weighted_knn_accuracy": sum(row["predicted_segment_id"] == row["segment_id"] for row in predictions) / len(predictions),
            "majority_baseline_accuracy": sum(row["majority_segment_id"] == row["segment_id"] for row in predictions) / len(predictions),
            "weighted_knn_macro_recall": macro_recall(predictions, "predicted_segment_id"),
        },
        "limitations": [
            "All folds are derived from one capture and one UE track.",
            "The model is evaluated only on clean coordinate-derived segment labels.",
            "No validation or test results have been used in this report.",
        ],
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.predictions.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Grouped CV rows: {report['overall']['held_out_rows']:,}; folds: {len(folds)}")
    print(f"Weighted k-NN accuracy: {report['overall']['weighted_knn_accuracy']:.3f}")
    print(f"Majority baseline accuracy: {report['overall']['majority_baseline_accuracy']:.3f}")
    print(f"Weighted k-NN macro recall: {report['overall']['weighted_knn_macro_recall']:.3f}")
    print(f"Wrote {args.report}")
    print(f"Wrote {args.predictions}")


if __name__ == "__main__":
    main()
