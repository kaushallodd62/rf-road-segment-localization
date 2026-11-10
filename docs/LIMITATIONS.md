# Limitations

## One capture

The data cover one vehicle and one capture. Whole-pass folds reduce mixing of
adjacent measurements within a pass, but do not test a new vehicle, drive, route
or environment. Seeds repeat fitting on the same observations; they are not
independent field trials. Histories overlap within passes.

Some segments have very few observations or appear briefly. The reported recall
for those segments can change substantially after a small number of predictions.
Doppler and channel-response measurements were unavailable.

## Labels and timing

Location labels were derived from the vehicle's reported coordinates, and
direction from reported heading. They were not independent ground truth. No
spoofed-location examples were evaluated. The physical alignment between RF
measurement windows and reported positions was not established, and the actual
receiver geometry was uncertain.

A useful segment classifier is not by itself a validated location-spoof detector.
This study did not establish unique RF localization, calibrated acceptance
thresholds or performance against real attacks.

## Evaluation history

Validation was used for optimization, and test was evaluated before development
ended. Later development used training folds only. The published comparison
tables are those later training-fold results, not a new untouched-test evaluation.
Repeatedly using the same development folds also limits their strength as evidence
of generalization. The project ended without a new independent capture.

## Partial histories

Available-history count may encode position relative to capture or reset starts.
It cannot be assumed to generalize to different start locations. Filling missing
history with fitting medians is one representation, not a physical measurement.
Restoring early observations also changes the fitting distribution, and did not
improve the aggregate score in the fixed comparison.

## Reproducibility and sharing

The private capture, GPS/road geometry, per-observation predictions, model artifacts,
internal journals and unpublished third-party material are excluded. Field results
cannot be independently reproduced from this repository alone. The synthetic demo
checks the software methods on invented data and makes no claim about RF physics.

The original experiment code is frozen, including comments written at earlier
stages. This document records the final evaluation status. No open-source license
has been assigned to this snapshot, and no lab or mentor endorsement is implied.
