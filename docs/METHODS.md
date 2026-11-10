# Methods

## Observations and labels

The source was an older C-V2X capture of one vehicle. I assigned road-segment
labels by matching reported coordinates to segmented road centerlines. Coordinates
and segment IDs supplied the targets, not the RF model inputs. Direction came
from reported heading.

The initial dataset used exported summaries. A later audit found that some rows
republished the same analyzer window at a later timestamp. Within the training
portion, 1,613 publications became 1,393 distinct windows by keeping the earliest
publication for each node/session/window key and checking repeated values.
Distinct windows still have temporal dependence.

The history experiments used eight summary fields: RSRP, RSSI, burst power,
estimated CBR, frame busy ratio, receiver gain, SCI CRC failure percentage and TB
CRC failure percentage. Freshness and BSM rate were excluded from those inputs.
Doppler and channel response were unavailable.

## Evaluation

A pass was one contiguous directional transit. Passes 1–10 were assigned to
training, 11–13 to validation and 14–17 to test. Five training folds each held out
two complete passes, keeping the same pass out of fitting and scoring.

Later direction and history comparisons used those training folds only. Each fit
calculated its own replacement values for missing inputs. Neural models also used
fitting-only scaling and pass-separated stopping data. Repeated seeds were 17,
42 and 123. Macro recall gave every segment equal weight despite different counts.
The [limitations](LIMITATIONS.md) record the earlier validation/test use.

## Models and history

The work began with distance-weighted k-NN, forests and an MLP. Experiments examined
feature subsets, leaf size, regularization, class weights and direct edits to the
MLP output weights. Pairwise binary classifiers then helped identify which segment
comparisons warranted closer investigation, including comparisons by direction.

For temporal inputs, histories stayed within node, session, vehicle track, pass
and direction. They reset when the exported timestamp gap exceeded 1.5 seconds.
They never crossed folds or used future measurements. True segment changes did
not reset histories; a running system would not know those labels.

The main history comparison used the same 1,217 queries with ten available
observations, even when testing shorter histories. The forest flattened eight RF
fields per observation, current first, then appended current direction. Ten
observations therefore gave 81 columns. Its fixed recipe was 300 trees, minimum
leaf size 2, two candidate columns per split and median imputation.

A small tanh RNN, GRU and causal temporal CNN were tested with fixed recipes and
pass-separated stopping. This was a bounded comparison, not an exhaustive search.

## Preserving incomplete histories

Requiring ten observations excluded 176 early windows unevenly across segments.
A later candidate kept available observations, filled missing older slots using
fitting-column medians, and added the available-history count: 82 columns.
A control used that same format while still fitting only complete histories.

Both were scored on the original common queries. Recovered queries and the full
1,393-window population were reported separately. This prevented a change in the
scoring population from being mistaken for a model improvement.
