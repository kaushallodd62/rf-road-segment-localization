# Results

All numbers below come from the original saved experiment summaries. They are
same-capture development results. Each table states its query population; scores
from different populations should not be treated as a single improvement curve.
Macro recall is the average recall across the 16 segments.

## Direction helped

These comparisons used all 1,393 distinct training windows and the same five
whole-pass folds. Values average three seeds.

| Model | Inputs | Accuracy | Macro recall |
|---|---|---:|---:|
| Forest | Eight RF fields | 46.88% | 32.33% |
| Forest | RF plus reported direction | 53.55% | 39.30% |
| MLP | Eight RF fields | 45.85% | 27.28% |
| MLP | RF plus reported direction | 54.37% | 38.44% |

Pairwise segment classifiers motivated this experiment: separating directions
changed which pairs were easy to distinguish. Pairwise discrimination was a
diagnostic, not a measurement of full 16-class localization performance.

## Longer history helped the forest overall

All rows here use the same 1,217 queries, five whole-pass training folds and
three seeds. Every model includes current reported direction. A history of ten
includes the current observation and nine earlier observations.

| Model | Observations | Accuracy | Macro recall |
|---|---:|---:|---:|
| Forest | 1 | 57.66% | 41.45% |
| Forest | 3 | 61.93% | 40.92% |
| Forest | 5 | 63.35% | 40.76% |
| Forest | 10 | 67.41% | 45.31% |
| Basic RNN | 10 | 52.01% | 31.13% |
| GRU | 10 | 52.86% | 32.11% |
| Causal temporal CNN | 10 | 49.06% | 29.27% |

The neural results apply to the tested recipes. They do not establish that every
sequence model would lose, or explain the gap by themselves. Tabular-model
benchmarks offer context: [Grinsztajn et al. (2022)](https://arxiv.org/abs/2207.08815)
and [McElfresh et al. (2023)](https://arxiv.org/abs/2305.02997). Their tasks differ
from this capture and do not prove the mechanism behind these results.

## The averages hid important failures

S13 was observed only once or twice per visit. It had zero recall at every tested
history length. S19 had eight common-query observations: current-only recall was
37.5%, while the ten-observation model always chose the segment just left.

Direction remained important after it became an input. S07 recall was 84.9%
northbound versus 4.2% southbound; S08 was 65.3% versus zero. Both had examples
from other same-direction passes. They were usually mistaken for S12 southbound.

S10 exposed a different problem. One visit lasted 27 observations, compared with
3–6 on the other visits. On that pass, current-only recall was 96.3%, versus 18.5%
with ten observations. The fitting data for that held-out pass contained no S10
example whose entire ten-observation history came from S10. That suggests a
history pattern the model had not learned to handle; it does not establish a
physical cause.

S-labels are anonymous display labels, not road-distance or adjacency codes.

## Keeping more examples did not improve the aggregate result

These rows again use the original 1,217 common queries and average three seeds.
The format-only control isolates changes caused by adding the history count.

| Forest variant | Accuracy | Macro recall |
|---|---:|---:|
| Original ten-observation model | 67.41% | 45.31% |
| Availability count added; complete histories only | 66.91% | 45.11% |
| Same format; incomplete histories retained | 66.37% | 45.25% |

Retention recovered 176 observations, with 45.27% accuracy on that additional
population. S04 accounted for 63 recovered observations and had 91.53% recall on
them. Its original southbound failures remained. S13 gained no examples, and S16
gained eleven but still had zero recall. Retention addressed a coverage problem
without resolving most classification failures.

Increasing the forest's minimum leaf size from 2 to 10 also reduced common-query
accuracy from 67.41% to 59.30% and macro recall from 45.31% to 33.14%.
The original ten-observation forest remained the research reference. Further
class weighting after retention was proposed but not run before the project ended.
