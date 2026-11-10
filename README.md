# RF road-segment localization

C-V2X vehicles broadcast safety messages containing their reported location, speed
and heading. A vehicle can misreport its location, which raises a question: could
the received radio signal help check where it actually is?

This project explored predicting a vehicle's road segment from receiver-side RF
measurements. The idea was to compare that estimate with the location reported
in the message. The experiments focused on how reliably the RF measurements could
distinguish segments in a small field capture.

This repository contains the original experiment code, an account of the findings,
and a synthetic example that runs without the original field data.

## How I approached it

I started with exploratory data analysis: plotting vehicle traces, checking feature
availability and correlations, and assigning road-segment labels from reported
coordinates. I grouped measurements into complete directional passes for splitting
and used distance-weighted k-NN as an initial baseline.

I then compared random forests and an MLP, exploring feature selection, forest leaf
sizes, neural-network regularization and class weighting. I also tried direct
search over neural-network weights. Alongside accuracy, I tracked recall for each
segment, confusion matrices and geographic error to see which locations the models
struggled with.

A closer audit found that some exported rows repeated the same underlying
measurement. After separating those publications from distinct RF windows, I
examined segment pairs and the effect of travel direction. That led to experiments
with recent measurement history, followed by RNN, GRU and temporal CNN comparisons.
The final analysis looked at short segment visits, uneven training coverage and
ways to retain observations with incomplete history.

## Main findings

Reported travel direction improved both the random forest and MLP. Recent RF
measurements improved the forest overall, although longer histories made some
short segments harder to recognize. The small RNN, GRU and temporal CNN recipes
I tested did not outperform the forest.

On the same 1,217 query measurements, forest accuracy rose from **57.66%** with the
current observation to **67.41%** with ten observations. Macro recall rose from
**41.45%** to **45.31%**. These are averages over five whole-pass training folds
and three seeds, from one capture. They are development results.

Failure analysis showed that the aggregate scores hid substantial differences
between segments, directions and vehicle passes. Retaining incomplete histories
recovered examples but did not improve the overall score. The
[results](docs/RESULTS.md) describe those tradeoffs and the experiments that led
to them.

These experiments used the available capture to develop the processing and
evaluation approach. The intended next phase was a larger capture with multiple
vehicles and richer RF measurements, including Doppler and channel response, to
investigate the remaining ambiguities and test generalization across vehicles and
passes. The project ended before that capture took place.

## Try the synthetic example

From the repository root, using Python 3.12:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r test_model/requirements-sequences.txt
python demo/synthetic_demo.py
```

The example imports the original methods and checks pass separation, fitting-only
imputation and causal histories. All its measurements are invented. Its output
does not reproduce or validate the field-study results.

## Read the study

- [Methods](docs/METHODS.md): observations, splits, models and history construction.
- [Results](docs/RESULTS.md): paired comparisons and the failures behind the averages.
- [Limitations](docs/LIMITATIONS.md): what this capture and evaluation can support.
- [Code guide](docs/CODE_GUIDE.md): where each stage lives and what needs private data.

## Code and data

The files in `test_model/` retain their original names, paths and contents.
`CODE_SHA256SUMS` records their hashes. The synthetic example and these documents
were added for publication. The experiment scripts expect intermediate files that
are not included, so they cannot reproduce the original field results on their own.

The code and documentation were developed with AI assistance; the long working
journals remain in the private archive. No license is granted in this snapshot.
