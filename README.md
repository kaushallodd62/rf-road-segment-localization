# RF road-segment localization

Can RF measurements help identify a vehicle's road segment? I explored this
question using a small C-V2X field capture, then investigated how travel direction,
recent measurements and uneven sampling affected the predictions.

The project is complete. The private capture is excluded. The repository contains
the original experiment code, a short account of the findings, and a synthetic
example that runs without the field data.

## What I found

Reported travel direction improved both the random forest and MLP. Adding recent
RF measurements improved the forest overall, although some short segments became
harder to recognize. The small RNN, GRU and temporal CNN recipes I tested did not
outperform the forest.

On the same 1,217 query measurements, the forest's accuracy rose from **57.66%**
with the current observation to **67.41%** with ten observations. Macro recall
rose from **41.45%** to **45.31%**. These are averages over five whole-pass
training folds and three seeds, from one capture. They are development results.

The useful part of the work was understanding the mistakes: repeated exports of
the same measurement, direction-dependent failures, short visits, and a slow pass
that the longer-history model handled poorly. Keeping observations with incomplete
history recovered data but did not improve the overall score.

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

This study evaluated road-segment prediction, an initial question relevant to
checking reported vehicle locations. It did not validate a spoofing detector.
The code and documentation were developed with AI assistance; the long working
journals remain in the private archive. No license is granted in this snapshot.
