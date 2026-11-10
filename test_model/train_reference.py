#!/usr/bin/env python3
"""Recreate the frozen seed-17 reference from training rows only."""
import argparse
import csv
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from compare_models import OUT
from final_evaluate import SOURCES, digest, features, file_hash, fit_candidate


def main(output):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    selected=json.loads((OUT/'final_selection.json').read_text())
    assert selected['reference_seed']==17 and selected['fit_split']=='train'
    assert selected['code_hashes']=={n:file_hash(Path(__file__).parent/n) for n in SOURCES}
    assert selected['versions']=={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__}
    with (OUT/'modeling_dataset.csv').open(newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['split']=='train']
    assert digest(rows)==selected['training_rows_sha256']
    X=features(rows);y=np.array([r['segment_id'] for r in rows]);labels=np.unique(y)
    folds=np.array([int(r['cv_fold']) for r in rows]);passes=np.array([int(r['pass_index']) for r in rows])
    pipe,_=fit_candidate(selected['selected_model'],17,X,y,folds,passes,labels)
    joblib.dump(pipe,output)
    print(f'Wrote {selected["selected_model"]} seed 17 to {output}')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    with threadpool_limits(limits=1):main(args.out)
