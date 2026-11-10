import json
import unittest
import numpy as np
from inspect_history_failures import segment_visits,summarize
from compare_models import OUT,read

class FailureTests(unittest.TestCase):
    def test_visits_restart_and_keep_previous_segment(self):
        rows=[{'segment_id':s} for s in ['a','a','b','b','a','a']]
        v=segment_visits(rows,[[0,1,2,3],[4,5]])
        self.assertEqual(v[2]['age'],0);self.assertEqual(v[2]['previous_segment'],'a')
        self.assertTrue(v[3]['observed_entry']);self.assertEqual(v[3]['visit_length'],2)
        self.assertFalse(v[4]['observed_entry']);self.assertIsNone(v[4]['previous_segment'])
        self.assertEqual(v[4]['age'],0)

    def test_empty_summary_is_explicit(self):
        self.assertEqual(summarize([]),dict(windows=0,recall=None))

    def test_summary_distinguishes_windows_from_seed_predictions(self):
        rows=[dict(timestamp_unix_ms='1',correct=k==0,true_in_top3=True,
            true_probability=.2,same_direction_fit_windows=4,
            same_direction_fit_passes=2,segment_windows_in_history=1,
            prediction='b') for k in range(3)]
        result=summarize(rows)
        self.assertEqual(result['windows'],1);self.assertEqual(result['predictions'],3)
        self.assertAlmostEqual(result['recall'],1/3)
        self.assertEqual(result['confusions'][0]['count'],2)

    def test_saved_support_and_recall_replay(self):
        report=json.loads((OUT/'history_forest_failures_v1.json').read_text())
        records=read('history_forest_failures_v1_observations.csv')
        base=json.loads((OUT/'rnn_history_lengths_v1.json').read_text())
        self.assertEqual(sum(r['query_windows'] for r in report['segments']),1217)
        for seed in report['seeds']:
            rr=[r for r in records if int(r['seed'])==seed]
            self.assertEqual(len(rr),1217)
            self.assertTrue(all(1<=int(r['pass_index'])<=10 for r in rr))
            accuracy=np.mean([r['correct']=='True' for r in rr])
            recall=np.mean([np.mean([r['correct']=='True' for r in rr if r['segment']==s]) for s in base['labels']])
            old=next(r for r in base['runs'] if r['family']=='RF' and r['length']==10 and r['seed']==seed)
            np.testing.assert_allclose([accuracy,recall],[old['metrics']['accuracy'],old['metrics']['macro_recall']])

if __name__=='__main__':unittest.main()
