"""Leaf comparison isolation and saved-result checks."""
import json
import unittest
import numpy as np
from sklearn.metrics import accuracy_score,recall_score
import compare_history_leaf_sizes as e


class LeafTests(unittest.TestCase):
    def test_fold_medians_and_group_isolation(self):
        x=np.array([[0,np.nan],[2,4],[4,6],[6,8],[999,999],[1000,1000]],float)
        y=np.array(['a','b','a','b','a','b']);folds=np.array([1,1,2,2,3,3]);passes=folds.copy()
        p,d=e.fit_fold(x,y,folds,passes,3,17)
        np.testing.assert_allclose(d['median'],[3,6])
        x[4:]=-999
        pp,dd=e.fit_fold(x,y,folds,passes,3,17)
        self.assertEqual(d,dd);np.testing.assert_allclose(p,pp)
        with self.assertRaises(AssertionError):e.fit_fold(x,y,folds,np.ones(6),3,17)

    def test_saved_metrics_and_paired_queries(self):
        path=e.OUT/(e.STEM+'.json')
        if not path.exists():self.skipTest('Run after experiment for saved-result replay')
        report=json.loads(path.read_text());rows=e.read(e.STEM+'_predictions.csv')
        reference={(int(r['seed']),r['timestamp_unix_ms']):r for r in e.read('rnn_history_lengths_v1_predictions.csv') if r['family']=='RF' and int(r['length'])==10}
        for run in report['runs']:
            rr=[r for r in rows if int(r['seed'])==run['seed'] and int(r['leaf'])==run['leaf']]
            self.assertEqual(len(rr),1217);self.assertEqual(len({r['timestamp_unix_ms'] for r in rr}),1217)
            self.assertTrue(all(1<=int(r['pass_index'])<=10 for r in rr))
            y=[r['segment_id'] for r in rr];pred=[r['predicted_segment_id'] for r in rr]
            self.assertAlmostEqual(accuracy_score(y,pred),run['metrics']['accuracy'])
            self.assertAlmostEqual(recall_score(y,pred,labels=report['labels'],average='macro',zero_division=0),run['metrics']['macro_recall'])
            for row in rr:
                old=reference[run['seed'],row['timestamp_unix_ms']]
                for k in ['segment_id','pass_index','cv_fold']:self.assertEqual(row[k],old[k])
                if run['leaf']==2:self.assertEqual(row['probabilities'],old['probabilities'])


if __name__=='__main__':unittest.main()
