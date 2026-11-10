"""Causality, fitting boundaries, matched inputs and saved score replay."""
import json
import unittest
import numpy as np
from sklearn.metrics import accuracy_score,recall_score
import compare_partial_history as e


class PartialHistoryTests(unittest.TestCase):
    def rows(self,n=12):
        return [dict(timestamp_unix_ms=str(1000*i),node_id='n',session_id='s',pass_index='1',direction='N',cv_fold='1',segment_id='a' if i<5 else 'b') for i in range(n)]

    def test_partial_order_padding_labels_and_resets(self):
        rows=self.rows();h,_,_=e.partial_histories(rows,['t']*12)
        np.testing.assert_array_equal(h[0],[0]+[-1]*9)
        np.testing.assert_array_equal(h[2],[2,1,0]+[-1]*7)
        np.testing.assert_array_equal(h[11],np.arange(11,1,-1))
        q,old,_,_=e.common_histories(rows,['t']*12)
        np.testing.assert_array_equal(h[q],old[:,::-1])
        changed=[dict(r,segment_id='z') for r in rows]
        np.testing.assert_array_equal(e.partial_histories(changed,['t']*12)[0],h)
        for field,value in [('pass_index','2'),('direction','S'),('session_id','new')]:
            changed=[dict(r) for r in rows]
            for r in changed[6:]:r[field]=value
            hh,_,_=e.partial_histories(changed,['t']*12)
            self.assertEqual(hh[6,0],6);self.assertTrue(np.all(hh[6,1:]==-1))
        changed=[dict(r) for r in rows]
        for r in changed[6:]:r['timestamp_unix_ms']=str(int(r['timestamp_unix_ms'])+2000)
        hh,_,_=e.partial_histories(changed,['t']*12)
        self.assertTrue(np.all(hh[6,1:]==-1))

    def test_features_do_not_fill_from_future(self):
        rows=self.rows();h,_,_=e.partial_histories(rows,['t']*12)
        x=np.arange(96,dtype=float).reshape(12,8);d=np.ones(12)
        a=e.partial_inputs(x,d,h)
        self.assertEqual(a.shape,(12,82));np.testing.assert_array_equal(a[:,-1],[1,2,3,4,5,6,7,8,9,10,10,10])
        self.assertTrue(np.isnan(a[0,8:80]).all())
        xx=x.copy();xx[6:]=99999
        np.testing.assert_allclose(a[:6],e.partial_inputs(xx,d,h)[:6],equal_nan=True)
        np.testing.assert_array_equal(a[11,:80],x[np.arange(11,1,-1)].reshape(-1))

    def test_fitting_medians_exclude_held_and_ineligible_rows(self):
        x=np.tile(np.arange(6,dtype=float)[:,None],(1,82));x[0,0]=np.nan
        y=np.array(['a','b','a','b','a','b']);folds=np.array([1,1,2,2,3,3]);passes=folds.copy()
        eligible=np.array([True,True,True,False,True,True])
        pipe,detail=e.fit_fold(x,y,folds,passes,eligible,3,17)
        np.testing.assert_allclose(detail['median'][:2],[1.5,1])
        modified=x.copy();modified[3:]=999999
        other,dd=e.fit_fold(modified,y,folds,passes,eligible,3,17)
        self.assertEqual(detail,dd)
        np.testing.assert_array_equal(pipe.predict_proba(x[:3]),other.predict_proba(x[:3]))
        with self.assertRaises(AssertionError):e.fit_fold(x,y,folds,np.ones(6),eligible,3,17)

    def test_saved_metrics_and_lineage(self):
        path=e.OUT/(e.STEM+'.json')
        if not path.exists():self.skipTest('Saved replay runs after fitting')
        p=json.loads(path.read_text());rows=e.read(e.STEM+'_predictions.csv');lineage=e.read(e.STEM+'_lineage.csv')
        self.assertEqual(len(lineage),1393)
        for r in lineage:
            times=json.loads(r['history_timestamps']);n=int(r['available_history'])
            self.assertEqual(times[0],r['timestamp_unix_ms'])
            self.assertEqual(sum(t is not None for t in times),n)
            self.assertTrue(all(int(t)<=int(r['timestamp_unix_ms']) for t in times if t))
        for run in p['runs']:
            rr=[r for r in rows if r['arm']==run['arm'] and r['scope']==run['scope'] and int(r['seed'])==run['seed']]
            expected=1217 if run['scope']=='common' else 176 if run['scope']=='recovered' else 1393
            self.assertEqual(len(rr),expected);self.assertEqual(len({r['timestamp_unix_ms'] for r in rr}),expected)
            self.assertTrue(all(1<=int(r['pass_index'])<=10 for r in rr))
            y=[r['segment_id'] for r in rr];pred=[r['predicted_segment_id'] for r in rr]
            self.assertAlmostEqual(accuracy_score(y,pred),run['metrics']['accuracy'])
            self.assertAlmostEqual(recall_score(y,pred,labels=p['labels'],average='macro',zero_division=0),run['metrics']['macro_recall'])
            for row in rr:
                pp=np.array(json.loads(row['probabilities']))
                self.assertEqual(p['labels'][pp.argmax()],row['predicted_segment_id'])


if __name__=='__main__':unittest.main()
