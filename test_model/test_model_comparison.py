"""Checks for preprocessing isolation and saved out-of-fold experiment outputs."""
import json
import unittest
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, log_loss
from compare_models import OUT, model, read


class ComparisonChecks(unittest.TestCase):
    def test_forest_regularization_outputs(self):
        from sklearn.metrics import f1_score
        report=json.loads((OUT/'forest_regularization.json').read_text())
        data={r['timestamp_unix_ms']:r for r in read('modeling_dataset.csv') if r['split']=='train'}
        predictions=read('forest_regularization_predictions.csv')
        self.assertEqual(len(report['runs']),12)
        self.assertEqual(len(predictions),12*len(data))
        for run in report['runs']:
            subset=[r for r in predictions if int(r['leaf'])==run['leaf'] and int(r['seed'])==run['seed']]
            self.assertEqual(len(subset),len(data))
            self.assertEqual({r['timestamp_unix_ms'] for r in subset},set(data))
            for r in subset:
                original=data[r['timestamp_unix_ms']]
                self.assertEqual(r['segment_id'],original['segment_id'])
                self.assertEqual(r['cv_fold'],original['cv_fold'])
            y=[r['segment_id'] for r in subset];p=[r['predicted_segment_id'] for r in subset]
            m=run['metrics']
            self.assertAlmostEqual(accuracy_score(y,p),m['accuracy'])
            self.assertAlmostEqual(recall_score(y,p,average='macro'),m['macro_recall'])
            self.assertAlmostEqual(f1_score(y,p,average='macro'),m['macro_f1'])
            d=np.array([float(r['distance_m']) for r in subset])
            self.assertTrue(np.isfinite(d).all() and (d>=0).all())
            self.assertAlmostEqual(float(np.quantile(d,.9,method='inverted_cdf')),m['all_distance_p90'])
        for s in report['summary']:
            runs=[r for r in report['runs'] if r['leaf']==s['leaf']]
            for k in ['accuracy','macro_recall','accuracy_gap','all_distance_p90']:
                self.assertAlmostEqual(float(np.mean([r['metrics'][k] for r in runs])),s[k])

    def test_forest_feature_group_outputs(self):
        from sklearn.metrics import f1_score
        from compare_forest_features import GROUPS
        from segment_road import local_xy, point_to_line
        report=json.loads((OUT/'forest_feature_groups.json').read_text())
        data={r['timestamp_unix_ms']:r for r in read('modeling_dataset.csv') if r['split']=='train'}
        predictions=read('forest_feature_groups_predictions.csv')
        previous={(int(r['seed']),r['timestamp_unix_ms']):r['predicted_segment_id'] for r in read('model_comparison_predictions.csv') if r['model']=='RF'}
        points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in data}
        roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
        lon0=np.mean([float(points[t]['longitude']) for t in data]);lat0=np.mean([float(points[t]['latitude']) for t in data])
        self.assertEqual(report['feature_groups'],GROUPS)
        self.assertEqual(len(report['runs']),12)
        self.assertEqual(len(predictions),12*len(data))
        for run in report['runs']:
            subset=[r for r in predictions if r['group']==run['group'] and int(r['seed'])==run['seed']]
            self.assertEqual(len(subset),len(data))
            self.assertEqual({r['timestamp_unix_ms'] for r in subset},set(data))
            for r in subset:
                original=data[r['timestamp_unix_ms']]
                for key in ['segment_id','cv_fold','pass_index']:self.assertEqual(r[key],original[key])
                if run['group']=='All ten':self.assertEqual(r['predicted_segment_id'],previous[run['seed'],r['timestamp_unix_ms']])
                point=points[r['timestamp_unix_ms']]
                xy=local_xy(float(point['longitude']),float(point['latitude']),lon0,lat0)
                self.assertAlmostEqual(float(r['distance_m']),point_to_line(xy,roads[r['predicted_segment_id']],lon0,lat0)[0])
            y=np.array([r['segment_id'] for r in subset]);p=np.array([r['predicted_segment_id'] for r in subset])
            m=run['metrics'];labels=report['labels']
            for k,fn in [('accuracy',accuracy_score),('macro_recall',lambda a,b:recall_score(a,b,labels=labels,average='macro',zero_division=0)),('macro_f1',lambda a,b:f1_score(a,b,labels=labels,average='macro',zero_division=0))]:
                self.assertAlmostEqual(fn(y,p),m[k])
            np.testing.assert_allclose(recall_score(y,p,labels=labels,average=None,zero_division=0),m['per_segment_recall'])
            d=np.array([float(r['distance_m']) for r in subset]);wrong=y!=p
            self.assertEqual(int(wrong.sum()),m['wrong_count'])
            for name,values in [('all',d),('wrong',d[wrong])]:
                self.assertAlmostEqual(float(np.median(values)),m[name+'_distance_median'])
                self.assertAlmostEqual(float(np.quantile(values,.9,method='inverted_cdf')),m[name+'_distance_p90'])
            for fold in run['folds']:
                indices=np.array([int(r['cv_fold'])==fold['fold'] for r in subset])
                self.assertEqual(int(indices.sum()),fold['held_rows'])
                self.assertAlmostEqual(accuracy_score(y[indices],p[indices]),fold['held']['accuracy'])
                self.assertAlmostEqual(recall_score(y[indices],p[indices],labels=labels,average='macro',zero_division=0),fold['held']['macro_recall'])
                self.assertAlmostEqual(f1_score(y[indices],p[indices],labels=labels,average='macro',zero_division=0),fold['held']['macro_f1'])
        for label,support in zip(report['labels'],report['support']):
            self.assertEqual(sum(r['segment_id']==label for r in data.values()),support)
        for summary in report['summary']:
            runs=[r for r in report['runs'] if r['group']==summary['group']]
            for key in ['accuracy','macro_recall','macro_f1','all_distance_median','all_distance_p90','wrong_distance_median','wrong_distance_p90']:
                self.assertAlmostEqual(float(np.mean([r['metrics'][key] for r in runs])),summary[key])

    def test_paired_forest_inspection(self):
        from inspect_forest_feature_errors import summarize
        rows=[]
        for seed in [17,42,123]:
            rows.extend([
                dict(seed=seed,segment_id='a',pred4='a',pred10='a',distance4=1.,distance10=1.),
                dict(seed=seed,segment_id='a',pred4='b',pred10='a',distance4=9.,distance10=2.),
                dict(seed=seed,segment_id='a',pred4='a',pred10='b',distance4=1.,distance10=5.),
                dict(seed=seed,segment_id='a',pred4='b',pred10='c',distance4=8.,distance10=4.),
            ])
        result=summarize(rows);m=result['mean']
        self.assertEqual(result['support'],4)
        for key in ['both_correct','only_ten_correct','only_four_correct','both_wrong','both_wrong_different_predictions']:
            self.assertEqual(m[key],1)
        self.assertEqual(m['accuracy4'],.5);self.assertEqual(m['accuracy10'],.5)
        self.assertEqual(m['closer10'],2);self.assertEqual(m['closer4'],1);self.assertEqual(m['equal_distance'],1)
        self.assertEqual(m['four_all_p90'],9.);self.assertEqual(m['ten_wrong_median'],4.5)
        report=json.loads((OUT/'forest_feature_error_inspection.json').read_text())
        predictions=read('forest_feature_groups_predictions.csv')
        for seed_result in report['overall']['seeds']:
            seed=seed_result['seed']
            groups={g:{r['timestamp_unix_ms']:r for r in predictions if r['group']==g and int(r['seed'])==seed} for g in ['Power + gain','All ten']}
            four,ten=groups['Power + gain'],groups['All ten']
            self.assertEqual(set(four),set(ten))
            corrected=sum(four[t]['segment_id']!=four[t]['predicted_segment_id'] and ten[t]['segment_id']==ten[t]['predicted_segment_id'] for t in four)
            regressed=sum(four[t]['segment_id']==four[t]['predicted_segment_id'] and ten[t]['segment_id']!=ten[t]['predicted_segment_id'] for t in four)
            self.assertEqual(corrected,seed_result['only_ten_correct'])
            self.assertEqual(regressed,seed_result['only_four_correct'])
            self.assertAlmostEqual((corrected-regressed)/len(four),seed_result['accuracy10']-seed_result['accuracy4'])
        self.assertEqual(sum(x['support'] for x in report['passes']),report['overall']['support'])
        self.assertEqual(sum(x['support'] for x in report['pass_segments']),report['overall']['support'])

    def test_forest_leaf_score_outputs(self):
        report=json.loads((OUT/'forest_leaf_scores.json').read_text())
        data={r['timestamp_unix_ms']:r for r in read('modeling_dataset.csv') if r['split']=='train'}
        predictions=read('forest_leaf_scores_predictions.csv')
        previous={(int(r['leaf']),int(r['seed']),r['timestamp_unix_ms']):r for r in read('forest_regularization_predictions.csv')}
        labels=np.array(report['labels'])
        self.assertEqual(len(predictions),len(data)*12)
        for run in report['runs']:
            subset=[r for r in predictions if int(r['leaf'])==run['leaf'] and int(r['seed'])==run['seed']]
            self.assertEqual(len(subset),len(data))
            self.assertEqual({r['timestamp_unix_ms'] for r in subset},set(data))
            for r in subset:
                original=data[r['timestamp_unix_ms']]
                for key in ['segment_id','cv_fold','pass_index','direction']:self.assertEqual(r[key],original[key])
                old=previous[run['leaf'],run['seed'],r['timestamp_unix_ms']]
                self.assertEqual(r['predicted_segment_id'],old['predicted_segment_id'])
                self.assertEqual(r['distance_m'],old['distance_m'])
            y=np.array([r['segment_id'] for r in subset]); p=np.array([json.loads(r['probabilities']) for r in subset])
            self.assertTrue(np.isfinite(p).all() and (p>=0).all())
            np.testing.assert_allclose(p.sum(axis=1),1)
            np.testing.assert_array_equal(labels[p.argmax(axis=1)],[r['predicted_segment_id'] for r in subset])
            indices=np.searchsorted(labels,y);true=p[np.arange(len(y)),indices]
            np.testing.assert_allclose(true,[float(r['true_segment_score']) for r in subset])
            top3=(np.argsort(p,axis=1)[:,-3:]==indices[:,None]).any(axis=1)
            np.testing.assert_array_equal(top3,[r['in_top3']=='True' for r in subset])
            greater=(p>true[:,None]).sum(axis=1);equal=(p==true[:,None]).sum(axis=1)
            m=run['metrics']
            self.assertAlmostEqual(top3.mean(),m['top3_accuracy'])
            self.assertAlmostEqual((greater<3).mean(),m['top3_possible'])
            self.assertAlmostEqual((greater+equal<=3).mean(),m['top3_guaranteed'])
            self.assertEqual(int((true==0).sum()),m['true_score_zero'])
            self.assertAlmostEqual(log_loss(y,p,labels=labels),m['log_loss'])
            self.assertAlmostEqual(np.mean(np.sum((p-(y[:,None]==labels[None,:]))**2,axis=1)),m['multiclass_brier'])
            self.assertEqual(sum(b['count'] for b in run['confidence_bins']),len(data))
            for fold in run['folds']:
                mask=np.array([int(r['cv_fold'])==fold['fold'] for r in subset])
                self.assertEqual(int(mask.sum()),fold['held_rows'])
                self.assertAlmostEqual(top3[mask].mean(),fold['metrics']['top3_accuracy'])
                self.assertAlmostEqual(log_loss(y[mask],p[mask],labels=labels),fold['metrics']['log_loss'])
        for pair in report['paired_2_vs_10']:
            two={r['timestamp_unix_ms']:r for r in predictions if int(r['leaf'])==2 and int(r['seed'])==pair['seed']}
            ten={r['timestamp_unix_ms']:r for r in predictions if int(r['leaf'])==10 and int(r['seed'])==pair['seed']}
            self.assertEqual(set(two),set(ten))
            self.assertEqual(sum(two[t]['in_top3']=='False' and ten[t]['in_top3']=='True' for t in two),pair['two_out_ten_in'])
            self.assertEqual(sum(two[t]['in_top3']=='True' and ten[t]['in_top3']=='False' for t in two),pair['two_in_ten_out'])
            self.assertEqual(sum(two[t]['in_top3']=='False' and ten[t]['in_top3']=='False' for t in two),pair['both_out'])
            zeros=[t for t in two if float(two[t]['true_segment_score'])==0]
            self.assertEqual(len(zeros),pair['two_zero_count'])
            self.assertAlmostEqual(np.median([float(ten[t]['true_segment_score']) for t in zeros]),pair['two_zero_ten_score_median'])
        for s in report['summary']:
            rr=[v for v in report['runs'] if v['leaf']==s['leaf']]
            for k in ['top3_accuracy','log_loss','multiclass_brier','true_score_zero']:
                self.assertAlmostEqual(np.mean([v['metrics'][k] for v in rr]),s[k])

    def test_group_stopped_mlp_outputs(self):
        from sklearn.metrics import f1_score
        from group_stopped_mlp import choose_epoch, PATIENCE, MAX_EPOCHS
        from segment_road import local_xy, point_to_line
        self.assertEqual(choose_epoch([1.,.9,.91,.92],patience=2),(2,4))
        self.assertEqual(choose_epoch([1.,.99995,.99991,1.],patience=3,delta=.0001),(1,4))
        report=json.loads((OUT/'group_stopped_mlp.json').read_text())
        train={r['timestamp_unix_ms']:r for r in read('modeling_dataset.csv') if r['split']=='train'}
        points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in train}
        roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
        lon0=np.mean([float(points[t]['longitude']) for t in train]);lat0=np.mean([float(points[t]['latitude']) for t in train])
        labels=np.array(report['labels']);predictions=read('group_stopped_mlp_predictions.csv')
        self.assertEqual(len(predictions),3*len(train));self.assertEqual(len(report['runs']),12)
        for run in [v for v in report['runs'] if v['model']=='MLP grouped stop']:
            subset=[r for r in predictions if int(r['seed'])==run['seed']]
            self.assertEqual(len(subset),len(train));self.assertEqual({r['timestamp_unix_ms'] for r in subset},set(train))
            y=np.array([r['segment_id'] for r in subset]);p=np.array([json.loads(r['probabilities']) for r in subset])
            self.assertTrue(np.isfinite(p).all() and (p>=0).all());np.testing.assert_allclose(p.sum(axis=1),1)
            pred=labels[p.argmax(axis=1)];np.testing.assert_array_equal(pred,[r['predicted_segment_id'] for r in subset])
            for row in subset:
                original=train[row['timestamp_unix_ms']]
                for k in ['segment_id','cv_fold','pass_index','direction']:self.assertEqual(row[k],original[k])
                point=points[row['timestamp_unix_ms']];xy=local_xy(float(point['longitude']),float(point['latitude']),lon0,lat0)
                self.assertAlmostEqual(float(row['distance_m']),point_to_line(xy,roads[row['predicted_segment_id']],lon0,lat0)[0])
            m=run['metrics'];indices=np.searchsorted(labels,y)
            self.assertAlmostEqual(accuracy_score(y,pred),m['accuracy'])
            self.assertAlmostEqual(recall_score(y,pred,labels=labels,average='macro',zero_division=0),m['macro_recall'])
            self.assertAlmostEqual(f1_score(y,pred,labels=labels,average='macro',zero_division=0),m['macro_f1'])
            self.assertAlmostEqual(log_loss(y,p,labels=labels),m['log_loss'])
            self.assertAlmostEqual(np.mean(np.sum((p-(y[:,None]==labels[None,:]))**2,axis=1)),m['multiclass_brier'])
            self.assertAlmostEqual((np.argsort(p,axis=1)[:,-3:]==indices[:,None]).any(axis=1).mean(),m['top3_accuracy'])
            for segment in run['segments']:
                mask=y==segment['segment_id']
                self.assertEqual(int(mask.sum()),segment['support'])
                self.assertAlmostEqual(float((pred[mask]==y[mask]).mean()),segment['recall'])
                self.assertAlmostEqual(log_loss(y[mask],p[mask],labels=labels),segment['log_loss'])
            for d in run['stopping']:
                outer=d['fold'];inner=outer%5+1
                self.assertEqual(d['inner_fold'],inner)
                inner_fit=[r for r in train.values() if int(r['cv_fold']) not in (outer,inner)]
                inner_held=[r for r in train.values() if int(r['cv_fold'])==inner]
                outer_held=[r for r in train.values() if int(r['cv_fold'])==outer]
                for key,rows in [('inner_fit_passes',inner_fit),('inner_held_passes',inner_held),('outer_held_passes',outer_held)]:
                    self.assertEqual(set(d[key]),{int(r['pass_index']) for r in rows})
                self.assertFalse(set(d['inner_fit_passes'])&set(d['inner_held_passes']))
                self.assertFalse((set(d['inner_fit_passes'])|set(d['inner_held_passes']))&set(d['outer_held_passes']))
                self.assertEqual(len(inner_fit),d['inner_fit_rows']);self.assertEqual(len(inner_held),d['inner_held_rows'])
                self.assertEqual(len(outer_held),d['outer_held_rows'])
                best,stop=choose_epoch(d['inner_log_loss'])
                self.assertEqual((best,stop),(d['selected_epoch'],d['stopped_epoch']))
                self.assertTrue(stop==MAX_EPOCHS or stop-best>=PATIENCE)
        baseline=json.loads((OUT/'model_comparison.json').read_text())
        forest=json.loads((OUT/'forest_leaf_scores.json').read_text())
        for run in report['runs']:
            if run['model']=='MLP original':
                original=next(v for v in baseline['runs'] if v['model']=='MLP' and v['seed']==run['seed'])
            elif run['model'].startswith('Forest'):
                original=next(v for v in forest['runs'] if v['leaf']==int(run['model'].split()[-1]) and v['seed']==run['seed'])
            else:continue
            for k in ['accuracy','macro_recall','macro_f1','top3_accuracy','log_loss','multiclass_brier']:
                self.assertAlmostEqual(run['metrics'][k],original['metrics'][k])
        for s in report['summary']:
            rr=[v for v in report['runs'] if v['model']==s['model']]
            for k in ['accuracy','macro_recall','log_loss','multiclass_brier','all_distance_p90']:
                self.assertAlmostEqual(np.mean([v['metrics'][k] for v in rr]),s[k])

    def test_weighted_group_mlp_outputs(self):
        from sklearn.metrics import f1_score
        from weighted_group_mlp import fitting_weights
        from group_stopped_mlp import choose_epoch, MAX_EPOCHS, PATIENCE
        from segment_road import local_xy, point_to_line
        from compare_models import FEATURES
        w,counts,cw=fitting_weights(np.array(['a','a','a','b']),np.array(['a','b']))
        np.testing.assert_array_equal(counts,[3,1]);np.testing.assert_allclose(cw,[2/3,2]);np.testing.assert_allclose(w,[2/3,2/3,2/3,2])
        report=json.loads((OUT/'weighted_group_mlp.json').read_text())
        baseline=json.loads((OUT/'group_stopped_mlp.json').read_text())
        train={r['timestamp_unix_ms']:r for r in read('modeling_dataset.csv') if r['split']=='train'}
        points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in train}
        roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
        lon0=np.mean([float(points[t]['longitude']) for t in train]);lat0=np.mean([float(points[t]['latitude']) for t in train])
        labels=np.array(report['labels']);predictions=read('weighted_group_mlp_predictions.csv')
        prior={(int(r['seed']),r['timestamp_unix_ms']):r for r in read('group_stopped_mlp_predictions.csv')}
        self.assertEqual(len(predictions),3*len(train));self.assertEqual(len(report['runs']),12)
        for run in [x for x in report['runs'] if x['model']=='MLP weighted']:
            subset=[r for r in predictions if int(r['seed'])==run['seed']]
            self.assertEqual(len(subset),len(train));self.assertEqual({r['timestamp_unix_ms'] for r in subset},set(train))
            y=np.array([r['segment_id'] for r in subset]);p=np.array([json.loads(r['probabilities']) for r in subset])
            self.assertTrue(np.isfinite(p).all() and (p>=0).all());np.testing.assert_allclose(p.sum(axis=1),1)
            pred=labels[p.argmax(axis=1)];np.testing.assert_array_equal(pred,[r['predicted_segment_id'] for r in subset])
            for row in subset:
                original=train[row['timestamp_unix_ms']]
                for k in ['segment_id','cv_fold','pass_index','direction']:self.assertEqual(row[k],original[k])
                point=points[row['timestamp_unix_ms']];xy=local_xy(float(point['longitude']),float(point['latitude']),lon0,lat0)
                self.assertAlmostEqual(float(row['distance_m']),point_to_line(xy,roads[row['predicted_segment_id']],lon0,lat0)[0])
            m=run['metrics'];indices=np.searchsorted(labels,y)
            self.assertAlmostEqual(accuracy_score(y,pred),m['accuracy'])
            self.assertAlmostEqual(recall_score(y,pred,labels=labels,average='macro',zero_division=0),m['macro_recall'])
            self.assertAlmostEqual(f1_score(y,pred,labels=labels,average='macro',zero_division=0),m['macro_f1'])
            self.assertAlmostEqual(log_loss(y,p,labels=labels),m['log_loss'])
            self.assertAlmostEqual(np.mean(np.sum((p-(y[:,None]==labels[None,:]))**2,axis=1)),m['multiclass_brier'])
            self.assertAlmostEqual((np.argsort(p,axis=1)[:,-3:]==indices[:,None]).any(axis=1).mean(),m['top3_accuracy'])
            for segment in run['segments']:
                mask=y==segment['segment_id'];self.assertEqual(int(mask.sum()),segment['support'])
                self.assertAlmostEqual(float((pred[mask]==y[mask]).mean()),segment['recall'])
                self.assertAlmostEqual(log_loss(y[mask],p[mask],labels=labels),segment['log_loss'])
            for d in run['stopping']:
                outer=d['fold'];inner=outer%5+1;self.assertEqual(d['inner_fold'],inner)
                inner_fit=[r for r in train.values() if int(r['cv_fold']) not in (outer,inner)]
                inner_held=[r for r in train.values() if int(r['cv_fold'])==inner]
                outer_fit=[r for r in train.values() if int(r['cv_fold'])!=outer]
                outer_held=[r for r in train.values() if int(r['cv_fold'])==outer]
                for key,rows in [('inner_fit_passes',inner_fit),('inner_held_passes',inner_held),('outer_held_passes',outer_held)]:
                    self.assertEqual(set(d[key]),{int(r['pass_index']) for r in rows})
                self.assertFalse(set(d['inner_fit_passes'])&set(d['inner_held_passes']))
                self.assertFalse((set(d['inner_fit_passes'])|set(d['inner_held_passes']))&set(d['outer_held_passes']))
                for key,rows in [('inner_preprocessing_and_weights',inner_fit),('outer_preprocessing_and_weights',outer_fit)]:
                    meta=d[key];counts=np.array([sum(r['segment_id']==label for r in rows) for label in labels])
                    np.testing.assert_array_equal(meta['counts'],counts)
                    np.testing.assert_allclose(meta['class_weights'],len(rows)/(len(labels)*counts))
                    xx=np.array([[float(r[f]) if r[f] else np.nan for f in FEATURES] for r in rows])
                    med=np.nanmedian(xx,axis=0);np.testing.assert_allclose(meta['median'],med)
                    np.testing.assert_allclose(meta['scale_mean'],np.where(np.isnan(xx),med,xx).mean(axis=0))
                best,stop=choose_epoch(d['inner_unweighted_log_loss'])
                self.assertEqual((best,stop),(d['selected_epoch'],d['stopped_epoch']))
                self.assertTrue(stop==MAX_EPOCHS or stop-best>=PATIENCE)
            pair=next(x for x in report['paired'] if x['seed']==run['seed'])
            self.assertEqual(sum(r['predicted_segment_id']==r['segment_id'] and prior[run['seed'],r['timestamp_unix_ms']]['predicted_segment_id']!=r['segment_id'] for r in subset),pair['only_weighted_correct'])
            self.assertEqual(sum(r['predicted_segment_id']!=r['segment_id'] and prior[run['seed'],r['timestamp_unix_ms']]['predicted_segment_id']==r['segment_id'] for r in subset),pair['only_unweighted_correct'])
        for run in [v for v in report['runs'] if v['model']!='MLP weighted']:
            previous=next(x for x in baseline['runs'] if x['model']==run['model'] and x['seed']==run['seed'])
            self.assertEqual(run['metrics'],previous['metrics'])
        for summary in report['summary']:
            rr=[v for v in report['runs'] if v['model']==summary['model']]
            for k in ['accuracy','macro_recall','log_loss','multiclass_brier','all_distance_p90']:
                self.assertAlmostEqual(np.mean([v['metrics'][k] for v in rr]),summary[k])

    def test_sqrt_group_mlp_outputs(self):
        from sklearn.metrics import f1_score
        from sqrt_group_mlp import fitting_weights
        from group_stopped_mlp import choose_epoch, MAX_EPOCHS, PATIENCE
        from segment_road import local_xy, point_to_line
        from compare_models import FEATURES
        w,counts,cw=fitting_weights(np.array(['a','a','a','b']),np.array(['a','b']))
        np.testing.assert_array_equal(counts,[3,1]);np.testing.assert_allclose(w.mean(),1);self.assertAlmostEqual(cw[1]/cw[0],np.sqrt(3))
        report=json.loads((OUT/'sqrt_group_mlp.json').read_text())
        baseline=json.loads((OUT/'group_stopped_mlp.json').read_text())
        train={r['timestamp_unix_ms']:r for r in read('modeling_dataset.csv') if r['split']=='train'}
        points={r['timestamp_unix_ms']:r for r in read('location_points_segmented.csv') if r['timestamp_unix_ms'] in train}
        roads={f['properties']['segment_id']:f['geometry']['coordinates'] for f in json.loads((OUT/'road_segments.geojson').read_text())['features']}
        lon0=np.mean([float(points[t]['longitude']) for t in train]);lat0=np.mean([float(points[t]['latitude']) for t in train])
        labels=np.array(report['labels']);predictions=read('sqrt_group_mlp_predictions.csv')
        prior={(int(r['seed']),r['timestamp_unix_ms']):r for r in read('group_stopped_mlp_predictions.csv')}
        self.assertEqual(len(predictions),3*len(train));self.assertEqual(len(report['runs']),15)
        for run in [x for x in report['runs'] if x['model']=='MLP sqrt']:
            subset=[r for r in predictions if int(r['seed'])==run['seed']]
            self.assertEqual(len(subset),len(train));self.assertEqual({r['timestamp_unix_ms'] for r in subset},set(train))
            y=np.array([r['segment_id'] for r in subset]);p=np.array([json.loads(r['probabilities']) for r in subset])
            self.assertTrue(np.isfinite(p).all() and (p>=0).all());np.testing.assert_allclose(p.sum(axis=1),1)
            pred=labels[p.argmax(axis=1)];np.testing.assert_array_equal(pred,[r['predicted_segment_id'] for r in subset])
            for row in subset:
                original=train[row['timestamp_unix_ms']]
                for k in ['segment_id','cv_fold','pass_index','direction']:self.assertEqual(row[k],original[k])
                point=points[row['timestamp_unix_ms']];xy=local_xy(float(point['longitude']),float(point['latitude']),lon0,lat0)
                self.assertAlmostEqual(float(row['distance_m']),point_to_line(xy,roads[row['predicted_segment_id']],lon0,lat0)[0])
            m=run['metrics'];indices=np.searchsorted(labels,y)
            self.assertAlmostEqual(accuracy_score(y,pred),m['accuracy'])
            self.assertAlmostEqual(recall_score(y,pred,labels=labels,average='macro',zero_division=0),m['macro_recall'])
            self.assertAlmostEqual(f1_score(y,pred,labels=labels,average='macro',zero_division=0),m['macro_f1'])
            self.assertAlmostEqual(log_loss(y,p,labels=labels),m['log_loss'])
            self.assertAlmostEqual(np.mean(np.sum((p-(y[:,None]==labels[None,:]))**2,axis=1)),m['multiclass_brier'])
            self.assertAlmostEqual((np.argsort(p,axis=1)[:,-3:]==indices[:,None]).any(axis=1).mean(),m['top3_accuracy'])
            for segment in run['segments']:
                mask=y==segment['segment_id'];self.assertEqual(int(mask.sum()),segment['support'])
                self.assertAlmostEqual(float((pred[mask]==y[mask]).mean()),segment['recall'])
                self.assertAlmostEqual(log_loss(y[mask],p[mask],labels=labels),segment['log_loss'])
            for d in run['stopping']:
                outer=d['fold'];inner=outer%5+1;self.assertEqual(d['inner_fold'],inner)
                inner_fit=[r for r in train.values() if int(r['cv_fold']) not in (outer,inner)]
                inner_held=[r for r in train.values() if int(r['cv_fold'])==inner]
                outer_fit=[r for r in train.values() if int(r['cv_fold'])!=outer]
                outer_held=[r for r in train.values() if int(r['cv_fold'])==outer]
                for key,rows in [('inner_fit_passes',inner_fit),('inner_held_passes',inner_held),('outer_held_passes',outer_held)]:
                    self.assertEqual(set(d[key]),{int(r['pass_index']) for r in rows})
                self.assertFalse(set(d['inner_fit_passes'])&set(d['inner_held_passes']))
                self.assertFalse((set(d['inner_fit_passes'])|set(d['inner_held_passes']))&set(d['outer_held_passes']))
                for key,rows in [('inner_preprocessing_and_weights',inner_fit),('outer_preprocessing_and_weights',outer_fit)]:
                    meta=d[key];counts=np.array([sum(r['segment_id']==label for r in rows) for label in labels])
                    np.testing.assert_array_equal(meta['counts'],counts)
                    raw=1/np.sqrt(counts);np.testing.assert_allclose(meta['class_weights'],raw/(np.sum(raw*counts)/len(rows)))
                    xx=np.array([[float(r[f]) if r[f] else np.nan for f in FEATURES] for r in rows])
                    med=np.nanmedian(xx,axis=0);np.testing.assert_allclose(meta['median'],med)
                    np.testing.assert_allclose(meta['scale_mean'],np.where(np.isnan(xx),med,xx).mean(axis=0))
                best,stop=choose_epoch(d['inner_unweighted_log_loss'])
                self.assertEqual((best,stop),(d['selected_epoch'],d['stopped_epoch']))
                self.assertTrue(stop==MAX_EPOCHS or stop-best>=PATIENCE)
            pair=next(x for x in report['paired'] if x['seed']==run['seed'])
            self.assertEqual(sum(r['predicted_segment_id']==r['segment_id'] and prior[run['seed'],r['timestamp_unix_ms']]['predicted_segment_id']!=r['segment_id'] for r in subset),pair['only_weighted_correct'])
            self.assertEqual(sum(r['predicted_segment_id']!=r['segment_id'] and prior[run['seed'],r['timestamp_unix_ms']]['predicted_segment_id']==r['segment_id'] for r in subset),pair['only_unweighted_correct'])
        full=json.loads((OUT/'weighted_group_mlp.json').read_text())
        for run in [v for v in report['runs'] if v['model']!='MLP sqrt']:
            source=full if run['model']=='MLP full inverse' else baseline
            old_name='MLP weighted' if run['model']=='MLP full inverse' else run['model']
            previous=next(x for x in source['runs'] if x['model']==old_name and x['seed']==run['seed'])
            self.assertEqual(run['metrics'],previous['metrics'])
        for summary in report['summary']:
            rr=[v for v in report['runs'] if v['model']==summary['model']]
            for k in ['accuracy','macro_recall','log_loss','multiclass_brier','all_distance_p90']:
                self.assertAlmostEqual(np.mean([v['metrics'][k] for v in rr]),summary[k])

    def test_preprocessing_uses_fitting_values_only(self):
        p = model('MLP',17)[:-1]
        fit=np.array([[1.,np.nan],[3.,10.],[5.,20.]])
        p.fit(fit)
        before=p[0].statistics_.copy()
        p.transform([[100000.,np.nan]])
        np.testing.assert_allclose(before,[3.,15.])
        np.testing.assert_array_equal(p[0].statistics_,before)
        np.testing.assert_allclose(p[1].mean_,[3.,15.])

    def test_saved_results_match_predictions(self):
        report=json.loads((OUT/'model_comparison.json').read_text())
        data={r['timestamp_unix_ms']:r for r in read('modeling_dataset.csv') if r['split']=='train'}
        rows=read('model_comparison_predictions.csv'); labels=report['labels']
        self.assertEqual(len(report['runs']),6)
        self.assertEqual(len(rows),6*len(data))
        for run in report['runs']:
            subset=[r for r in rows if r['model']==run['model'] and int(r['seed'])==run['seed']]
            self.assertEqual(len(subset),len(data))
            self.assertEqual({r['timestamp_unix_ms'] for r in subset},set(data))
            for r in subset:
                original=data[r['timestamp_unix_ms']]
                for key in ['segment_id','cv_fold','pass_index','direction']:self.assertEqual(r[key],original[key])
            y=[r['segment_id'] for r in subset];pred=[r['predicted_segment_id'] for r in subset]
            probs=np.array([json.loads(r['probabilities']) for r in subset])
            self.assertTrue(np.isfinite(probs).all());self.assertTrue((probs>=0).all())
            np.testing.assert_allclose(probs.sum(axis=1),1)
            np.testing.assert_array_equal(np.array(labels)[probs.argmax(axis=1)],pred)
            m=run['metrics']
            self.assertAlmostEqual(accuracy_score(y,pred),m['accuracy'])
            self.assertAlmostEqual(recall_score(y,pred,labels=labels,average='macro'),m['macro_recall'])
            self.assertAlmostEqual(log_loss(y,probs,labels=labels),m['log_loss'])
            np.testing.assert_array_equal(confusion_matrix(y,pred,labels=labels),m['confusion'])
            for f in run['folds']:
                self.assertEqual(f['fit_rows']+f['held_rows'],len(data))
                held={r['pass_index'] for r in subset if int(r['cv_fold'])==f['fold']}
                fitting={r['pass_index'] for r in subset if int(r['cv_fold'])!=f['fold']}
                self.assertFalse(held & fitting)


if __name__=='__main__':unittest.main()
