import pandas as pd
import numpy as np
from feature_engine.encoding import OneHotEncoder
import gc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import cleanlab
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import sklearn
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
import inspect
from sklearn.ensemble import VotingClassifier
from cleanlab.rank import get_label_quality_ensemble_scores
from cleanlab.count import num_label_issues
from sklearn.ensemble import ExtraTreesClassifier
import tqdm

train_path = "/home2/ivr-mappingcode/ivr/data/preprocessed/train_with_silence.csv"
test_path = "/home2/ivr-mappingcode/ivr/data/preprocessed/test_with_silence.csv"
save_folder = '/home2/ivr-mappingcode/huynv/source/clean/save'
description1 = 'no_transcript'
description2 = 'all_data'

metadata = pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)])
metadata = metadata.reset_index(drop=True)

metadata["state"] = metadata["state"].astype(object)
metadata["tell_call"] = metadata["tell_call"].astype(object)

encoder = OneHotEncoder(variables=['state', 'tell_call'])
metadata = encoder.fit_transform(metadata)

non_speech_features = ["time_call", "time_remain", "total_time_remain", "state_50.0", "state_40.0", "tell_call_viettel", "tell_call_vinaphone", "tell_call_mobiphone", "tell_call_others"]

for fea in non_speech_features:
    metadata[fea] = metadata[fea].astype(np.float64)
    metadata[fea] = metadata[fea].values
metadata['time_call'] = metadata['time_call'].apply(lambda x : x / 23)

audio_features = ["rolloff_min", "flatness", "spec_bw", "cent", "zcr", "rmse", "loudness", "snr"]
for fea in audio_features:
    metadata[fea] = metadata[fea].apply(lambda x: -10000 if x == "SILENCE" else x)
    metadata[fea] = metadata[fea].astype(np.float64)
    metadata[fea] = metadata[fea].values

all_im_feature = non_speech_features + audio_features

sample_weights = compute_sample_weight(class_weight='balanced', y=metadata["note_content"].values)

standarder = StandardScaler()
normalizer = MinMaxScaler()

tmp_feature = [fea for fea in all_im_feature if fea != 'time_call']
metadata[tmp_feature] = standarder.fit_transform(metadata[tmp_feature])
metadata[tmp_feature] = normalizer.fit_transform(metadata[tmp_feature])
del tmp_feature

num_crossval_folds = 10

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

filter_by_list = [
    "prune_by_noise_rate",
    "prune_by_class",
    "both",
    "confident_learning",
    "predicted_neq_given",
]
indices_ranked_method_list = [
    "self_confidence",
    "normalized_margin",
    "confidence_weighted_entropy"
]

def add_pred_probs(func, **kwargs):
    print('Running cl_all_data_no_trans.py')
    model_name = type(kwargs['estimator']).__name__.replace('Classifier', '')
    filter_by = kwargs['filter_by']
    return_indices_ranked_by = kwargs['return_indices_ranked_by']
    description3 = f'{model_name}-{filter_by}-{return_indices_ranked_by}.csv'
    path_to_error_csv = os.path.join(save_folder, model_name, description1, description2, description3)

    if os.path.exists(path_to_error_csv):
        return
    
    if 'fit_params' in kwargs:
        print(type(kwargs['estimator']).__name__.replace('Classifier', ''), 'in add_pred_probs function')
        pred_probs=func(
            estimator=kwargs['estimator'],
            X = kwargs['X'],
            y = kwargs['y'],
            cv = kwargs['cv'],
            method = kwargs['method'],
            fit_params=kwargs['fit_params']
        )
    else:
        print('in add_pred_probs function')
        # print(type(kwargs['estimator']).__name__.replace('Classifier', ''), 'in add_pred_probs function')
        pred_probs=func(
            estimator=kwargs['estimator'],
            X = kwargs['X'],
            y = kwargs['y'],
            cv = kwargs['cv'],
            method = kwargs['method'],
        )

    make_dir(os.path.join(save_folder, model_name))
    make_dir(os.path.join(save_folder, model_name, description1))
    make_dir(os.path.join(save_folder, model_name, description1, description2))

    label_issues_indices = cleanlab.filter.find_label_issues(
        labels=metadata['note_content'].values,
        pred_probs=pred_probs,
        return_indices_ranked_by=return_indices_ranked_by,  # ranks the label issues
        filter_by=filter_by
    )

    del pred_probs
    gc.collect()

    error_list = list(metadata.iloc[label_issues_indices]['record_audio'])

    error_df = pd.concat([pd.read_csv(train_path) , pd.read_csv(test_path)])
    error_df = error_df.apply(lambda row : row[error_df['record_audio'].isin(error_list)])
    assert len(error_df) == len(error_list)

    error_df.to_csv(path_to_error_csv, index=False)
    del error_df
    del error_list
    gc.collect()

pair_filter_rank = []
for filter_by in filter_by_list:
    for return_indices_ranked_by in indices_ranked_method_list:
        pair_filter_rank.append((filter_by, return_indices_ranked_by))
# pair_filter_rank = pair_filter_rank

for filter_by, return_indices_ranked_by in tqdm.tqdm(pair_filter_rank):
    print(filter_by, return_indices_ranked_by)

    add_pred_probs(func=cross_val_predict, 
            estimator=ExtraTreesClassifier(n_estimators=100, n_jobs=4, class_weight="balanced"),
            X = metadata[all_im_feature].values,
            y = metadata['note_content'].values,
            cv = num_crossval_folds,
            method="predict_proba",
            filter_by = filter_by,
            return_indices_ranked_by = return_indices_ranked_by,
        )

    add_pred_probs(func=cross_val_predict, 
            estimator=VotingClassifier(
                estimators=[
                    ('xgboost', XGBClassifier(tree_method='gpu_hist', n_estimators=1000)), 
                    ('catboost', CatBoostClassifier(eval_metric="TotalF1", task_type="GPU", auto_class_weights="Balanced", learning_rate=0.1)), 
                    ('randomforest', RandomForestClassifier(n_estimators=100, n_jobs=4, class_weight="balanced")),
                    ('extratrees', ExtraTreesClassifier(n_estimators=100, n_jobs=4, class_weight="balanced"))
                ],
                voting='soft', weights=[1, 1, 1, 1]
            ),
            X = metadata[all_im_feature].values,
            y = metadata['note_content'].values,
            cv = num_crossval_folds,
            method="predict_proba",
            filter_by = filter_by,
            return_indices_ranked_by = return_indices_ranked_by,
            fit_params={"sample_weight": sample_weights}
        )

    add_pred_probs(func=cross_val_predict, 
            estimator=XGBClassifier(tree_method='gpu_hist', n_estimators=1000),
            X = metadata[all_im_feature].values,
            y = metadata['note_content'].values,
            cv = num_crossval_folds,
            method="predict_proba",
            filter_by = filter_by,
            return_indices_ranked_by = return_indices_ranked_by,
            fit_params={"sample_weight": sample_weights}
        )

    add_pred_probs(func=cross_val_predict, 
            estimator=CatBoostClassifier(eval_metric="TotalF1", task_type="GPU", auto_class_weights="Balanced", learning_rate=0.1),
            X = metadata[all_im_feature].values,
            y = metadata['note_content'].values,
            cv = num_crossval_folds,
            method="predict_proba",
            filter_by = filter_by,
            return_indices_ranked_by = return_indices_ranked_by,
        )

    add_pred_probs(func=cross_val_predict, 
            estimator=RandomForestClassifier(n_estimators=100, n_jobs=4, class_weight="balanced"),
            X = metadata[all_im_feature].values,
            y = metadata['note_content'].values,
            cv = num_crossval_folds,
            method="predict_proba",
            filter_by = filter_by,
            return_indices_ranked_by = return_indices_ranked_by,
        )