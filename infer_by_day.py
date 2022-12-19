import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import cleanlab
from cleanlab.classification import CleanLearning
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Read csv file with all feature 
df = pd.read_csv('/home2/tamht/train-test/all.csv')#,index_col=0)

# Convert categorical variable to integer label 
df['tell_call'].replace(df['tell_call'].value_counts().index,[i for i in range(len(df['tell_call'].value_counts().index))],inplace=True)

# Replace silence audio feature with dummy data
for x in ['rolloff_min','flatness','spec_bw','cent','zcr','rmse','loudness','snr']:
    df[x].replace("SILENCE",-1000,inplace=True)

# Filter by month 
df.drop(df[df['mapping_code'].isnull()].index,inplace=True)
df3 = df[df['record_audio'].str.contains("/2022/03/")]
df4 = df[df['record_audio'].str.contains("/2022/04/")]
df7 = df[df['record_audio'].str.contains("/2022/07/")]
df8 = df[df['record_audio'].str.contains("/2022/08/")]
df9 = df[df['record_audio'].str.contains("/2022/09/")]
df10 = df[df['record_audio'].str.contains("/2022/10/")]

# Training data
df = df3
X=np.array(df.drop(['record_audio','transcription','note_content'],axis=1))
y=np.array(df['note_content'])

# Parameter
check_month = '04'
name = 'emsemble_DT'
path = '/home2/tamht/'+check_month+"_"+name

# Create directory
isExist = os.path.exists(path)
if not isExist:
  os.makedirs(path)

# Add dummy samples to trainning data with missing note_content label  
a_tr = []
for z in range(25):
    if z not in df['note_content'].value_counts().index:
        a_tr.append(z)
for z in a_tr:
    X = np.append(X,[[-1000 for i in range(X.shape[1])]],axis=0)
    y = np.append(y,z)

# 3 models
classifier_1 = XGBClassifier(tree_method='gpu_hist', n_estimators=1000)
classifier_2 = CatBoostClassifier(eval_metric="TotalF1", task_type="GPU", auto_class_weights="Balanced", learning_rate=0.1)
classifier_3 = RandomForestClassifier(n_estimators=100, n_jobs=4, class_weight="balanced")

# Train the model with Cleanlab 
cl1 = CleanLearning(clf = classifier_1)  # any sklearn-compatible classifier
cl1.fit(X, y)
cl2 = CleanLearning(clf = classifier_2)  # any sklearn-compatible classifier
cl2.fit(X, y)
cl3 = CleanLearning(clf = classifier_3)  # any sklearn-compatible classifier
cl3.fit(X, y)

# Infer and report by day in specified month
accuracy = open("/home2/tamht/"+check_month+"_"+name+"/accuracy.txt", "w")
weighted_f1 = open("/home2/tamht/"+check_month+"_"+name+"/weighted_f1.txt", "w")

for x in ['01','02','03','04','05','06','07','08','09']+[str(i) for i in range(10,32)]:
    # Infering data (needed to specified)
    df_test = df4[df4['record_audio'].str.contains("/2022/"+check_month+"/"+x+"/")]
    X_test=np.array(df_test.drop(['record_audio','transcription','note_content'],axis=1))
    y_test=np.array(df_test['note_content'])
    if len(X_test) == 0:
        continue

    # Add dummy samples to trainning data with missing note_content label  
    a_t = []
    for z in range(25):
        if z not in df_test['note_content'].value_counts().index:
            a_t.append(z)
    for z in a_t:
        X_test = np.append(X_test,[[-1000 for i in range(X_test.shape[1])]],axis=0)
        y_test = np.append(y_test,z)

    # Predict 
    cv_pred_probs_1 = cl1.predict_proba(X_test)
    cv_pred_probs_2 = cl2.predict_proba(X_test)
    cv_pred_probs_3 = cl3.predict_proba(X_test)

    # Ensemble
    pred_probs = (cv_pred_probs_1 + cv_pred_probs_2 + cv_pred_probs_3)/3  # uniform aggregation of predictions

    # Write weighted_f1 and accuracy to file
    predicted_labels = pred_probs.argmax(axis=1)
    cv_accuracy = accuracy_score(y_test, predicted_labels)
    accuracy.write("2022/"+check_month+"/"+str(x)+": ")
    accuracy.write(str(cv_accuracy)+'\n')
    weighted_f1.write("2022/"+check_month+"/"+str(x)+': ')
    weighted_f1.write(str(f1_score(y_test, predicted_labels, average='weighted'))+'\n')

    # Write report to file
    report = open("/home2/tamht/"+check_month+"_"+name+"/report-"+str(x)+".txt", "w")
    report.write(classification_report(y_test, predicted_labels))
    report.close()

    # Write problematic samples to separated files
    label_issues_indices = cleanlab.filter.find_label_issues(
        labels=y_test,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",  # ranks the label issues
    )
    df.reset_index(drop=True,inplace=True)                                                                                              
    error = df.iloc[label_issues_indices]
    error.to_csv('/home2/tamht/'+check_month+"_"+name+"/2022-"+check_month+"-"+str(x)+'.csv')                              

accuracy.close()
weighted_f1.close()