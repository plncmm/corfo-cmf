
import sys
sys.path.append('../src')
from sklearn.svm import SVC
from multiprocessing.spawn import prepare
import yaml
import pandas as pd 
import numpy as np
from datasets.datasets import ClaimDataset
import os
import re
from sklearn.metrics import classification_report
from utils.sklearn_utils import prepare_data
from sklearn.model_selection import GridSearchCV

if __name__=='__main__':
    with open('../params.yaml') as file:
        config = yaml.safe_load(file)
    
    filepath = config["filepath"]
    model = config["model"]
    device = config["device"]
    pre_processing = config["pre_processing"]
    svm_config = config["svm_config"]


    claim_dataset = ClaimDataset(filepath, pre_processing, config["sample_frac"])
    df = claim_dataset.df

    if svm_config["smote"]:
        counts = df['label'].value_counts()
        df = df[~df['label'].isin(counts[counts < 10].index)]

    
    

    X_train, X_test, y_train, y_test = prepare_data(df, svm_config)
    
    tuned_parameters = {'C':[1,10,100,1000],
            'gamma':[1,0.1,0.001,0.0001], 
            'kernel':['linear','rbf'],
            "class_weight":["balanced"],
            'probability':[True]}

    estimator = SVC()
    final_svc = GridSearchCV(estimator, tuned_parameters, cv=10, scoring='f1_micro')

    final_svc.fit(X_train,y_train)
    final_svc_predict = final_svc.predict(X_test)
    print(classification_report(y_test, final_svc_predict))

    if not os.path.exists(svm_config["output_dir"]):
        os.makedirs(svm_config["output_dir"])

    # Save the trained model as a pickle string.
    with open(f'{svm_config["output_dir"]}/model_pkl', 'wb') as files:
        pickle.dump(model, files)
    