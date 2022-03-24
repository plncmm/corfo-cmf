import sys
sys.path.append('../src')
from xgboost import XGBClassifier
from multiprocessing.spawn import prepare
import yaml
import pandas as pd 
import numpy as np
from datasets.datasets import ClaimDataset
import os
import pickle
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
    xgboost_config = config["xgboost_config"]


    claim_dataset = ClaimDataset(filepath, pre_processing, config["sample_frac"])
    df = claim_dataset.df

    if xgboost_config["smote"]:
        counts = df['label'].value_counts()
        df = df[~df['label'].isin(counts[counts < 10].index)]

   
    

    X_train, X_test, y_train, y_test = prepare_data(df, xgboost_config)
    
    tuned_parameters = {
            'booster' : ['gblinear', 'gbtree'],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, None],
            'n_estimators': [50, 100, 200, 300, 500, 750, 1000],
            'max_depth': [2, 4, 8, 10, 30, 50, 80, None],
            'subsample': [0.3, 0.5, 0.75, None],
            'scale_pos_weight ':[3,4,5]
        }

    estimator = XGBClassifier()

    clf = clf = GridSearchCV(estimator, tuned_parameters, cv=10, scoring='f1_micro')
                          
                          

    clf.fit(X_train, y_train)
    clf_predict = clf.predict(X_test)
    print(classification_report(y_test, clf_predict))

    if not os.path.exists(xgboost_config["output_dir"]):
        os.makedirs(xgboost_config["output_dir"])

    # Save the trained model as a pickle string.
    with open(f'{xgboost_config["output_dir"]}/model_pkl', 'wb') as files:
        pickle.dump(model, files)
    
