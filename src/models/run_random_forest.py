import sys
sys.path.append('../src')
from multiprocessing.spawn import prepare
import yaml
import pandas as pd 
import numpy as np
from datasets.datasets import ClaimDataset
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
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
    random_forest_config = config["random_forest_config"]
    claim_dataset = ClaimDataset(filepath, pre_processing, config["sample_frac"])
    df = claim_dataset.df
    
    if random_forest_config["smote"]:
        counts = df['label'].value_counts()
        df = df[~df['label'].isin(counts[counts < 10].index)]
    
    X_train, X_test, y_train, y_test = prepare_data(df, random_forest_config)
    if random_forest_config["balanced"]:
        tuned_parameters = {
            #'bootstrap': [True, False],
            'max_depth': [10, 50, 100]
            #'max_features': ['auto', 'sqrt'],
            #'min_samples_leaf': [1, 2, 4],
            #'min_samples_split': [2, 5, 10],
            #'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
            #"class_weight":["balanced"]
        }
        estimator = BalancedRandomForestClassifier()
        SRF = GridSearchCV(estimator, tuned_parameters, cv=10, scoring='f1_micro', verbose=4)
    else: 
        estimator = RandomForestClassifier()
        tuned_parameters = {
            'bootstrap': [True, False]
            #'max_depth': [10, 30, 50, 80, 100, None],
            #'max_features': ['auto', 'log2', None],
            #'min_samples_leaf': [1, 2, 4],
            #'min_samples_split': [2, 5, 10],
            #'n_estimators': [100, 200, 500, 1000, 1200, 1400, 1600, 1800],
            #"class_weight":["balanced", None]
        }
        SRF = GridSearchCV(estimator, tuned_parameters, cv=10, scoring='f1_micro', verbose=4)

    #Train SRF
    SRF.fit(X_train, y_train)
    #SRF prediction result
    y_pred = SRF.predict(X_test)
    #Create confusion matrix
    
    print(classification_report(y_test, y_pred))

    if not os.path.exists(random_forest_config["output_dir"]):
        os.makedirs(random_forest_config["output_dir"])

    # Save the trained model as a pickle string.
    with open(f'{random_forest_config["output_dir"]}/model_pkl', 'wb') as files:
        pickle.dump(model, files)