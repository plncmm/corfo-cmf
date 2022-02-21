import sys
sys.path.append('../src')
from multiprocessing.spawn import prepare
import yaml
import pandas as pd 
import numpy as np
from datasets.datasets import ClaimDataset
from sklearn.linear_model import LogisticRegression
import os
import re
import pickle
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
    logistic_config = config["logistic_config"]
    claim_dataset = ClaimDataset(filepath, pre_processing, config["sample_frac"])
    df = claim_dataset.df

    if logistic_config["smote"]:
        counts = df['label'].value_counts()
        df = df[~df['label'].isin(counts[counts < 10].index)]

    X_train, X_test, y_train, y_test = prepare_data(df, logistic_config)
    
    tuned_parameters = {"C": np.logspace(-4, 4, 20),
                        "penalty": ["l1","l2"],
                        "class_weight": ["balanced"]
    }

    estimator = LogisticRegression()
    
    logistic = GridSearchCV(estimator, tuned_parameters, cv=3, scoring='f1_micro')
    print("Grid Search")
    #Train SRF
    logistic.fit(X_train, y_train)
    #SRF prediction result
    y_pred = logistic.predict(X_test)
    #Create confusion matrix
    
    print(classification_report(y_test, y_pred))

    if not os.path.exists(logistic_config["output_dir"]):
        os.makedirs(logistic_config["output_dir"])

    # Save the trained model as a pickle string.
    with open(f'{logistic_config["output_dir"]}/model_pkl', 'wb') as files:
        pickle.dump(model, files)
    





