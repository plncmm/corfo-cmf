import sys
sys.path.append('../src')
import yaml
import pandas as pd 
import numpy as np
import os
import pickle
import re
from datasets.datasets import ClaimDataset
from utils.sklearn_utils import prepare_data
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

if __name__=='__main__':
    with open('../params.yaml') as file:
        config = yaml.safe_load(file)
    
    filepath = config["filepath"]
    model = config["model"]
    device = config["device"]
    pre_processing = config["pre_processing"]
    nv_config = config["nv_config"]


    claim_dataset = ClaimDataset(filepath, pre_processing, config["sample_frac"])
    df = claim_dataset.df
    if nv_config["smote"]:
        counts = df['label'].value_counts()
        df = df[~df['label'].isin(counts[counts < 10].index)]

    X_train, X_test, y_train, y_test = prepare_data(df, nv_config)
    
    estimator = MultinomialNB()

    tuned_parameters = {
        'alpha': [1e-1, 1e-2]
    }


    clf = GridSearchCV(estimator, tuned_parameters, cv=3, scoring='f1_micro')
    clf.fit(X_train, y_train)

    print(classification_report(y_test, clf.predict(X_test), digits=4))

    if not os.path.exists(nv_config["output_dir"]):
        os.makedirs(nv_config["output_dir"])

    # Save the trained model as a pickle string.
    with open(f'{nv_config["output_dir"]}/model_pkl', 'wb') as files:
        pickle.dump(model, files)
    