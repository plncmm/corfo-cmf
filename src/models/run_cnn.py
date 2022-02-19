import sys
sys.path.append('../src')
import yaml
import os
import pickle
import pandas as pd 
import numpy as np
import sklearn
import torch
import imblearn
from keras.layers import Dense, Embedding, Dropout, Conv1D, MaxPooling1D, Embedding, Input, Flatten, BatchNormalization
from keras.models import Sequential
from datasets.datasets import ClaimDataset
from sklearn.metrics import classification_report
from gensim.models import KeyedVectors
from utils.general_utils import enable_reproducibility
from utils.lstm_cnn_utils import prepare_data
from tqdm import tqdm
import keras
import tensorflow as tf
from sklearn import preprocessing
from keras.callbacks import CSVLogger
from sklearn.utils import class_weight
csv_logger = CSVLogger('../logs/cnn_log.csv', append=True, separator=';')


if __name__=='__main__':    
    enable_reproducibility(seed_val=1000)
    
    with open('../params.yaml') as file:
        config = yaml.safe_load(file)
        
        
    filepath = config["filepath"]
    model = config["model"]
    device = config["device"]
    pre_processing = config["pre_processing"]
    cnn_config = config["cnn_config"]

    claim_dataset = ClaimDataset(filepath, pre_processing, config["sample_frac"])
    
    df = claim_dataset.df
    
    if cnn_config["smote"]:
        counts = df['label'].value_counts()
        df = df[~df['label'].isin(counts[counts < 10].index)]

    sentences, labels = df.text.values, df.label.values
    
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    target_names = list(le.classes_)
    
    class_weights = list(class_weight.compute_class_weight(class_weight = 'balanced',
                                             classes = np.unique(labels),
                                             y = labels))

    weights={}
    for index, weight in enumerate(class_weights):
        weights[index]=weight
        
    X_train, X_valid, X_test, y_train, y_valid, y_test, word_index, max_length = prepare_data(sentences, labels, cnn_config)
    
    EMBEDDING_DIM = cnn_config["emb_dim"]

    if cnn_config["pre_trained_embs"]: 
        w2v = KeyedVectors.load_word2vec_format(cnn_config["embeddings_path"], binary=False)  # C text format
        
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        with tqdm(word_index.items(), unit="tokens") as pbar:
            for word, i in pbar:
                if word.lower() in w2v.key_to_index:
                    embedding_matrix[i] = w2v[word.lower()]
                elif word.upper() in w2v.key_to_index:
                    embedding_matrix[i] = w2v[word.upper()]
                else:
                    embedding_matrix[i] = np.zeros(EMBEDDING_DIM, dtype=float)

    
    


    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix] if cnn_config["pre_trained_embs"] else None,
                                input_length=max_length,
                                trainable=cnn_config["trainable_embs"]))
    model.add(Dropout(cnn_config["dropout"]))
    model.add(Conv1D(cnn_config["filters"], cnn_config["kernel_size"], activation="relu"))
    model.add(MaxPooling1D(cnn_config["kernel_size"]))
    model.add(Dropout(cnn_config["dropout"]))
    model.add(BatchNormalization())
    model.add(Conv1D(cnn_config["filters"], cnn_config["kernel_size"], activation="relu"))
    model.add(MaxPooling1D(cnn_config["kernel_size"]))
    model.add(Dropout(cnn_config["dropout"]))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(cnn_config["dense_units"], activation="relu"))
    model.add(Dense(len(target_names), activation="softmax"))
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['acc',
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall()])
    history = model.fit(X_train, y_train, class_weight = weights,
                        batch_size=cnn_config["batch_size"],
                        epochs=cnn_config["epochs"],
                        validation_data=(X_valid, y_valid),
                        callbacks = [csv_logger]
                        )
    #predictions on test data
    predicted = model.predict(X_test)

    print(classification_report(y_test, predicted.round()))

    if not os.path.exists(cnn_config["output_dir"]):
        os.makedirs(cnn_config["output_dir"])

    with open(f'{cnn_config["output_dir"]}/model_pkl', 'wb') as files:
        pickle.dump(model, files)