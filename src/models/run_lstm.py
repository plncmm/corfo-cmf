import sys
sys.path.append('../src')
import yaml
import os
import pickle
import pandas as pd 
import numpy as np
from tqdm import tqdm
from datasets.datasets import ClaimDataset
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, Flatten, Bidirectional
from sklearn.metrics import classification_report
from utils.general_utils import enable_reproducibility
from utils.lstm_cnn_utils import prepare_data
from gensim.models import KeyedVectors
import keras
import tensorflow as tf
from sklearn import preprocessing
from keras.callbacks import CSVLogger
from sklearn.utils import class_weight

csv_logger = CSVLogger('../logs/lstm_log.csv', append=True, separator=';')


from keras import backend as K

if __name__=='__main__':
    enable_reproducibility(seed_val=1000)
    
    
    with open('../params.yaml') as file:
        config = yaml.safe_load(file)
        
        
    filepath = config["filepath"]
    model = config["model"]
    device = config["device"]
    pre_processing = config["pre_processing"]
    lstm_config = config["lstm_config"]

    claim_dataset = ClaimDataset(filepath, pre_processing, config["sample_frac"])
    
    df = claim_dataset.df

    if lstm_config["smote"]:
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

    X_train, X_valid, X_test, y_train, y_valid, y_test, word_index, max_length = prepare_data(sentences, labels, lstm_config)
    
    EMBEDDING_DIM = lstm_config["emb_dim"]

    if lstm_config["pre_trained_embs"]: 
        w2v = KeyedVectors.load_word2vec_format(lstm_config["embeddings_path"], binary=False)  # C text format
        
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
                        weights=[embedding_matrix] if lstm_config["pre_trained_embs"] else None,
                        input_length=max_length,
                        trainable=lstm_config["trainable_embs"])
    )
    model.add(Bidirectional(LSTM(lstm_config["hidden_units"], return_sequences=True)))
    model.add(Flatten())
    model.add(Dropout(lstm_config["dropout"]))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(len(target_names),activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    history = model.fit(X_train, 
                        y_train, 
                        class_weight = weights,
                        batch_size = lstm_config["batch_size"],
                        epochs = lstm_config["epochs"],
                        validation_data = (X_valid, y_valid),
                        callbacks = [csv_logger]
    )


    predicted = model.predict(X_test)

    print(classification_report(y_test, predicted.round()))

    if not os.path.exists(lstm_config["output_dir"]):
        os.makedirs(lstm_config["output_dir"])

    # Save the trained model as a pickle string.
    with open(f'{lstm_config["output_dir"]}/model_pkl', 'wb') as files:
        pickle.dump(model, files)


