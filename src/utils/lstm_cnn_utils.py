import numpy as np
import imblearn
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def prepare_data(sentences, labels, config):
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)

    train_x, test_x, train_y, test_y = train_test_split(sentences, labels, stratify=labels, test_size=config["test_size"])
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1)
    
    

    train_sequences = tokenizer.texts_to_sequences(train_x)
    val_sequences = tokenizer.texts_to_sequences(valid_x)
    test_sequences = tokenizer.texts_to_sequences(test_x)

    word_index = tokenizer.word_index
    print('Hay %s tokens únicos.' % len(word_index))

    max_length = 500
    train_data = pad_sequences(train_sequences, maxlen=max_length,padding='post')
    val_data = pad_sequences(val_sequences, maxlen=max_length,padding='post')
    test_data = pad_sequences(test_sequences, maxlen=max_length,padding='post')

    train_labels = to_categorical(np.asarray(train_y))
    valid_labels = to_categorical(np.asarray(valid_y))
    test_labels = to_categorical(np.asarray(test_y))

    if config["smote"]:
        sm = imblearn.over_sampling.SMOTE() # Hacemos SMOTe hasta obtener un balance de 30% de la clase minoritaria
        rus = imblearn.under_sampling.RandomUnderSampler() # Hacemos submuestreo hasta obtener un balance del 50%
        steps = [('SMOTE', sm), ('RUS', rus)] # Ponemos ambos pasos
        sm_rus = imblearn.pipeline.Pipeline(steps=steps) # Creamos un pipeline para realizar ambas tareas
        train_data, train_labels = sm_rus.fit_resample(train_data, train_labels)

    return train_data, val_data, test_data, train_labels, valid_labels, test_labels, word_index, max_length

