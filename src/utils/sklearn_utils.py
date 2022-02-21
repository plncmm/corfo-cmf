from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import imblearn
from gensim.models import KeyedVectors
import spacy
import numpy as np 
from tqdm import tqdm 



def to_vector(tokens,model, emb_dim):
    """ Receives a sentence string along with a word embedding model and 
    returns the vector representation of the sentence"""

    
    selected_wv = []
    for token in tokens:
        token = token.text
        if token.lower() in model.key_to_index:
            selected_wv.append(model[token.lower()])
        elif token.upper() in model.key_to_index:
            selected_wv.append(model[token.upper()])
        else:
            selected_wv.append(np.zeros(emb_dim, dtype=float))



    # si seleccionamos por lo menos un embedding para el tweet, lo agregamos y luego lo añadimos.
    
    doc_embeddings = np.mean(np.array(selected_wv), axis=0)
    # si no, añadimos un vector de ceros que represente a ese documento.
    
    return doc_embeddings

def prepare_data(df, config):
    X = df['text']
    y = df['label']
    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(y)
    if not config["use_embeddings"]:
        

        ##tf-idf verctor representation
        tfidf_vect = TfidfVectorizer(analyzer=config["analyzer"], max_features=config["max_features"])
        tfidf_vect.fit(X)
        
        X = tfidf_vect.transform(X)
        

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=config["test_size"])

        #oversample = SMOTE()
        if config["smote"]:
            sm = imblearn.over_sampling.SMOTE() # Hacemos SMOTe hasta obtener un balance de 30% de la clase minoritaria
            rus = imblearn.under_sampling.RandomUnderSampler() # Hacemos submuestreo hasta obtener un balance del 50%
            steps = [('SMOTE', sm), ('RUS', rus)] # Ponemos ambos pasos
            sm_rus = imblearn.pipeline.Pipeline(steps=steps) # Creamos un pipeline para realizar ambas tareas
            X_train, y_train = sm_rus.fit_resample(X_train, y_train)
    
    else:

       

        nlp = spacy.load("es_core_news_sm")
        

        w2v = KeyedVectors.load_word2vec_format(config["embeddings_path"], binary=False)  # C text format
        emb_dim = config["emb_dim"]
        new_vectors = []
        with tqdm(X, unit="tokens") as pbar:
            for sentence in pbar:
                new_vectors.append(to_vector(list(nlp(sentence)), w2v, emb_dim))  



        X_train, X_test, y_train, y_test = train_test_split(new_vectors, y, stratify = y, test_size=config["test_size"])

        #oversample = SMOTE()
        if config["smote"]:
            sm = imblearn.over_sampling.SMOTE() # Hacemos SMOTe hasta obtener un balance de 30% de la clase minoritaria
            rus = imblearn.under_sampling.RandomUnderSampler() # Hacemos submuestreo hasta obtener un balance del 50%
            steps = [('SMOTE', sm), ('RUS', rus)] # Ponemos ambos pasos
            sm_rus = imblearn.pipeline.Pipeline(steps=steps) # Creamos un pipeline para realizar ambas tareas
            X_train, y_train = sm_rus.fit_resample(X_train, y_train)


    return X_train, X_test, y_train, y_test