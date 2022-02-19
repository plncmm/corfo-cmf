import torch
import string 
import nltk
import pandas as pd 
import numpy as np
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import SnowballStemmer

class ClaimDataset:
    def __init__(self, filepath, pre_processing, sample) -> None:
        self.filepath = filepath
        self.sentences, self.labels, self.df = self.get_sentences_and_labels(sample, pre_processing["label"], pre_processing["do_lower_case"], pre_processing["remove_punctuation"], pre_processing["remove_stopwords"], pre_processing["remove_frequent_words"], pre_processing["stemming"], pre_processing["remove_short_examples"], pre_processing["lemmatization"])

    def get_sentences_and_labels(self, sample, label, do_lower_case, remove_punctuation, remove_stopwords, remove_frequent_words, stemming, remove_short_examples, lemmatization):
        if self.filepath == 'cleaned_dataset.xlsx': 
            col_names = ["MERCADO_INGRESO", "TIPO_ENTIDAD", "NOMBRE_ENTIDAD", "TIPO_PRODUCTO", "TIPO_MATERIA", "text"]
        else:
            col_names = ["text", label]
        df = pd.read_excel(self.filepath, usecols=col_names)
        df.rename(columns={label: 'label'}, inplace = True)
        
        avg_len = np.mean([len(v.split()) for v in df['text']])
        print(f'Avg len: {avg_len}')
        

        if do_lower_case:
            df["text"] = df["text"].str.lower()
        
        if remove_punctuation:
            PUNCT_TO_REMOVE = string.punctuation
            def remove_punctuation(text):
                """custom function to remove the punctuation"""
                return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

            df["text"] = df["text"].apply(lambda text: remove_punctuation(text))
        
        if remove_stopwords:
            nltk.download('stopwords')
            STOPWORDS = set(stopwords.words('spanish'))
            def remove_stopwords(text):
                """custom function to remove the stopwords"""
                return " ".join([word for word in str(text).split() if word not in STOPWORDS])

            df["text"] = df["text"].apply(lambda text: remove_stopwords(text))
            
        if remove_frequent_words:
            
            cnt = Counter()
            for text in df["text"].values:
                for word in text.split():
                    cnt[word] += 1
       
            FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
            def remove_freqwords(text):
                """custom function to remove the frequent words"""
                return " ".join([word for word in str(text).split() if word not in FREQWORDS])

            df["text"] = df["text"].apply(lambda text: remove_freqwords(text))
        
        if stemming:

            stemmer = SnowballStemmer('spanish')

            def stem_words(text):
                return " ".join([stemmer.stem(word) for word in text.split()])

            df["text"] = df["text"].apply(lambda text: stem_words(text))
        
        if remove_short_examples:
            df = df[df['text'].apply(lambda x: len(x.split()) > 10)]
          
            
        # ToDO: Lemmatizer, Balanceo de clases.

        df = df.sample(frac = sample)
        df = df[df['label'].notna()]
        sentences = df.text.values
        labels = df.label.values
        return sentences, labels, df


