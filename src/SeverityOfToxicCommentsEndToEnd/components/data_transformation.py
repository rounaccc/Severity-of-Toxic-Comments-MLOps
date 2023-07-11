import os
from SeverityOfToxicCommentsEndToEnd.logging import logger
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import itertools
from string import ascii_lowercase
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from DataTransformationParam import *
from SeverityOfToxicCommentsEndToEnd.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.stopword_list = []
        self.dual_alpha_list = []
        self.train_text = []
        self.lemma_train_text = []
        self.processed_train_text = []
    
    def clean_text(self, text, remove_repeat_text=True, remove_patterns_text=True, is_lower=True):

        if is_lower:
            text=text.lower()
            
        if remove_patterns_text:
            for target, patterns in RE_PATTERNS.items():
                for pat in patterns:
                    text=str(text).replace(pat, target)

        if remove_repeat_text:
            text = re.sub(r'(.)\1{2,}', r'\1', text) 

        text = str(text).replace("\n", " ")
        text = re.sub(r'[^\w\s]',' ',text)
        text = re.sub('[0-9]',"",text)
        text = re.sub(" +", " ", text)
        text = re.sub("([^\x00-\x7F])+"," ",text)
        return text 
    
    def lemma(self, text, lemmatization=True):
        lemmatizer = WordNetLemmatizer()
        output=''
        if lemmatization:
            text=text.split(' ')
            for word in text:
                word1 = lemmatizer.lemmatize(word, pos = "n") #noun 
                word2 = lemmatizer.lemmatize(word1, pos = "v") #verb
                word3 = lemmatizer.lemmatize(word2, pos = "a") #adjective
                word4 = lemmatizer.lemmatize(word3, pos = "r") #adverb
                output=output + " " + word4
        else:
            output=text
        
        return str(output.strip())
    
    def iter_all_strings(self):
        for size in itertools.count(1):
            for s in itertools.product(ascii_lowercase, repeat=size):
                yield "".join(s)
    
    def dual_alpha(self):
        for s in self.iter_all_strings():
            self.dual_alpha_list.append(s)
            if s == 'zz':
                break
    
    def alter_dual_alpha(self):
        self.dual_alpha_list.remove('i')
        self.dual_alpha_list.remove('a')
        self.dual_alpha_list.remove('am')
        self.dual_alpha_list.remove('an')
        self.dual_alpha_list.remove('as')
        self.dual_alpha_list.remove('at')
        self.dual_alpha_list.remove('be')
        self.dual_alpha_list.remove('by')
        self.dual_alpha_list.remove('do')
        self.dual_alpha_list.remove('go')
        self.dual_alpha_list.remove('he')
        self.dual_alpha_list.remove('hi')
        self.dual_alpha_list.remove('if')
        self.dual_alpha_list.remove('is')
        self.dual_alpha_list.remove('in')
        self.dual_alpha_list.remove('me')
        self.dual_alpha_list.remove('my')
        self.dual_alpha_list.remove('no')
        self.dual_alpha_list.remove('of')
        self.dual_alpha_list.remove('on')
        self.dual_alpha_list.remove('or')
        self.dual_alpha_list.remove('ok')
        self.dual_alpha_list.remove('so')
        self.dual_alpha_list.remove('to')
        self.dual_alpha_list.remove('up')
        self.dual_alpha_list.remove('us')
        self.dual_alpha_list.remove('we')

        for letter in self.dual_alpha_list:
            self.stopword_list.append(letter)
    
    def alter_stopwords(self):
        for word in potential_stopwords:
            self.stopword_list.append(word)
        print(len(self.stopword_list))

    def remove_stopwords(self, text, remove_stop=True):
        output = ""
        if remove_stop:
            text=text.split(" ")
            for word in text:
                if word not in self.stopword_list:
                    output=output + " " + word
        else :
            output=text

        return str(output.strip())
    
    def perform_data_transformation(self):
        df = pd.read_csv(self.config.data_path)
        for line in df['comment_text']: 
            self.train_text.append(self.clean_text(line))

        for line in self.train_text:
            self.lemma_train_text.append(self.lemma(line))
        
        self.dual_alpha()
        self.alter_dual_alpha()
        self.alter_stopwords()
        for line in self.lemma_train_text: 
            self.processed_train_text.append(self.remove_stopwords(line))
    
    def tokenizer(self):
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(self.processed_train_text))
        list_tokenized_train = tokenizer.texts_to_sequences(self.processed_train_text)
        word_index=tokenizer.word_index
        training_padded=pad_sequences(list_tokenized_train, maxlen=maxpadlen, padding = 'post')
        # save tokenizer
        with open(self.config.tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)