import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
from SeverityOfToxicCommentsEndToEnd.entity import ModelTrainerConfig
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.embeddings_index_fasttext = {}
        self.word_index = None
        self.embedding_matrix_fasttext = None
        self.train_padded = []
        self.test_padded = []
        self.train_labels = None
        self.test_labels = None
        self.df = None
        self.df_test = None
    
    def DefineTokenizer(self, path):
        self.df = pd.read_csv(path)
        self.df = self.df[self.df['comment_text'].notnull()]
        processed_train_text = self.df['comment_text'].to_list()
        self.df_test = pd.read_csv(self.config.processed_test_data_dir)
        self.df_test = self.df_test[self.df_test['comment_text'].notnull()]
        processed_test_text = self.df_test['comment_text'].to_list()

        if not os.path.exists(path):
            tokenizer = Tokenizer(num_words=self.config.max_features)
            tokenizer.fit_on_texts(list(processed_train_text))
            with open(self.config.tokenizer_dir, 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            tokenizer = Tokenizer(num_words=self.config.max_features)
            tokenizer.fit_on_texts(list(processed_train_text))
            list_tokenized_train = tokenizer.texts_to_sequences(processed_train_text)
            self.word_index = tokenizer.word_index
            self.train_padded = pad_sequences(list_tokenized_train, maxlen=self.config.maxpadlen, padding='post')
            list_tokenized_test = tokenizer.texts_to_sequences(processed_test_text)
            self.test_padded = pad_sequences(list_tokenized_test, maxlen=self.config.maxpadlen, padding='post')

    def InitializeFastText(self):
        f = open(self.config.fasttext_model_path, encoding='utf8')
        for line in f:
            line.encode('utf-8').strip()
            values = line.split()
            word = values[0]
            self.embeddings_index_fasttext[word] = np.asarray(values[1:], dtype='float32')
        f.close()
        self.embedding_matrix_fasttext = np.random.random((len(self.word_index) + 1, self.config.embedding_dim))
        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index_fasttext.get(word)
            if embedding_vector is not None:
                self.embedding_matrix_fasttext[i] = embedding_vector
            
    def LSTMTrainer(self):
        self.train_labels = np.array(self.df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])
        self.test_labels = np.array(self.df_test[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(self.word_index) + 1,
                                self.config.embedding_dim,
                                weights = [self.embedding_matrix_fasttext],
                                input_length = self.config.maxpadlen,
                                trainable=False,
                                name = 'embeddings'),
        tf.keras.layers.Input(shape=(self.config.maxpadlen, ),dtype='int32'),
        tf.keras.layers.LSTM(40,return_sequences=True, name='lstm_layer'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.Dense(30, activation='relu', kernel_initializer='he_uniform'),
            tf.keras.layers.Dropout(.1),
            tf.keras.layers.Dense(6, activation='sigmoid', kernel_initializer='glorot_uniform')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        history = model.fit(self.train_padded, self.train_labels, epochs=self.config.epochs, batch_size=self.config.batch_size, validation_data=(self.test_padded, self.test_labels))
        model.save(self.config.root_dir,'/model.h5')