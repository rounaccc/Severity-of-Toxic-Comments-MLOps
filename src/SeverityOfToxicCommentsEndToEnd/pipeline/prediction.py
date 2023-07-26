from SeverityOfToxicCommentsEndToEnd.config.configuration import ConfigurationManager
import pickle
import tensorflow as tf
import keras
from SeverityOfToxicCommentsEndToEnd.components.data_transformation import DataTransformation

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_trainer_config()
    
    def predict(self, text):
        # preprocessing the text
        data_transformation = DataTransformation(self.config)
        text = data_transformation.clean_text(text)
        text = data_transformation.lemma(text)
        data_transformation.dual_alpha()
        data_transformation.alter_dual_alpha()
        data_transformation.alter_stopwords()
        text = data_transformation.remove_stopwords(text)
        print(text)

        tokenizer = self.config.tokenizer_dir
        with open(tokenizer, 'rb') as handle:
            tokenizer = pickle.load(handle)

        model = tf.keras.models.load_model(self.config.root_dir)

        text = tokenizer.texts_to_sequences([text])
        text = keras.preprocessing.sequence.pad_sequences(text, maxlen=200, padding='post')
        prediction = model.predict(text)

        return prediction
