from SeverityOfToxicCommentsEndToEnd.config.configuration import ConfigurationManager
import pickle
import tensorflow as tf
import keras

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_trainer_config()
    
    def predict(self, text):
        tokenizer = self.config.tokenizer_dir
        with open(tokenizer, 'rb') as handle:
            tokenizer = pickle.load(handle)

        model = tf.keras.models.load_model(self.config.root_dir)

        text = tokenizer.texts_to_sequences([text])
        text = keras.preprocessing.sequence.pad_sequences(text, maxlen=200, padding='post')
        print(text)
        prediction = model.predict(text)

        return prediction[0]
