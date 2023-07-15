from SeverityOfToxicCommentsEndToEnd.config.configuration import ConfigurationManager
from SeverityOfToxicCommentsEndToEnd.components.model_trainer import ModelTrainer
from SeverityOfToxicCommentsEndToEnd.logging import logger

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config = model_trainer_config)
        model_trainer.DefineTokenizer(model_trainer_config.processed_train_data_dir)
        model_trainer.InitializeFastText()
        model_trainer.LSTMTrainer()