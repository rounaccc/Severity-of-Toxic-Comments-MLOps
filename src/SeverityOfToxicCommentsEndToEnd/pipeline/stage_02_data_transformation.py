from SeverityOfToxicCommentsEndToEnd.config.configuration import ConfigurationManager
from SeverityOfToxicCommentsEndToEnd.components.data_transformation import DataTransformation
from SeverityOfToxicCommentsEndToEnd.logging import logger

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config = data_transformation_config)
        data_transformation.perform_data_transformation()
        data_transformation.tokenizer()