from SeverityOfToxicCommentsEndToEnd.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from SeverityOfToxicCommentsEndToEnd.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from SeverityOfToxicCommentsEndToEnd.pipeline.stage_03_model_trainer import ModelTrainerTrainingPipeline

from SeverityOfToxicCommentsEndToEnd.logging import logger

STAGE_NAME = "Stage 01 - Data Ingestion"
try:
    logger.info(f">>>>> Running {STAGE_NAME} <<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>> Finished {STAGE_NAME} <<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Stage 02 - Data Transformation"
try:
    logger.info(f">>>>> Running {STAGE_NAME} <<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f">>>>> Finished {STAGE_NAME} <<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Stage 03 - Model Training"
try:
    logger.info(f">>>>> Running {STAGE_NAME} <<<<<")
    model_training = ModelTrainerTrainingPipeline()
    model_training.main()
    logger.info(f">>>>> Finished {STAGE_NAME} <<<<<")
except Exception as e:
    logger.exception(e)
    raise e