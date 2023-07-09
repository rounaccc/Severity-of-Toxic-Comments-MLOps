from SeverityOfToxicCommentsEndToEnd.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
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