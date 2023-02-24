from sensor.configuration.mongo_db_connection import MongoDBClient
import os,sys
from sensor.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from sensor.logger import logging
from sensor.exception import SensorException

from sensor.pipeline.training_pipeline import TrainPipeline

from sensor.constant.training_pipeline import SCHEMA_FILE_PATH

from sensor.entity.config_entity import DataIngestionConfig


# if __name__ == '__main__':
#     mongodb_client=MongoDBClient()
#     print(mongodb_client.database.list_collection_names())


# if __name__== '__main__':
#     training_pipeline_config=TrainingPipelineConfig()
#     data_ingestion_config=DataIngestionConfig(training_pipeline_config=training_pipeline_config)
#     print(data_ingestion_config.__dict__)



if __name__ == '__main__':

	try:

		training_pipeline=TrainPipeline()
		training_pipeline.run_pipeline()

	except Exception as e:
		raise SensorException(e,sys)


