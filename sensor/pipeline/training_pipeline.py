from sensor.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
import os,sys
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact,ModelPusherArtifact
from sensor.components.data_ingestion import DataIngestion
from sensor.entity.config_entity import DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig,ModelPusherConfig
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.model_pusher import ModelPusher
from sensor.constant.training_pipeline import SAVED_MODEL_DIR,TRAINING_BUCKET_NAME
from sensor.cloud_storage.s3_syncer import S3Sync


class TrainPipeline:
    is_pipeline_running=False
    def __init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()
        self.s3_sync=S3Sync()

        

    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data ingestion")
            self.data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = self.data_ingestion.initialize_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except  Exception as e:
            raise  SensorException(e,sys)

    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config = data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except  Exception as e:
            raise  SensorException(e,sys)

    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config
            )
            data_transformation_artifact =  data_transformation.initiatialize_data_transformation()
            return data_transformation_artifact
        except  Exception as e:
            raise  SensorException(e,sys)    
        

    def start_model_training(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            model_trainer_config=ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer=ModelTrainer(data_transformation_artifact=data_transformation_artifact,model_trainer_config=model_trainer_config)
            model_trainer_artifact=model_trainer.initialize_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e,sys)
        

    def start_model_evaluation(self,data_validation_artifact:DataValidationArtifact,
                               model_trainer_artifact:ModelTrainerArtifact):
        try:
            model_eval_config=ModelEvaluationConfig(training_pipeling_config=self.training_pipeline_config)
            model_eval=ModelEvaluation(data_validation_artifact=data_validation_artifact
                                       ,model_trainer_artficat=model_trainer_artifact
                                       ,model_eval_config=model_eval_config)
            model_eval_artifact=model_eval.initialize_model_evaluation()
            return model_eval_artifact
        except Exception as e:
            raise SensorException(e,sys)
        
    
    def start_model_pusher(self,model_eval_artifact:ModelEvaluationArtifact):
        try:
            model_pusher_config=ModelPusherConfig(training_pipeline_config=self.training_pipeline_config)
            model_pusher=ModelPusher(model_eval_artifact=model_eval_artifact,model_pusher_config=model_pusher_config)
            model_pusher_artifact=model_pusher.initialize_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise SensorException(e,sys)


    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.artifact_dir,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise SensorException(e,sys)
            
    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}"
            self.s3_sync.sync_folder_to_s3(folder = SAVED_MODEL_DIR,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise SensorException(e,sys)



    def run_pipeline(self):

        try:
            TrainPipeline.is_pipeline_running=True
            self.data_ingestion_artifact:DataIngestionArtifact= self.start_data_ingestion()
            self.data_validation_artifact:DataValidationArtifact=self.start_data_validation(data_ingestion_artifact=self.data_ingestion_artifact)
            self.data_transformation_artifact:DataTransformationArtifact=self.start_data_transformation(data_validation_artifact=self.data_validation_artifact)
            self.model_trainer_artifact:ModelTrainerArtifact=self.start_model_training(self.data_transformation_artifact)
            self.model_eval_artifact:ModelEvaluationArtifact=self.start_model_evaluation(self.data_validation_artifact,self.model_trainer_artifact)
            if(not self.model_eval_artifact.is_model_accepted):
                raise Exception("This is not the best model. Try Retraining")
            self.model_pusher_artifact:ModelPusherArtifact=self.start_model_pusher(self.model_eval_artifact)
            TrainPipeline.is_pipeline_running=False
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()
        except Exception as e:
            self.sync_artifact_dir_to_s3()
            TrainPipeline.is_pipeline_running=False
            raise SensorException(e,sys)



            


        