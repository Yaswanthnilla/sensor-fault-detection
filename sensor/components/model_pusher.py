from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity.artifact_entity import ModelEvaluationArtifact,ModelPusherArtifact
from sensor.entity.config_entity import ModelPusherConfig
from sensor.exception import SensorException
import os,sys
import shutil


class ModelPusher:
    
    def __init__(self,model_pusher_config:ModelPusherConfig,model_eval_artifact:ModelEvaluationArtifact):
        try:
            self.model_pusher_config=model_pusher_config
            self.model_eval_artifact=model_eval_artifact
        except Exception as e:
            raise SensorException(e,sys)
   
    

    def initialize_model_pusher(self)->ModelPusherArtifact:

        try:

            trained_model_file_path=self.model_eval_artifact.trained_model_path
            model_pusher_file_path=self.model_pusher_config.model_pusher_file_path
            saved_model_file_path=self.model_pusher_config.saved_models_file_path

            #Make model pusher directory
            dir_path=os.path.dirname(model_pusher_file_path)
            os.makedirs(dir_path,exist_ok=True)
            shutil.copy(trained_model_file_path,model_pusher_file_path)

            #Make saved models directory
            dir_path=os.path.dirname(saved_model_file_path)
            os.makedirs(dir_path,exist_ok=True)
            shutil.copy(trained_model_file_path,saved_model_file_path)

            model_pusher_artifact=ModelPusherArtifact(model_pusher_file_path=model_pusher_file_path,saved_models_file_path=saved_model_file_path)
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        
        except Exception as e:
            raise SensorException(e,sys)







