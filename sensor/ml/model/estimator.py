from sensor.exception import SensorException
from sensor.constant.training_pipeline import MODEL_FILE_NAME,SAVED_MODEL_DIR,TRAINING_BUCKET_NAME
import os,sys,subprocess

class TargetValueMapping:
    def __init__(self):
        self.neg: int = 0
        self.pos: int = 1

    def to_dict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))
    

class SensorModel:
    def __init__(self,preprocessor:object,model:object):
        try:
            self.preprocessor=preprocessor
            self.model=model
        except Exception as e:
            raise SensorException(e,sys)

    def predict(self,x):

        try:
            x_transform=self.preprocessor.transform(x)
            pred=self.model.predict(x_transform)
            return pred
        except Exception as e:
            raise SensorException(e,sys)
        


class ModelResolver:

    def __init__(self,model_dir=SAVED_MODEL_DIR):
        try:
            # self.model_dir = model_dir
            self.model_dir = f's3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}'

        except Exception as e:
            raise e

    def get_best_model_path(self)->str:
        try:
            aws_command = f"aws s3 ls {self.model_dir}/"
            output = subprocess.check_output(aws_command, shell=True, text=True)
            print(output)
            timestamps = [int(line.split()[1].rstrip('/')) for line in output.strip().split('\n') if line.startswith('PRE ')]
            print(timestamps)
            latest_timestamp = max(timestamps)
            latest_model_path= f'{self.model_dir}/{latest_timestamp}/{MODEL_FILE_NAME}'
            print(latest_model_path)
            return latest_model_path
        except Exception as e:
            raise e

    def is_model_exists(self)->bool:
        try:
            if not os.path.exists(self.model_dir):
                return False

            timestamps = os.listdir(self.model_dir)
            if len(timestamps)==0:
                return False
            
            latest_model_path = self.get_best_model_path()

            if not os.path.exists(latest_model_path):
                return False

            return True
        except Exception as e:
            raise e

