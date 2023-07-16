from sensor.constant.training_pipeline import TARGET_COLUMN,MODEL_EVALUATION_THRESHOLD_SCORE
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity.artifact_entity import DataValidationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from sensor.entity.config_entity import ModelEvaluationConfig
from sensor.exception import SensorException
from sensor.utils.main_utils import load_object,write_yaml_file
from xgboost import XGBClassifier
from sensor.ml.metric.classification_metric import get_classification_score,to_dict
from sensor.ml.model.estimator import SensorModel
import pandas as pd
from sensor.ml.model.estimator import TargetValueMapping,ModelResolver
import os,sys

class ModelEvaluation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,model_eval_config:ModelEvaluationConfig,
                    model_trainer_artficat:ModelTrainerArtifact):
        try:

            self.data_validation_artifact=data_validation_artifact
            self.model_eval_config=model_eval_config
            self.model_trainer_artficat=model_trainer_artficat
        except Exception as e:
            raise SensorException(e,sys)
    
    def initialize_model_evaluation(self)->ModelEvaluationArtifact:
        try:
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            #valid train and test file dataframe
            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)

            df=pd.concat([train_df,test_df])
            y_true=df[TARGET_COLUMN]
            y_true.replace(TargetValueMapping().to_dict(),inplace=True)
            df.drop([TARGET_COLUMN],axis=1,inplace=True)

            trained_model_file_path=self.model_trainer_artficat.trained_model_file_path
 
            # To check whether this is best model than the prev one or not
            model_resolver=ModelResolver()

            # If no model is available then this is the best model. So return the artifact
            if not model_resolver.is_model_exists():
                model_eval_artifact=ModelEvaluationArtifact(
                    is_model_accepted=True,
                    best_model_path=None,
                    trained_model_path=trained_model_file_path,
                    best_model_metric_artifact=None,
                    improved_accuracy=None,
                    train_model_metric_artifact=self.model_trainer_artficat.test_metric_artifact
                )

                return model_eval_artifact
            
            # Else compare with the latest available model

            latest_model_path=model_resolver.get_best_model_path()

            latest_model=load_object(latest_model_path)
            trained_model=load_object(trained_model_file_path)

            trained_model_pred=trained_model.predict(df)
            latest_model_pred=latest_model.predict(df)

            trained_model_metrics=get_classification_score(y_true=y_true,y_pred=trained_model_pred)
            latest_model_metrics=get_classification_score(y_true=y_true,y_pred=latest_model_pred)

            improved_accuracy=trained_model_metrics.f1_score-latest_model_metrics.f1_score

            if(improved_accuracy>MODEL_EVALUATION_THRESHOLD_SCORE):
                is_model_accepted=True
            else:
                is_model_accepted=False
            
            model_eval_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, 
                    improved_accuracy=float(improved_accuracy), 
                    best_model_path=latest_model_path, 
                    trained_model_path=trained_model_file_path, 
                    train_model_metric_artifact=trained_model_metrics, 
                    best_model_metric_artifact=latest_model_metrics)
            
            model_eval_report={'is_model_accepted':model_eval_artifact.is_model_accepted,
                               'improved_accuracy':model_eval_artifact.improved_accuracy,
                               'best_model_path':model_eval_artifact.best_model_path,
                               'trained_model_path':model_eval_artifact.trained_model_path,
                               'best_model_metrics':to_dict(model_eval_artifact.best_model_metric_artifact),
                               'trained_model_metrics':to_dict(model_eval_artifact.train_model_metric_artifact)
                               }
            

            logging.info(f"Model evaluation artifact: {model_eval_artifact}")
            write_yaml_file(file_path=self.model_eval_config.report_file_path,content=model_eval_report)

            return model_eval_artifact
            

        except Exception as e:
            raise SensorException(e,sys)
