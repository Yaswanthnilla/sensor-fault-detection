import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sensor.entity.config_entity import DataTransformationConfig
from sensor.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.ml.model.estimator import TargetValueMapping
from sensor.utils.main_utils import save_numpy_array_data,save_object

class DataTransformation:
	def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
		try:
			self.data_transformation_config = data_transformation_config
			self.data_validation_artifact = data_validation_artifact

		except Exception as e:
			raise SensorException(e, sys)

	@staticmethod
	def read_file(file_path) -> pd.DataFrame:

		try:
			dataframe = pd.read_csv(file_path)
			return dataframe
		except Exception as e:
			raise SensorException(e, sys)

	@classmethod
	def get_data_transformer_object(cls) -> Pipeline:
		try:
			robust_scaler = RobustScaler()
			simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
			preprocessor = Pipeline(
				steps=[
				("Imputer", simple_imputer),
				("RobustScaler", robust_scaler)
			 ]
			)
			return preprocessor
		except Exception as e:
			raise SensorException(e, sys)


	def initiatialize_data_transformation(self)->DataTransformationArtifact:
		try:

			train_df=DataTransformation.read_file(self.data_validation_artifact.valid_train_file_path)
			test_df=DataTransformation.read_file(self.data_validation_artifact.valid_test_file_path)
			preprocessor = self.get_data_transformer_object()

			input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
			target_train_df=train_df[TARGET_COLUMN]
			target_train_df=target_train_df.replace(TargetValueMapping().to_dict())




			input_feature_test_df=test_df.drop(columns=[TARGET_COLUMN],axis=1)
			target_test_df=test_df[TARGET_COLUMN]
			target_test_df=target_test_df.replace(TargetValueMapping().to_dict())


			preprocessor_object = preprocessor.fit(input_feature_train_df)
			transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
			transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)

			smt = SMOTETomek(sampling_strategy="minority")

			input_feature_train_final, target_feature_train_final = smt.fit_resample(
					transformed_input_train_feature, target_train_df
				)

			input_feature_test_final, target_feature_test_final = smt.fit_resample(
					transformed_input_test_feature, target_test_df
				)

			train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
			test_arr = np.c_[ input_feature_test_final, np.array(target_feature_test_final)]



			save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path,array=train_arr)
			save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path,array=test_arr)
			save_object(file_path=self.data_transformation_config.transformed_object_file_path,obj=preprocessor_object)
			

			data_transformation_artifact = DataTransformationArtifact(
					transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
					transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
					transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
				)
			logging.info(f"Data transformation artifact: {data_transformation_artifact}")
			return data_transformation_artifact
		except Exception as e:
			raise SensorException(e, sys)











