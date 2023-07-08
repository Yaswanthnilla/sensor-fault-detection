from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.config_entity import DataIngestionConfig
from sensor.entity.artifact_entity import DataIngestionArtifact
import os,sys
from pandas import DataFrame
from sensor.data_access.sensor_data import SensorData
from sklearn.model_selection import train_test_split
from sensor.utils.main_utils import read_yaml_file
from sensor.constant.training_pipeline import SCHEMA_DROP_COLS,SCHEMA_FILE_PATH


class DataIngestion:

	def __init__(self,data_ingestion_config:DataIngestionConfig):
		try:
			self.data_ingestion_config=data_ingestion_config
			# self._schema_config=read_yaml_file("C:/Users/N YASWANTH KUMAR/OneDrive/Desktop/sensor-fault-detection/sensor/config/schema.yaml")
			logging.info(SCHEMA_FILE_PATH)
			self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)

		except Exception as e:
			raise SensorException(e,sys)




	

	def export_data_into_feature_store(self)->DataFrame:

		try:
			logging.info("Exporting data from mongodb to feature_store folder")

			# Creating object to the SensorData() class in data_access.py file which is used to export data
			# as a datframe from mongodb
			sensor_data=SensorData()
			dataframe=sensor_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)

			#creating feature_store folder and saving the sensor.csv file in it.

			feature_store_file_path=self.data_ingestion_config.feature_store_file_path
			dir_path=os.path.dirname(feature_store_file_path)
			os.makedirs(dir_path,exist_ok=True)


			dataframe.to_csv(feature_store_file_path,index=False,header=True)

			return dataframe

		except Exception as e:
			raise SensorException(e,sys)




	def split_data_as_train_test(self,dataframe:DataFrame):

		try:
			train_data,test_data=train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio)
			logging.info("Performed train test split on the dataframe")


			# dir_path is the path to the ingested folder in artifact folder
			# os.path.dirname will give the ingested folder path

			dir_path=os.path.dirname(self.data_ingestion_config.training_file_path)
			dir_path1=os.path.dirname(self.data_ingestion_config.testing_file_path)

			print(dir_path)
			print(dir_path1)
			os.makedirs(dir_path,exist_ok=True)

			logging.info("Exporting train and test file path")

			train_data.to_csv(self.data_ingestion_config.training_file_path,header=True,index=False)
			test_data.to_csv(self.data_ingestion_config.testing_file_path,header=True,index=False)

			logging.info("Exporting train and test data completed")

			logging.info(
				"Exited split_data_as_train_test method of Data_Ingestion class"
			)

		except Exception as e:
			raise SensorException(e,sys)
			


	def initialize_data_ingestion(self)->DataIngestionArtifact:

		try:

			dataframe=self.export_data_into_feature_store()

			logging.info(f"dropped column = {self._schema_config[SCHEMA_DROP_COLS]}")
			dataframe.drop(self._schema_config[SCHEMA_DROP_COLS],axis=1,inplace=True)

			self.split_data_as_train_test(dataframe=dataframe)

			data_ingestion_artifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,test_file_path=self.data_ingestion_config.testing_file_path)
			return data_ingestion_artifact

		except Exception as e:
			raise SensorException(e,sys)







	


