from sensor.configuration.mongo_db_connection import MongoDBClient
import os,sys
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH,SAVED_MODEL_DIR
from fastapi import FastAPI,UploadFile,File
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response
from sensor.ml.model.estimator import ModelResolver,TargetValueMapping
from sensor.utils.main_utils import load_object,read_yaml_file
from fastapi.middleware.cors import CORSMiddleware
from sensor.constant.application import APP_HOST,APP_PORT
from sensor.constant import prediction_pipeline
import pandas as pd,numpy as np
from datetime import datetime
import shutil
from sensor.constant.training_pipeline import SCHEMA_DROP_COLS,SCHEMA_FILE_PATH,TARGET_COLUMN,TRAINING_BUCKET_NAME
from sensor.ml.metric.classification_metric import get_classification_score,to_dict
from sensor.cloud_storage.s3_syncer import S3Sync
from sensor.constant.s3_bucket import PREDICTION_BUCKET_NAME



env_file_path=os.path.join(os.getcwd(),"env.yaml")

def set_env_variable(env_file_path):

    if os.getenv('MONGO_DB_URL',None) is None:
        env_config = read_yaml_file(env_file_path)
        os.environ['MONGO_DB_URL']=env_config['MONGO_DB_URL']



app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


df=None
@app.post("/uploadcsv")
def upload_csv_file_for_prediction(csv_file: UploadFile = File(...)):
    # timestamp = round(datetime.now().timestamp())
    # testing_file_path=os.path.join(prediction_pipeline.TEST_FILES_DIR,f"{prediction_pipeline.PREDICTION_FILE_NAME}")
    # print(testing_file_path)
    # os.makedirs(os.path.dirname(testing_file_path),exist_ok=True)
    df = pd.read_csv(csv_file.file)
    # df.to_csv(testing_file_path,index=False)
    # s3_sync=S3Sync()
    # aws_bucket_url=f"s3://{TRAINING_BUCKET_NAME}/{prediction_pipeline.TEST_FILES_DIR}"
    # s3_sync.sync_folder_to_s3(folder=prediction_pipeline.TEST_FILES_DIR,aws_bucket_url=aws_bucket_url)
    file_path=f"s3://{TRAINING_BUCKET_NAME}/{prediction_pipeline.TEST_FILES_DIR}/{prediction_pipeline.PREDICTION_FILE_NAME}"
    print(file_path)
    df.to_csv(file_path,index=False)
    return Response(f"{csv_file.filename} uploaded succesfully. Go for Prediction")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainPipeline()
        if train_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

    

@app.get("/predict")
async def predict_route():
    try:
        # if not os.path.exists(prediction_pipeline.TEST_FILES_DIR):
        #     return Response("Upload the csv file to be predicted")
        # timestamps = os.listdir(prediction_pipeline.TEST_FILES_DIR)
        # if len(timestamps)==0:
        #     return Response("Upload the csv file to be predicted")
        # timestamp=os.listdir(prediction_pipeline.TEST_FILES_DIR)[0]
        # file=os.listdir(os.path.join(prediction_pipeline.TEST_FILES_DIR,f"{timestamp}"))[0]
        # test_file_path=os.path.join(prediction_pipeline.TEST_FILES_DIR,f"{timestamp}",f"{file}")
        # # print(test_file_path)
        # df=pd.read_csv(test_file_path)
        test_file_path=f's3://{TRAINING_BUCKET_NAME}/{prediction_pipeline.TEST_FILES_DIR}/{prediction_pipeline.PREDICTION_FILE_NAME}'
        print(test_file_path)
        df=pd.read_csv(test_file_path)
        print(df.head())
        if df is None:
            return Response("Upload the csv file to be predicted")
        else:
            schema_file=read_yaml_file(SCHEMA_FILE_PATH)
            df.drop(schema_file[SCHEMA_DROP_COLS],axis=1,inplace=True)
            target_val_mapping=TargetValueMapping()
            target_col=df[TARGET_COLUMN].replace(target_val_mapping.to_dict())
            df.drop([TARGET_COLUMN],axis=1,inplace=True)
            df.replace({"na": np.nan}, inplace=True)
            model_resolver=ModelResolver()
            model_path=model_resolver.get_best_model_path()
            model=load_object(model_path)
            y_pred=pd.Series(model.predict(df))
            pred_metric=get_classification_score(y_true=target_col,y_pred=y_pred)
            y_pred=y_pred.replace(target_val_mapping.reverse_mapping())
            print(to_dict(pred_metric))
            # shutil.rmtree(prediction_pipeline.TEST_FILES_DIR)
            command = f"aws s3 rm s3://{TRAINING_BUCKET_NAME}/{prediction_pipeline.TEST_FILES_DIR} --recursive"
            os.system(command)
            return Response(f"Prediction successful and the predictions are {y_pred}")
    except Exception as e:
        command = f"aws s3 rm s3://{PREDICTION_BUCKET_NAME}/{prediction_pipeline.TEST_FILES_DIR} --recursive"
        os.system(command)
        # shutil.rmtree(prediction_pipeline.TEST_FILES_DIR)
        return Response(f"Error Occurred! {e}")

        
    


if __name__=="__main__":
    set_env_variable(env_file_path=env_file_path)
    app_run(app, host=APP_HOST, port=APP_PORT)



