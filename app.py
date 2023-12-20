from src.ML_project.logger import logging
from src.ML_project.exception import Custom_exception
from src.ML_project.components.data_ingestion import Data_ingestion
from src.ML_project.components.data_ingestion import Data_ingestion_config
from src.ML_project.components.data_transformation import Data_transformation
from src.ML_project.components.data_transformation import Data_transformation_config
from src.ML_project.components.model_trainer import model_trainer
from src.ML_project.components.model_trainer import model_trainer_config
import sys


if __name__=="__main__":
    logging.info("The execution has started")

    try:
        # a=1/0
        data_ingestion=Data_ingestion()
        data_ingestion.initiate_Data_ingestion()
        # data_ingestion_config=Data_ingestion_config()

        #data_ingestion_config=DataIngestionConfig()
        # data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_Data_ingestion()

        #data_transformation_config=DataTransformationConfig()
        data_transformation=Data_transformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        ## Model Training

        model_train=model_trainer()
        # model_train.initiate_model_trainer(train_arr,test_arr)
        print(model_train.initiate_model_trainer(train_arr,test_arr))
    except Exception as e:
        logging.info("Custom exception")
        raise Custom_exception(e,sys)