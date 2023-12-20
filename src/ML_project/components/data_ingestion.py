import os
import sys
from src.ML_project.exception import Custom_exception
from src.ML_project.logger import logging
import pandas as pd
from src.ML_project.utils import read_sql_data
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class Data_ingestion_config:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')


class Data_ingestion:
    def __init__(self):
        self.ingestion_config=Data_ingestion_config()

    def initiate_Data_ingestion(self):
        try:
            # pass
            # df=read_sql_data()
            df=pd.read_csv(os.path.join('notebook/data','raw.csv'))
            # logging.info("Reading completed mysql database")
            logging.info("reading from mysql database completed")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # creating artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise Custom_exception(e,sys)