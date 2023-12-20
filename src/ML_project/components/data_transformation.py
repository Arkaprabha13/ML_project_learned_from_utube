import sys
from dataclasses import dataclass
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.ML_project.exception import Custom_exception
from src.ML_project.logger import logging
import os

import pickle
from src.ML_project.utils import save_object

@dataclass
class Data_transformation_config:
    preproccesor_object_file_path=os.path.join('artifacts','preprocessor.pki')

class Data_transformation:
    def __init__(self):
        self.Data_transformation_config=Data_transformation_config()

    def get_data_transformer_object(self):
        try:
            # pass
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scalar',StandardScaler())

            ])
            cat_pipeline=Pipeline(steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder",OneHotEncoder()),
            ("scaler",StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns:{categorical_columns}")
            logging.info(f"Numerical Columns:{numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]

            )
            return preprocessor


        except Exception as e:
            logging.info("Exception occured in data transforamtion file")
            raise Custom_exception(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            # pass
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading train and test data")

            preprocessor_object=self.get_data_transformer_object()

            target_column_name='math_score'
            numerical_columns = ["writing_score", "reading_score"]
            
            # train data set
            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            # test data set
            input_features_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("applying preproccesing on training and test data")

            input_feature_train_arr=preprocessor_object.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessor_object.transform(input_features_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preproccesing object")

            save_object(

                file_path=self.Data_transformation_config.preproccesor_object_file_path,
                obj=preprocessor_object
            )

            return (
                train_arr,
                test_arr,
                self.Data_transformation_config.preproccesor_object_file_path
            )



        except Exception as e:
            raise Custom_exception(sys,e)

