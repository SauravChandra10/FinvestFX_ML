import os
import sys 
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfigInvoice:
    preprocessor_obj_file_path = os.path.join('artifacts','invoice_preprocessor.pkl')


class DataTransformationInvoice:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfigInvoice()

    def get_data_transformer_object(self):
        try:
            categorical_columns = [
                'Invoice Currency'
            ]
            numerical_columns   = [
                'Customer Name',
                'Credit Terms',
                'Invoice Amount',
                'Day',
                'Month',
                'Year'
            ]

            num_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info("Pipeline has been completed")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            logging.info("Preprocessing done")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,data_path):
        try:
            df = pd.read_csv(data_path)
            logging.info("Read data into dataframe")

            # remove duplicates
            df.drop_duplicates(inplace=True)

            # remove NaN
            df = df.dropna(subset=['Actual Delay over and above Agreed Credit Terms'])

            # covert date into day,month and year
            df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], format='%Y-%m-%d')
            df['Day'] = df['Invoice Date'].dt.day
            df['Month'] = df['Invoice Date'].dt.month
            df['Year'] = df['Invoice Date'].dt.year

            df.drop(columns=['Invoice Date'],inplace=True)

            X=df.drop(['Actual Delay over and above Agreed Credit Terms'],axis=1)
            y=df['Actual Delay over and above Agreed Credit Terms']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            preprocessing_obj = self.get_data_transformer_object()

            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            train_arr = np.c_[
                X_train_arr,np.array(y_train)
            ]

            test_arr = np.c_[
                X_test_arr,np.array(y_test)
            ]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)

class DataTransformationCurrency:
    """
    This class handles data transformation steps for training and testing data.
    """
    def initiate_data_transormation(self,train_path,test_path):
        """
        Performs data transformation on training and testing CSV files.

        Args:
            train_path (str): The path to the training CSV file.
            test_path (str): The path to the testing CSV file.

        Returns:
            tuple: A tuple containing the transformed training and testing data as NumPy arrays.

        Raises:
            CustomException: If an error occurs during data transformation.
        """
        try:
            # Read training and testing data as pandas DataFrames
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            # logging.info("Reading the train and test file")

            # Create datetime, year, and day columns from the 'Date' column
            train_df['date']=pd.to_datetime(train_df['Date'], format = '%Y-%m-%d')
            train_df['Year'] = train_df['date'].dt.year
            train_df['Day'] = train_df['date'].dt.day

            test_df['date']=pd.to_datetime(test_df['Date'], format = '%Y-%m-%d')
            test_df['Year'] = test_df['date'].dt.year
            test_df['Day'] = test_df['date'].dt.day

            # Separate target feature from training and testing DataFrames
            target_feature_train_df=train_df['INR']
            train_df=train_df.drop(columns=['INR','Date','date'],axis=1)

            target_feature_test_df=test_df['INR']
            test_df=test_df.drop(columns=['INR','Date','date'],axis=1)

            # logging.info("Applying Preprocessing on training and test dataframe")

            # Combine features and target feature back into NumPy arrays
            train_arr = np.c_[train_df,np.array(target_feature_train_df)]
            test_arr = np.c_[test_df,np.array(target_feature_test_df)]

            return (
                train_arr,
                test_arr,
            )
        
        except Exception as e:
            raise CustomException(e,sys)