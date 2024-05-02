import os
import sys
import numpy as np

from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


from src.utils import save_object, evaluate_models
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfigInvoice:
    trained_model_file_path = os.path.join('artifacts','invoice_model.pkl')

class ModelTrainerInvoice:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfigInvoice()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Gradient Boosting Regressor" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression()
            }

            params = {
                "Gradient Boosting Regressor":{
                    "learning_rate" : [0.001],
                    "max_depth" : [8],
                    "n_estimators" : [2000],
                    "subsample" : [0.5]
                },
                "Linear Regression":{}
            }

            model_report:dict = evaluate_models(
                X_train,y_train,X_test,y_test,models,params
            )

            best_score = min(model_report.values())

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_score)]

            best_model = models[best_model_name]

            logging.info(f"{best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            return best_model_name,best_score

        except Exception as e:
            raise CustomException(e,sys)
        
@dataclass
class ModelTrainerConfigCurrency:
    """
    This dataclass holds configuration settings for the model trainer.
    """

    trained_model_file_path = os.path.join('artifacts','currency_model.pkl')
    # Path to the file where the trained model will be saved.


class ModelTrainerCurrency:
    """
    This class handles training a machine learning model.
    """

    def __init__(self):
        """
        Initialize the model trainer with a default configuration.
        """
        self.model_trainer_config = ModelTrainerConfigCurrency()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        """
        Trains a model on provided training data and evaluates its performance.

        Args:
            train_arr (np.array): The training data as a NumPy array.
            test_arr (np.array): The testing data as a NumPy array.

        Returns:
            float: The root mean squared error (RMSE) of the best model on the test data.

        Raises:
            CustomException: If an error occurs during training or evaluation.
        """
        try:
            # logging.info("Split train and test input data")

            # Separate features and target values from training and testing data
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            # Define a dictionary containing candidate models (here, RandomForestRegressor)
            models={
                "RandomForest": RandomForestRegressor()
            }

            # Define hyperparameter grids for each model in another dictionary
            params={
                "RandomForest":{
                    'n_estimators':[400],
                    'min_samples_split':[2],
                    'min_samples_leaf':[1],
                    'max_features':['sqrt'],
                    'max_depth':[None],
                    'bootstrap':[False]
                }
            }

            # Use the evaluate_models function to evaluate all models with their hyperparameter grids
            model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,models,params)

             # Find the model with the best score (lowest RMSE) based on the evaluation report
            best_model_score = max(sorted(model_report.values()))

            # Save the best model to the specified path using the save_object function
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            # Make predictions on the test data using the best model
            best_model = models[best_model_name]

            # logging.info("Best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            
            # Calculate the root mean squared error (RMSE) on the test data
            return np.sqrt(mean_squared_error(y_test,predicted))

        except Exception as e:
            raise CustomException(e,sys)