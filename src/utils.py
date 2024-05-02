import os
import sys
import numpy as np
import dill

from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            pred = model.predict(X_test)

            r2 = r2_score(y_test,pred)
            mae = mean_absolute_error(y_test,pred)
            rmse = np.sqrt(mean_squared_error(y_test,pred))

            logging.info(f"{model} r2:{r2}, mae:{mae}, rmse:{rmse}")

            report[list(models.keys())[i]] = rmse

        return report
    except Exception as e:
        raise CustomException(e,sys)