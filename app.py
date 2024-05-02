from flask import Flask, request, render_template, jsonify
import pandas as pd
import sys

from src.pipelines.prediction_pipeline import PredictPipelineInvoice, CurrencyCustomData, CurrencyCustomDataJSON, PredictPipelineCurrency
from src.pipelines.training_pipeline import TrainPipelineInvoice, TrainPipelineCurrency
from src.components.data_transformation import DataTransformationInvoice, DataTransformationCurrency
from src.components.model_trainer import ModelTrainerInvoice, ModelTrainerCurrency
from src.components.data_ingestion import DataIngestionCurrency

import io
from io import StringIO
from werkzeug.exceptions import RequestEntityTooLarge

from src.exception import CustomException

app = Flask(__name__)

@app.errorhandler(CustomException)
def handle_my_error(error):
    res={
        "status":False,
        "message":"Error",
        "data":error
    }
    response = jsonify(res)
    return response

@app.route('/currencytraining',methods=["POST"])
def currencytrain():
    try:
        # Check if request is JSON data
        if request.is_json:
            return currencytrainJSON()
        
        # Get the uploaded file from the request
        f = request.files.get('train_file')

        # Check if a file was uploaded and has the correct filename
        if not f:
            res={
                "status":False,
                "message":"Error",
                "data":"No file uploaded or filename is not 'train_file'"
            }
            return res, 400  

        # Get the filename of the uploaded file
        filename = f.filename

        # Check if the file is a CSV file
        if not filename.endswith('.csv'):
            res={
                "status":False,
                "message":"Error",
                "data":"Only CSV files allowed"
            }
            return res, 400
        
        # Read the uploaded file content into a string
        stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
        stream.seek(0)
        result = stream.read()

        # Convert the string content to a pandas DataFrame
        df = pd.read_csv(StringIO(result))

        # Create a DataIngestion object and call its method to initiate data ingestion
        obj=DataIngestionCurrency()
        train_data_path,test_data_path=obj.initiate_data_ingestion(df)

        # Create a DataTransformation object and call its method to transform the data
        data_transformation = DataTransformationCurrency()
        train_arr,test_arr = data_transformation.initiate_data_transormation(train_data_path,test_data_path)

        # Create a ModelTrainer object and call its method to train the model
        modeltrainer = ModelTrainerCurrency()
        print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

        # Prepare successful training response
        res={
            "status":True,
            "message":"Training done successfully!",
            "data":"model created at artifacts/currency_model.pkl"
        }
        return jsonify(res)
    
    except RequestEntityTooLarge as e:
        # Handle file size exceeding the limit
        res = {
            "status": False,
            "message": "Error!",
            "data": 'File size exceeds 5 MB'
        }
        return res, 413
    
    except Exception as e:
        # Handle other exceptions
        error = CustomException(e,sys).error_message
        return handle_my_error(error)

def currencytrainJSON():
    try:
        """
        This function handles training the model when data is provided in JSON format through a POST request.

        Returns:
            JSON response indicating success or failure and any error messages.
        """

        # Get the JSON data from the request
        data = request.json

        # Check if the request has any data
        if not data:
            # Prepare error response for missing JSON data
            res = {
                'status' : False,
                'message' : 'Error',
                'data' : 'Invalid JSON data'
            }
            return jsonify(res), 400
        
        # Define the required keys expected in the JSON data for training
        required_keys = ['Month', 'Weekday', 'EUR', 
                'JPY', 'BGN', 'CZK', 'DKK', 'GBP', 'HUF', 'PLN', 'RON', 'SEK', 'CHF', 'NOK', 'TRY', 'AUD', 'BRL', 'CAD', 'CNY', 'HKD', 'IDR', 'KRW', 'MXN', 'MYR', 'NZD', 'PHP', 'SGD', 'THB', 'ZAR',
                'Year', 'Day']
        
        # Find any missing keys in the provided JSON data
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            res = {
                'status': False,
                'message' : 'Error',
                'data': f'Missing keys in JSON data: {missing_keys}'
            }
            return jsonify(res), 400

        # Convert the JSON data to a pandas DataFrame
        df = pd.DataFrame([data])

        # Create a DataIngestion object and call its method to initiate data ingestion
        obj=DataIngestionCurrency()
        train_data_path,test_data_path=obj.initiate_data_ingestion(df)

        # Create a DataTransformation object and call its method to transform the data
        data_transformation = DataTransformationCurrency()
        train_arr,test_arr = data_transformation.initiate_data_transormation(train_data_path,test_data_path)

        # Create a ModelTrainer object and call its method to train the model
        modeltrainer = ModelTrainerCurrency()
        print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

        # Prepare successful training response
        res={
            "status":True,
            "message":"Training done successfully",
            "data":"model created at artifacts/currency_model.pkl"
        }
        return jsonify(res)

    except Exception as e:
        error = CustomException(e,sys).error_message
        return handle_my_error(error)

@app.route('/currencyprediction', methods=['POST'])
def currencypredict():
    """
    This function handles generating predictions on currency data. It supports receiving data in two formats:
        - JSON format through request body
        - CSV file upload

    Returns:
        JSON response containing the predicted values and success/error messages.
    """
    try:
        # Check if request data is in JSON format
        if request.is_json:
            return currencypredictJSON()
        
        # Handle case where data is uploaded as a CSV file
        # Get the uploaded file from the request
        f = request.files.get('test_file')

        # Check if a file was uploaded and has the correct filename
        if not f:
            res={
                "status":False,
                "message":"Error",
                "data":"No file uploaded or filename is not 'test_file'"
            }
            return res, 400  

        # Get the filename of the uploaded file
        filename = f.filename

        # Check if the file is a CSV file
        if not filename.endswith('.csv'):
            res={
                "status":False,
                "message":"Error",
                "data":"Only CSV files allowed"
            }
            return res, 400

        # Use CustomData class to extract data array from the uploaded file
        data = CurrencyCustomData(f).arr

        # Use the PredictPipeline to generate predictions on the data
        prediction = PredictPipelineCurrency().predict(data)

        # Convert the prediction results to a list
        prediction_list = prediction.tolist()

        # Prepare successful prediction response with the predictions
        res={
            "status":True,
            "message":"Prediction done successfully",
            "data":prediction_list
        }
        return jsonify(res)
    
    except RequestEntityTooLarge as e:
        # Handle file size exceeding the limit
        res = {
            "status": False,
            "message": "Error",
            "data": 'File size exceeds 5 MB'
        }
        return res, 413
    
    except Exception as e:
        error = CustomException(e,sys).error_message
        return handle_my_error(error)

def currencypredictJSON():
    """
    This function handles generating predictions when data is provided in JSON format through a POST request.

    Returns:
        JSON response containing the predicted values and success/error messages.
    """
    try:
        # Get the JSON data from the request
        data = request.json

        # Check if the request has any data
        if not data:
            res = {
                'status' : False,
                'message' : 'Error',
                'data' : 'Invalid JSON data'
            }
            return jsonify(res), 400
        
        # Define the required keys expected in the JSON data for prediction
        required_keys = ['Month', 'Weekday', 'EUR', 
                'JPY', 'BGN', 'CZK', 'DKK', 'GBP', 'HUF', 'PLN', 'RON', 'SEK', 'CHF', 'NOK', 'TRY', 'AUD', 'BRL', 'CAD', 'CNY', 'HKD', 'IDR', 'KRW', 'MXN', 'MYR', 'NZD', 'PHP', 'SGD', 'THB', 'ZAR',
                'Year', 'Day']
        
        # Find any missing keys in the provided JSON data
        missing_keys = [key for key in required_keys if key not in data]

        # Check if any keys are missing from the JSON data 
        if missing_keys:
            res = {
                'status': False,
                'message' : 'Error',
                'data': f'Missing keys in JSON data: {missing_keys}'
            }
            return jsonify(res), 400
        
        # Convert the JSON data to a pandas DataFrame
        df = pd.DataFrame([data])

        # Create a CustomDataJSON object using the DataFrame and extract the data array
        data = CurrencyCustomDataJSON(df).arr

        # Use the PredictPipeline to generate predictions on the data
        prediction = PredictPipelineCurrency().predict(data)

        # Convert the prediction results to a list
        prediction_list = prediction.tolist()

        # Prepare successful prediction response with the predictions
        res={
            "status":True,
            "message":"Prediction done successfully",
            "data":prediction_list
        }

        return jsonify(res)
        
    except Exception as e:
        error = CustomException(e,sys).error_message
        return handle_my_error(error)

@app.route('/invoicetraining', methods=['POST'])
def invoicetrain():
        try:
            if request.is_json:
                return invoicetrainJSON()

            if 'train_file' not in request.files:
                res={
                    'status' : False,
                    'message' : 'Error',
                    'data' : 'No file part'
                }
                return jsonify(res),400
            
            file = request.files['train_file']

            if file.filename == '':
                res={
                    'status' : False,
                    'message' : 'Error',
                    'data' : 'No selected file'
                }
                return jsonify(res),400
            
            df = pd.read_excel(file)

            obj = TrainPipelineInvoice(df)
            data_path = obj.initiate_data_ingestion()

            data_transformation = DataTransformationInvoice()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(data_path)

            model_trainer = ModelTrainerInvoice()
            model_name, model_score = model_trainer.initiate_model_trainer(train_arr,test_arr)

            res={
                "status":True,
                "message": "Training done successfully",
                "data": f"model created at artifacts/invoice_model.pkl, best model is {model_name} and rmse is {model_score}"
            }

            return jsonify(res)


        except Exception as e:
            error = CustomException(e,sys).error_message
            return handle_my_error(error)
        
def invoicetrainJSON():
    try:
        data = request.json

        if not data:
            res={
                'status' : False,
                'message' : 'Error',
                'data' : 'Invalid JSON data'
            }
            return jsonify(res),400  
        
        required_keys = ['Customer Name', 'Invoice Date', 'Credit Terms', 
                        'Invoice Currency', 'Invoice Amount',
                        'Actual Delay over and above Agreed Credit Terms']
            
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            res = {
                'status': False,
                'message' : 'Error',
                'data': f'Missing keys in JSON data: {missing_keys}'
            }
            return jsonify(res), 400
        
        df = pd.DataFrame([data])
        obj = TrainPipelineInvoice(df)
        data_path = obj.initiate_data_ingestion()

        data_transformation = DataTransformationInvoice()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(data_path)

        model_trainer = ModelTrainerInvoice()
        model_trainer.initiate_model_trainer(train_arr,test_arr)

        res={
            "status":True,
            "message":"Training done successfully",
            "data":"model created at artifacts/invoice_model.pkl"
        }

        return jsonify(res)
              
    except Exception as e:
        error = CustomException(e,sys).error_message
        return handle_my_error(error)

@app.route('/invoiceprediction', methods=['POST'])
def invoicepredict():
        try:
            if request.is_json:
                return invoicepredictJSON()
            
            if 'test_file' not in request.files:
                res={
                    'status' : False,
                    'message' : 'Error',
                    'data' : 'No file part'
                }
                return jsonify(res),400
            
            file = request.files['test_file']

            if file.filename == '':
                res={
                    'status' : False,
                    'message' : 'Error',
                    'data' : 'No selected file'
                }
                return jsonify(res),400
            
            df = pd.read_csv(file)

            predict_pipeline = PredictPipelineInvoice()

            pred = predict_pipeline.predict(df)

            prediction_list = pred.tolist()

            res = {
                 'status' : True,
                 'message' : 'Prediction done successfully',
                 'data' : prediction_list
            }

            return jsonify(res)

        except Exception as e:
            error = CustomException(e,sys).error_message
            return handle_my_error(error)

def invoicepredictJSON():
        try:
            data = request.json

            if not data:
                res={
                    'status' : False,
                    'message' : 'Error',
                    'data' : 'Invalid JSON data'
                }
                return jsonify(res),400
            
            required_keys = ['Customer Name', 'Invoice Date', 'Credit Terms', 
                            'Invoice Currency', 'Invoice Amount']
            
            missing_keys = [key for key in required_keys if key not in data]

            if missing_keys:
                res = {
                    'status': False,
                    'message' : 'Error',
                    'data': f'Missing keys in JSON data: {missing_keys}'
                }
                return jsonify(res), 400
        
            df = pd.DataFrame([data])

            predict_pipeline = PredictPipelineInvoice()

            pred = predict_pipeline.predict(df)

            prediction_list = pred.tolist()

            res={
                'status' : True,
                'message' : 'Prediction done successfully',
                'data' : prediction_list
            }
            
            return jsonify(res)

        except Exception as e:
            error = CustomException(e,sys).error_message
            return handle_my_error(error)

if __name__ == '__main__':
     app.run(host = '0.0.0.0', debug = True, port = 5000)