import sys, os, io
from io import StringIO
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class PredictPipelineInvoice:
    def __init__(self):
        pass

    def predict(self,df):

        try:

            model_path = 'artifacts/invoice_model.pkl'
            preprocessor_path = 'artifacts/invoice_preprocessor.pkl'

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            # remove duplicates
            df.drop_duplicates(inplace=True)

            # remove NaN

            # covert date into day,month and year
            df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], format='%d-%m-%Y')
            df['Day'] = df['Invoice Date'].dt.day
            df['Month'] = df['Invoice Date'].dt.month
            df['Year'] = df['Invoice Date'].dt.year

            df.drop(columns=['Invoice Date'],inplace=True)

            data_scaled = preprocessor.transform(df)

            pred = model.predict(data_scaled)

            return pred
        
        except Exception as e:
            raise CustomException(e,sys)
        
class PredictPipelineCurrency:
    """
    This class represents a pipeline for making predictions using a trained model.
    """

    def predict(self,features):
        """
        Makes predictions on a given set of features using the loaded model.

        Args:
            features (array-like): The features on which to make predictions.

        Returns:
            array-like: The predicted values.

        Raises:
            CustomException: If an error occurs during prediction.
        """
        try:
            # Define the path to the saved model file
            model_path=os.path.join("artifacts","currency_model.pkl")

            # Load the model from the specified path
            model=load_object(file_path=model_path)

            # Make predictions using the loaded model
            preds=model.predict(features)

            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CurrencyCustomData:
    """
    This class handles processing CSV data uploaded as files.
    """

    def __init__(self,f):
        """
        Initializes the CustomData object with the uploaded file object.

        Args:
            f (file object): The uploaded CSV file object.
        """

        # Read the file content as a string
        stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
        stream.seek(0)
        result = stream.read()

        # Convert the string content to a pandas DataFrame
        df = pd.read_csv(StringIO(result))

        # Process the DataFrame:
        # - Create datetime column from 'Date' with specific format
        # - Extract year and day from the datetime column
        # - Drop 'Date' and the processed datetime column
        df['date'] = pd.to_datetime(df['Date'], format = '%d-%m-%Y')
        df['Year'] = df['date'].dt.year
        df['Day'] = df['date'].dt.day
        df=df.drop(columns=['Date','date'],axis=1)

        # Convert the DataFrame to a NumPy array
        arr = np.array(df)

        # Store the NumPy array as an attribute
        self.arr=arr

    def get_data_as_data_frame(self):
        """
        Returns the stored data as a pandas DataFrame (if needed).

        Raises:
            CustomException: If an error occurs during conversion.
        """

        try:
            return self.arr.copy() # Returning a copy to avoid unintended modifications

        except Exception as e:
            raise CustomException(e, sys)
        
class CurrencyCustomDataJSON:
    """
    This class handles processing data provided in JSON format (assumed to be a DataFrame).
    """

    def __init__(self,df):
        """
        Initializes the CustomDataJSON object with the provided DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data for prediction.
        """

        # Convert the DataFrame to a NumPy array
        arr = np.array(df)

        # Store the NumPy array as an attribute
        self.arr=arr

    def get_data_as_data_frame(self):
        """
        Returns the stored data as a pandas DataFrame (if needed).

        Raises:
            CustomException: If an error occurs during conversion.
        """

        try:
            return self.arr.copy() # Returning a copy to avoid unintended modifications

        except Exception as e:
            raise CustomException(e, sys)