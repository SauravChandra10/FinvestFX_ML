import os
import sys 
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfigCurrency:
    """
    This dataclass holds configuration settings for data ingestion.
    """
    # The path where the raw data will be saved.
    raw_data_path:str = os.path.join('artifacts','currency_raw.csv')

    # The path where the training data will be saved.
    train_data_path:str = os.path.join('artifacts','currency_train.csv')

    # The path where the testing data will be saved.    
    test_data_path:str = os.path.join('artifacts','currency_test.csv')

class DataIngestionCurrency:
    """
    This class handles data ingestion tasks, including reading, splitting, and saving data.
    """

    def __init__(self):
        """
        Initialize the data ingestion object with a default configuration.
        """
        self.ingestion_config = DataIngestionConfigCurrency()

    def initiate_data_ingestion(self,df):
        """
        Reads a pandas DataFrame, splits it into training and testing sets, and saves them.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data to be ingested.

        Returns:
            tuple: A tuple containing the paths to the saved training and testing data.

        Raises:
            CustomException: If an error occurs during data ingestion.
        """
        try:
            # logging.info("Reading data")

            # Create the directory for the raw data file if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            # Split the DataFrame into training and testing sets (80%/20%)
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            # Save the raw data, training data, and testing data to CSV files
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            # logging.info("Data ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)