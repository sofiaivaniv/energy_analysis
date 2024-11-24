import pandas as pd
import numpy as np
from typing import Union, Optional, List
from datetime import datetime

class DataProcessor:
    '''
    A class for processing and cleaning energy consumption data.

    This class provides methods for data loading, cleaning, and preprocessing
    of energy consumption time series data.

    Examples
    --------
    >>> processor = DataProcessor()
    >>> df = processor.load_data("energy_data.csv")
    >>> clean_df = processor.clean_data(df)
    '''

    def __init__(self):
        """Initialize the DataProcessor."""
        self.data = None

    def load_data(self,
                 file_path: str,
                 date_column: str = 'timestamp',
                 consumption_column: str = 'consumption'
                ) -> pd.DataFrame:
        """
        Load energy consumption data from a file.

        Parameters
        ----------
        file_path : str
            Path to the data file
        date_column : str
            Name of the column containing timestamps
        consumption_column : str
            Name of the column containing consumption values

        Returns
        -------
        pd.DataFrame
            Loaded and validated DataFrame
        """
        try:
            self.data = pd.read_csv(file_path)
            self.data[date_column] = pd.to_datetime(self.data[date_column])

            if not all(col in self.data.columns for col in [date_column, consumption_column]):
                raise ValueError(f"Required columns {date_column} and {consumption_column} not found")

            return self.data
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def clean_data(self,
                  data: Optional[pd.DataFrame] = None,
                  handle_missing: str = 'interpolate'
                 ) -> pd.DataFrame:
        """
        Clean and preprocess the energy consumption data.

        Parameters
        ----------
        data : pd.DataFrame, optional
            DataFrame to clean, if None uses self.data
        handle_missing : str
            Method to handle missing values ('interpolate', 'drop', or 'zero')

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data provided or loaded")
            data = self.data.copy()

        # Handle missing values
        if handle_missing == 'interpolate':
            data = data.interpolate(method='time')
        elif handle_missing == 'drop':
            data = data.dropna()
        elif handle_missing == 'zero':
            data = data.fillna(0)
        else:
            raise ValueError("Invalid handle_missing method")

        # Remove duplicates
        data = data.drop_duplicates()

        return data