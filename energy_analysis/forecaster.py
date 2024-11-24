import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Tuple, Optional, Union

class EnergyForecaster:
    '''
    A class for forecasting energy consumption.

    This class provides methods for time series forecasting using
    various statistical and machine learning models.

    Examples
    --------
    >>> forecaster = EnergyForecaster()
    >>> forecast = forecaster.forecast_consumption(historical_data, periods=24)
    '''

    def __init__(self):
        """Initialize the EnergyForecaster."""
        self.model = None

    def prepare_data(self,
                    data: pd.DataFrame,
                    target_column: str = 'consumption',
                    test_size: float = 0.2
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for forecasting.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame with time series data
        target_column : str
            Name of the target column
        test_size : float
            Proportion of data to use for testing

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Training and testing DataFrames
        """
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")

        train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
        return train_data, test_data

    def forecast_consumption(self,
                           data: pd.DataFrame,
                           periods: int,
                           method: str = 'holt-winters',
                           seasonal_periods: Optional[int] = 24
                          ) -> pd.Series:
        """
        Forecast future energy consumption.

        Parameters
        ----------
        data : pd.DataFrame
            Historical consumption data
        periods : int
            Number of periods to forecast
        method : str
            Forecasting method to use
        seasonal_periods : int, optional
            Number of periods in a seasonal cycle

        Returns
        -------
        pd.Series
            Forecasted values
        """
        if method == 'holt-winters':
            model = ExponentialSmoothing(
                data,
                seasonal_periods=seasonal_periods,
                trend='add',
                seasonal='add'
            )
            self.model = model.fit()
            forecast = self.model.forecast(periods)
            return forecast
        else:
            raise ValueError(f"Unsupported forecasting method: {method}")