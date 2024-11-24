import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional

class EnergyAnalyzer:
    '''
    A class for analyzing energy consumption patterns and statistics.

    This class provides methods for statistical analysis and pattern
    detection in energy consumption data.

    Examples
    --------
    >>> analyzer = EnergyAnalyzer()
    >>> stats = analyzer.calculate_basic_stats(consumption_data)
    >>> patterns = analyzer.detect_patterns(consumption_data)
    '''

    def __init__(self):
        """Initialize the EnergyAnalyzer."""
        pass

    def calculate_basic_stats(self,
                            data: pd.DataFrame,
                            consumption_column: str = 'consumption'
                           ) -> Dict[str, float]:
        """
        Calculate basic statistical measures of energy consumption.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame with consumption data
        consumption_column : str
            Name of the consumption column

        Returns
        -------
        Dict[str, float]
            Dictionary containing statistical measures
        """
        if consumption_column not in data.columns:
            raise ValueError(f"Column {consumption_column} not found in data")

        stats_dict = {
            'mean': float(data[consumption_column].mean()),
            'median': float(data[consumption_column].median()),
            'std': float(data[consumption_column].std()),
            'min': float(data[consumption_column].min()),
            'max': float(data[consumption_column].max())
        }

        return stats_dict

    def detect_patterns(self,
                       data: pd.DataFrame,
                       consumption_column: str = 'consumption',
                       date_column: str = 'timestamp'
                      ) -> Dict[str, pd.Series]:
        """
        Detect patterns in energy consumption data.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame with consumption data
        consumption_column : str
            Name of the consumption column
        date_column : str
            Name of the date column

        Returns
        -------
        Dict[str, pd.Series]
            Dictionary containing detected patterns
        """
        data[date_column] = pd.to_datetime(data[date_column])

        patterns = {
            'daily_pattern': data.groupby(data[date_column].dt.hour)[consumption_column].mean(),
            'weekly_pattern': data.groupby(data[date_column].dt.dayofweek)[consumption_column].mean(),
            'monthly_pattern': data.groupby(data[date_column].dt.month)[consumption_column].mean()
        }

        return patterns