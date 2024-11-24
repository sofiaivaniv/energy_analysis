import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional, Tuple, List

class EnergyVisualizer:
    '''
    A class for visualizing energy consumption data and analysis results.

    This class provides methods for creating various plots and visualizations
    of energy consumption patterns and forecasts.

    Examples
    --------
    >>> visualizer = EnergyVisualizer()
    >>> visualizer.plot_consumption_trend(consumption_data)
    >>> visualizer.plot_daily_pattern(daily_patterns)
    '''

    def __init__(self):
        """Initialize the EnergyVisualizer."""
        self.style = 'seaborn'
        plt.style.use(self.style)

    def plot_consumption_trend(self,
                             data: pd.DataFrame,
                             date_column: str = 'timestamp',
                             consumption_column: str = 'consumption',
                             figsize: Tuple[int, int] = (12, 6)
                            ) -> None:
        """
        Plot energy consumption trend over time.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame with consumption data
        date_column : str
            Name of the date column
        consumption_column : str
            Name of the consumption column
        figsize : Tuple[int, int]
            Figure size in inches
        """
        plt.figure(figsize=figsize)
        plt.plot(data[date_column], data[consumption_column])
        plt.title('Energy Consumption Trend')
        plt.xlabel('Date')
        plt.ylabel('Consumption')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

    def plot_patterns(self,
                     patterns: dict,
                     pattern_type: str = 'daily',
                     figsize: Tuple[int, int] = (10, 6)
                    ) -> None:
        """
        Plot consumption patterns.

        Parameters
        ----------
        patterns : dict
            Dictionary containing pattern data
        pattern_type : str
            Type of pattern to plot ('daily', 'weekly', 'monthly')
        figsize : Tuple[int, int]
            Figure size in inches
        """
        if pattern_type not in patterns:
            raise ValueError(f"Pattern type {pattern_type} not found in patterns")

        plt.figure(figsize=figsize)
        pattern_data = patterns[pattern_type]

        if pattern_type == 'daily':
            x_label = 'Hour of Day'
        elif pattern_type == 'weekly':
            x_label = 'Day of Week'
        else:
            x_label = 'Month'

        plt.plot(pattern_data.index, pattern_data.values, marker='o')
        plt.title(f'{pattern_type.capitalize()} Consumption Pattern')
        plt.xlabel(x_label)
        plt.ylabel('Average Consumption')
        plt.grid(True)
        plt.tight_layout()