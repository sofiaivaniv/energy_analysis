import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from energy_analysis import (
    DataProcessor,
    EnergyAnalyzer,
    EnergyForecaster,
    EnergyVisualizer
)

class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the actual dataset once for all tests."""
        cls.raw_data = pd.read_csv('resources_04-08.2024_1_.xlsx')
        cls.processor = DataProcessor()

    def test_data_structure(self):
        """Test that the data has the expected structure."""
        expected_columns = [
            'id', 'date', 'organizationName', 'organizationID',
            'providerName', 'providerID', 'meterNumber', 'type',
            'quantity', 'unitName'
        ]
        self.assertTrue(all(col in self.raw_data.columns for col in expected_columns))

    def test_load_data(self):
        """Test loading the actual dataset."""
        loaded_data = self.processor.load_data(
            'resources_04-08.2024_1_.xlsx',
            date_column='date',
            consumption_column='quantity'
        )

        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertTrue(isinstance(loaded_data['date'].iloc[0], pd.Timestamp))
        self.assertEqual(loaded_data['quantity'].dtype, np.float64)

        # Verify actual values from dataset
        self.assertEqual(loaded_data.iloc[0]['quantity'], 104341.0)
        self.assertEqual(loaded_data.iloc[0]['organizationName'], 'ЗДО № 15')

    def test_clean_data(self):
        """Test cleaning the actual dataset."""
        # Create a copy with some artificial missing values to test cleaning
        dirty_data = self.raw_data.copy()
        dirty_data.loc[2, 'quantity'] = np.nan

        clean_data = self.processor.clean_data(dirty_data)

        self.assertFalse(clean_data['quantity'].isna().any())
        self.assertEqual(len(clean_data), len(self.raw_data))

class TestEnergyAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up analyzer with actual dataset."""
        cls.raw_data = pd.read_csv('resources_04-08.2024_1_.xlsx')
        cls.analyzer = EnergyAnalyzer()

    def test_calculate_basic_stats(self):
        """Test statistical calculations on actual data."""
        stats = self.analyzer.calculate_basic_stats(self.raw_data, 'quantity')

        # Test with actual values from dataset
        self.assertAlmostEqual(stats['min'], 0, places=1)  # ЗДО № 47
        self.assertAlmostEqual(stats['max'], 28971.64, places=1)  # ЗДО № 11

        # Verify that mean and median are within expected ranges
        self.assertTrue(1139 <= stats['mean'] <= 1140)
        self.assertTrue(109 <= stats['median'] <= 110 )

    def test_detect_patterns(self):
        """Test pattern detection on actual data."""
        # Convert date to datetime if not already
        data_with_datetime = self.raw_data.copy()
        data_with_datetime['date'] = pd.to_datetime(data_with_datetime['date'])

        patterns = self.analyzer.detect_patterns(
            data_with_datetime,
            consumption_column='quantity',
            date_column='date'
        )

        # Verify patterns for actual institutions
        monthly_pattern = patterns['monthly_pattern']
        self.assertTrue(all(50000 <= value <= 700000 for value in monthly_pattern))

class TestEnergyForecaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up forecaster with actual dataset."""
        cls.raw_data = pd.read_csv('resources_04-08.2024_1_.xlsx')
        cls.raw_data['date'] = pd.to_datetime(cls.raw_data['date'])
        cls.forecaster = EnergyForecaster()

    def test_prepare_data(self):
        """Test data preparation with actual dataset."""
        train_data, test_data = self.forecaster.prepare_data(
            self.raw_data,
            target_column='quantity',
            test_size=0.2
        )

        expected_train_size = int(len(self.raw_data) * 0.8)
        self.assertEqual(len(train_data), expected_train_size)
        self.assertEqual(len(test_data), len(self.raw_data) - expected_train_size)

    def test_forecast_consumption(self):
        """Test forecasting with actual consumption data."""
        # Prepare time series data for specific institution
        zdo_15_data = self.raw_data[self.raw_data['organizationName'] == 'ЗДО № 15']
        zdo_15_data = zdo_15_data.sort_values('date')

        forecast = self.forecaster.forecast_consumption(
            zdo_15_data['quantity'],
            periods=5,
            method='holt-winters',
            seasonal_periods=12  # Monthly seasonality
        )

        self.assertEqual(len(forecast), 5)
        # Verify forecasts are within reasonable range for this institution
        self.assertTrue(all(50000 <= value <= 150000 for value in forecast))

class TestEnergyVisualizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up visualizer with actual dataset."""
        cls.raw_data = pd.read_csv('resources_04-08.2024_1_.xlsx')
        cls.raw_data['date'] = pd.to_datetime(cls.raw_data['date'])
        cls.visualizer = EnergyVisualizer()

    def test_plot_consumption_trend(self):
        """Test plotting actual consumption trends."""
        # Test plotting for specific institution
        zdo_15_data = self.raw_data[self.raw_data['organizationName'] == 'ЗДО № 15'].copy()

        try:
            self.visualizer.plot_consumption_trend(
                zdo_15_data,
                date_column='date',
                consumption_column='quantity'
            )
            plt.close()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Failed to plot actual consumption trend: {str(e)}")

    def test_plot_patterns(self):
        """Test plotting actual consumption patterns."""
        analyzer = EnergyAnalyzer()
        patterns = analyzer.detect_patterns(self.raw_data)

        try:
            self.visualizer.plot_patterns(patterns, 'monthly')
            plt.close()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Failed to plot actual consumption patterns: {str(e)}")

if __name__ == '__main__':
    unittest.main()