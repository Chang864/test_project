# test.py
"""Unit tests for DeviceTestAnalyzer"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# add file path
current_path = os.getcwd()
sys.path.insert(0, current_path)
from Analyzer import DeviceTestAnalyzer


class TestDeviceTestAnalyzer:
    """Test class for DeviceTestAnalyzer"""
    
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'name': [f'device_{i//10}' for i in range(n_samples)],
            'test_status': np.random.choice(['pass', 'fail'], n_samples, p=[0.85, 0.15]),
            'mark': np.random.choice(['up', 'down'], n_samples, p=[0.6, 0.4]),
            'angle_x': np.random.normal(0, 1, n_samples),
            'angle_y': np.random.normal(0, 1, n_samples),
            'angle_z': np.random.normal(0, 1, n_samples),
            'number_of_drop': np.random.randint(1, 10, n_samples),
            'dropheight': np.ones(n_samples)
        }
        
        return pd.DataFrame(data)
    
    def test_initialization(self, sample_data):
        """Test analyzer initialization"""
        analyzer = DeviceTestAnalyzer(sample_data)
        assert analyzer.data.shape == (100, 8)
        assert isinstance(analyzer.results, dict)
        assert len(analyzer.results) == 0
    
    def test_exploratory_analysis(self, sample_data):
        """Test exploratory analysis"""
        analyzer = DeviceTestAnalyzer(sample_data)
        stats = analyzer.exploratory_analysis()
        
        assert 'total_tests' in stats
        assert 'failure_rate' in stats
        assert 0 <= stats['failure_rate'] <= 1
        assert stats['total_tests'] == 100
    
    def test_hypothesis_test(self, sample_data):
        """Test hypothesis testing"""
        analyzer = DeviceTestAnalyzer(sample_data)
        result = analyzer.hypothesis_test()
        
        if result:  
            assert 'sample_sizes' in result
            assert 'z_test' in result
            assert 'p_value' in result['z_test']
    
    def test_train_prediction_model(self, sample_data):
        """Test model training"""
        analyzer = DeviceTestAnalyzer(sample_data)
        model = analyzer.train_prediction_model()
        
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(model, RandomForestClassifier)
        assert hasattr(model, 'predict')
        assert 'model_performance' in analyzer.results


def test_integration():
    """Integration test: full analysis workflow"""
    # Build test dataset
    np.random.seed(42)
    data = pd.DataFrame({
        'name': ['device_1', 'device_2'] * 25,
        'test_status': ['pass', 'fail'] * 25,
        'mark': ['up', 'down'] * 25,
        'angle_x': np.random.normal(0, 1, 50),
        'angle_y': np.random.normal(0, 1, 50),
        'angle_z': np.random.normal(0, 1, 50),
        'number_of_drop': list(range(1, 51)),
        'dropheight': [1] * 50
    })
    
    analyzer = DeviceTestAnalyzer(data)
    
    # test analysis process
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        analyzer.run_full_analysis(tmpdir)
        
        # check if file is generated
        expected_files = [
            'analysis_report.txt',
            'exploratory_plots.png',
            'feature_importance.csv',
            'model_summary.txt',
            'hypothesis_test.json'
        ]
        
        for file in expected_files:
            file_path = os.path.join(tmpdir, file)
            assert os.path.exists(file_path), f"File not found: {file_path}"
    
    print(" Integration test passed")


if __name__ == "__main__":
    test = TestDeviceTestAnalyzer()
    sample_data = test.sample_data()
    
    print("Running tests...")
    
    test.test_initialization(sample_data)
    print(" test_initialization passed")
    
    test.test_exploratory_analysis(sample_data)
    print(" test_exploratory_analysis passed")
    
    test.test_hypothesis_test(sample_data)
    print(" test_hypothesis_test passed")
    
    test.test_train_prediction_model(sample_data)
    print(" test_train_prediction_model passed")
    
    test_integration()
    
    print("\n All tests passed!")
