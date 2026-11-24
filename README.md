# test_project
    This is a data analytics project, including data loading, data preprocessing, data analysis and modeling. 

# Analyzer
    A tool used to process and analyze device drop test data, with statistical analysis and machine learning capabilities.

# Setup
```bash
pip install pandas numpy matplotlib seaborn scipy click scikit-learn
```

# Usage
```bash
python CLI.py Data.csv --full
```

# Project Structure
    CLI.py              # Command-line interface
    Analyzer.py         # Analysis engine (stats + ML)
    DataLoader.py       # Data loading and preprocessing
    Test.py             # Unit tests
    Data.csv            # Sample dataset


# Analysis Included:

    Exploratory analysis
    Hypyothesis testing on difference in failure rates between 2 mark groups (up and down)
    Model reports and feature importance
    Generate related files


## Data Format

    csv file
    Columns: name, number_of_drop, test_status, angle_x, angle_y, angle_z, mark, dropheight
