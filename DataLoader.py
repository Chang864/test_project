"""Data loading and validation"""
import pandas as pd
from pathlib import Path


def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Load file and validate
    
    Args:
        filepath
        
    Returns:
        validated DataFrame
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"File doesn't exist: {filepath}")
    
    # load data
    df = pd.read_csv(filepath)
    
    # convert column names to lower case
    df.columns = [col.lower() for col in df.columns]
    
    # convert all strings to lower case
    for col in df.columns:
        if df[col].dtype == 'object':  # if string
            df[col] = df[col].astype(str).str.lower()
    
    # validaation
    validate_dataframe(df)
    
    print(f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns")
    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    """validate data"""
    if df.empty:
        raise ValueError("Dataframe is empty")
    
    required_columns = ['name', 'number_of_drop', 'test_status', 'angle_x', 'angle_y', 'angle_z'
                        , 'mark', 'dropheight']
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        raise ValueError(f"Columns missing: {missing}")
    
    # Check null values
    missing_values = df[required_columns].isnull().sum()
    if missing_values.any():
        print(f"Missing values found:\n{missing_values[missing_values > 0]}")
