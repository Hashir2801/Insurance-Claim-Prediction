import pandas as pd
import numpy as np

def perform_preprocessing(data):
    # Before preprocessing
    original_data = data.copy()
    
    # Calculate medians for numerical features
    numerical_features = [col for col in data.columns 
                        if col not in ['id', 'target'] 
                        and '_cat' not in col 
                        and '_bin' not in col]
    
    medians = {}
    for col in numerical_features:
        medians[col] = data[col][data[col] != -1].median()
    
    # After preprocessing
    processed_data = data.copy()
    for col in numerical_features:
        processed_data[col] = processed_data[col].replace(-1, medians[col])
    
    # Get sample categorical data
    categorical_features = [col for col in data.columns if '_cat' in col]
    cat_sample = {}
    for col in categorical_features[:3]:  # First 3 categorical features
        cat_sample[col] = {
            'unique_values': int(data[col].nunique()),
            'sample_values': data[col].unique()[:3].tolist()
        }
    
    # Prepare results
    results = {
        'missing_values': {
            'before': int((original_data == -1).sum().sum()),
            'after': int((processed_data == -1).sum().sum())
        },
        'categorical_features': cat_sample,
        'numerical_features': {
            'count': len(numerical_features),
            'sample_medians': {k: float(v) for k, v in list(medians.items())[:3]}  # First 3
        },
        'sample_data': {
            'before': original_data.iloc[0].to_dict(),  # First row
            'after': processed_data.iloc[0].to_dict()   # First row
        }
    }
    
    return results