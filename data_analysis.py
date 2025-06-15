import pandas as pd
import numpy as np

def perform_data_analysis(data):
    results = {
        'basic_info': {
            'rows': data.shape[0],
            'columns': data.shape[1],
            'missing_values': data.replace(-1, np.nan).isna().sum().sum(),
            'target_distribution': data['target'].value_counts().to_dict()
        },
        'data_types': {
            'binary': [col for col in data.columns if '_bin' in col],
            'categorical': [col for col in data.columns if '_cat' in col],
            'numerical': [col for col in data.columns if col not in ['id', 'target'] 
                         and '_bin' not in col and '_cat' not in col]
        },
        'feature_groups': {
            'individual': [col for col in data.columns if col.startswith('ps_ind_')],
            'regional': [col for col in data.columns if col.startswith('ps_reg_')],
            'car': [col for col in data.columns if col.startswith('ps_car_')],
            'calculated': [col for col in data.columns if col.startswith('ps_calc_')]
        },
        'sample_data': data.head().to_dict('records'),
        'descriptive_stats': data.describe().to_dict()
    }
    return results