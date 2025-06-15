import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def create_visualizations(data):
    # Ensure directory exists
    os.makedirs('static/images/visualizations', exist_ok=True)
    
    # 1. Target Distribution
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='target', data=data)
    ax.set_title('Target Variable Distribution')
    plt.savefig('static/images/visualizations/target_distribution.png', 
               bbox_inches='tight')
    plt.close()
    
    # 2. Missing Values Heatmap (sampled)
    plt.figure(figsize=(12, 8))
    missing_data = data.replace(-1, np.nan).isnull()
    sampled_cols = missing_data.columns[::5]  # Sample columns
    ax = sns.heatmap(missing_data[sampled_cols], cbar=False)
    ax.set_title('Missing Values Heatmap (Sampled Columns)')
    plt.savefig('static/images/visualizations/missing_values_heatmap.png',
               bbox_inches='tight')
    plt.close()
    
    # 3. Feature Distributions
    numerical_features = [col for col in data.columns 
                         if col not in ['id', 'target'] 
                         and '_bin' not in col 
                         and '_cat' not in col]
    
    for feature in numerical_features[:5]:  # First 5 features
        plt.figure(figsize=(8, 6))
        ax = sns.histplot(data[feature], kde=True)
        ax.set_title(f'Distribution of {feature}')
        plt.savefig(f'static/images/visualizations/dist_{feature}.png',
                   bbox_inches='tight')
        plt.close()
    
    # 4. Correlation Matrix
    plt.figure(figsize=(12, 10))
    corr = data[numerical_features[:10]].corr()  # First 10 features
    ax = sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    ax.set_title('Feature Correlation Matrix')
    plt.savefig('static/images/visualizations/correlation_matrix.png',
               bbox_inches='tight')
    plt.close()
    
    # 5. Group-wise Distributions
    group_prefixes = ['ps_ind_', 'ps_reg_', 'ps_car_', 'ps_calc_']
    for prefix in group_prefixes:
        group_cols = [col for col in data.columns if col.startswith(prefix)]
        if group_cols:
            plt.figure(figsize=(12, 6))
            data[group_cols[:5]].boxplot()  # First 5 features
            plt.title(f'{prefix} Group Feature Distribution')
            plt.xticks(rotation=45)
            plt.savefig(f'static/images/visualizations/group_{prefix}.png',
                       bbox_inches='tight')
            plt.close()