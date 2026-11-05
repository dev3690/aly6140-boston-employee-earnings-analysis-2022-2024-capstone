"""
Capstone Project: Boston Employee Earnings Analysis (2022-2024)
Authors: Dev Panchal, Om Patel
Course: ALY 6140 - Python & Analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import requests
from io import BytesIO

def load_and_combine_data():
    """
    Load employee earnings data from 2022-2024 from public URLs and combine into single DataFrame.
    
    Returns:
        pd.DataFrame: Combined dataset with Year column added
    """
    urls = {
        2024: 'https://data.boston.gov/dataset/418983dc-7cae-42bb-88e4-d56f5adcf869/resource/579a4be3-9ca7-4183-bc95-7d67ee715b6d/download/employee_earnings_report_2024.csv',
        2023: 'https://data.boston.gov/dataset/418983dc-7cae-42bb-88e4-d56f5adcf869/resource/6b3c5333-1dcb-4b3d-9cd7-6a03fb526da7/download/employee-earnings-report-2023.xlsx',
        2022: 'https://data.boston.gov/dataset/418983dc-7cae-42bb-88e4-d56f5adcf869/resource/63ac638b-36c4-487d-9453-1d83eb5090d2/download/finalconsolidatedcy22earnings_feb2023.xlsx-sheet1.csv'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    dfs = []
    for year, url in urls.items():
        print(f"Loading data for {year}...")
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            if url.endswith('.xlsx'):
                df = pd.read_excel(BytesIO(response.content))
            else:
                df = pd.read_csv(BytesIO(response.content))
            
            df['YEAR'] = year
            dfs.append(df)
            print(f"  ✓ Loaded {len(df)} records")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            raise
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Total records: {len(combined_df)}")
    return combined_df

def clean_employee_data(df):
    """
    Clean and prepare employee earnings data for analysis.
    
    Args:
        df (pd.DataFrame): Raw combined dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset with new features
    """
    df_clean = df.copy()
    
    # Standardize column names: strip, uppercase, replace spaces with underscore
    df_clean.columns = df_clean.columns.str.strip().str.upper().str.replace(' ', '_')
    
    # Handle QUINN/EDUCATION column variations
    if 'QUINN/EDUCATION_INCENTIVE' in df_clean.columns:
        df_clean.rename(columns={'QUINN/EDUCATION_INCENTIVE': 'QUINN_EDUCATION'}, inplace=True)
    
    # Numeric columns to clean
    numeric_cols = ['REGULAR', 'RETRO', 'OTHER', 'OVERTIME', 'INJURED', 'DETAIL', 'QUINN_EDUCATION', 'TOTAL_GROSS']
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    # Create overtime percentage feature
    df_clean['OVERTIME_PCT'] = np.where(
        df_clean['REGULAR'] > 0,
        (df_clean['OVERTIME'] / df_clean['REGULAR'] * 100),
        0
    )
    
    # Remove duplicates and invalid records
    df_clean.drop_duplicates(inplace=True)
    df_clean = df_clean[df_clean['TOTAL_GROSS'] > 0]
    
    print(f"\n✓ Cleaned: {len(df_clean)} records")
    return df_clean

def plot_feature_importance(model, feature_names, top_n=6):
    """
    Plot feature importance from Random Forest model.
    
    Args:
        model: Trained RandomForestRegressor
        feature_names (list): List of feature names
        top_n (int): Number of top features to display
    """
    importances = model.feature_importances_
    
    # FIX: Limit top_n to actual number of features
    top_n = min(top_n, len(feature_names))
    
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
    plt.bar(range(top_n), importances[indices], color='steelblue', edgecolor='black')
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nFeature Importance Rankings:")
    print("-" * 40)
    for i, idx in enumerate(indices, 1):
        print(f"{i:2d}. {feature_names[idx]:20s}: {importances[idx]:.4f}")