"""
Capstone Project: Multi-Year Predictive Analytics of Employee Earnings in Boston (2022-2024)
Authors: Dev Panchal, Om Patel
Course: ALY 6140 - Python & Analytics
Northeastern University, Fall 2025

This module contains reusable functions for data loading, preprocessing, 
visualization, statistical analysis, and model evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_and_combine_datasets(url_dict, year_column='Year'):
    """
    Load multiple employee earnings datasets from URLs and combine them with year labels.
    Handles both CSV and Excel file formats automatically.
    """
    import io
    
    # Try importing requests
    try:
        import requests
        USE_REQUESTS = True
    except ImportError:
        import urllib.request
        USE_REQUESTS = False
    
    print("="*80)
    print("LOADING AND COMBINING DATASETS FROM URLS")
    print("="*80)
    
    dataframes = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for year, url in url_dict.items():
        try:
            print(f"\nLoading data for year {year}...")
            
            if USE_REQUESTS:
                session = requests.Session()
                response = session.get(url, headers=headers, timeout=30, allow_redirects=True)
                response.raise_for_status()
                content = response.content
            else:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as response:
                    content = response.read()
            
            if url.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(content), low_memory=False)
            elif url.endswith('.xlsx') or url.endswith('.xls'):
                df = pd.read_excel(io.BytesIO(content))
            else:
                try:
                    df = pd.read_csv(io.BytesIO(content), low_memory=False)
                except:
                    df = pd.read_excel(io.BytesIO(content))
            
            df[year_column] = year
            print(f"   Loaded {len(df):,} records with {len(df.columns)} columns")
            dataframes.append(df)
            
        except Exception as e:
            print(f"   Error loading {year}: {str(e)}")
            continue
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"\n{'='*80}")
        print(f"SUCCESSFULLY COMBINED {len(dataframes)} DATASETS")
        print(f"   Total Records: {len(combined_df):,}")
        print(f"   Year Range: {combined_df[year_column].min()} - {combined_df[year_column].max()}")
        print(f"{'='*80}\n")
        return combined_df
    else:
        raise ValueError("No datasets were successfully loaded from URLs")


def load_and_combine_local_datasets(file_dict, year_column='Year'):
    """
    Load multiple employee earnings datasets from local files.
    """
    print("="*80)
    print("LOADING AND COMBINING LOCAL DATASETS")
    print("="*80)
    
    dataframes = []
    
    for year, filepath in file_dict.items():
        try:
            print(f"\nLoading data for year {year}...")
            
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath, low_memory=False)
            elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                df = pd.read_excel(filepath)
            
            df[year_column] = year
            print(f"   Loaded {len(df):,} records")
            dataframes.append(df)
            
        except Exception as e:
            print(f"   Error: {str(e)}")
            continue
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"\n{'='*80}")
        print(f"SUCCESSFULLY COMBINED {len(dataframes)} DATASETS")
        print(f"{'='*80}\n")
        return combined_df
    else:
        raise ValueError("No datasets were loaded")


def preprocess_employee_data(df, config=None):
    """
    Clean and preprocess employee earnings data.
    """
    print("="*80)
    print("DATA PREPROCESSING AND CLEANING")
    print("="*80)
    
    default_config = {
        'fill_missing_with_zero': ['OVERTIME', 'RETRO', 'OTHER', 'DETAIL', 'QUINN_EDUCATION', 'INJURED'],
        'create_features': True,
        'handle_outliers': False,
        'rename_columns': True,
        'drop_duplicates': True
    }
    
    if config is None:
        config = default_config
    else:
        config = {**default_config, **config}
    
    stats = {
        'initial_rows': len(df),
        'initial_columns': len(df.columns),
        'missing_before': df.isnull().sum().sum(),
        'rows_removed': 0,
        'columns_created': 0
    }
    
    df_clean = df.copy()
    
    # Step 1: Rename columns
    if config['rename_columns']:
        print("\n1. Standardizing column names...")
        df_clean.columns = df_clean.columns.str.strip().str.upper()
        
        # Fix TOTAL GROSS column inconsistency
        if 'TOTAL GROSS' in df_clean.columns and 'TOTAL_ GROSS' in df_clean.columns:
            print("   Fixing TOTAL GROSS column inconsistency...")
            df_clean['TOTAL GROSS'] = df_clean['TOTAL GROSS'].fillna(df_clean['TOTAL_ GROSS'])
            df_clean = df_clean.drop(columns=['TOTAL_ GROSS'])
            print("   Merged TOTAL GROSS and TOTAL_ GROSS columns")
        
        print("   Columns standardized")
    
    # Step 2: Drop duplicates
    if config['drop_duplicates']:
        print("\n2. Removing duplicate rows...")
        before_dup = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        stats['rows_removed'] = before_dup - len(df_clean)
        print(f"   Removed {stats['rows_removed']:,} duplicates")
    
    # Step 3: Initial missing value check
    print("\n3. Handling missing values (initial pass)...")
    missing_before = df_clean.isnull().sum().sum()
    print(f"   Initial missing values: {missing_before:,}")
    
    # Step 4: Convert data types AND handle missing
    print("\n4. Converting data types and handling missing values...")
    numeric_columns = ['REGULAR', 'RETRO', 'OTHER', 'OVERTIME', 'INJURED', 
                      'DETAIL', 'QUINN_EDUCATION', 'TOTAL GROSS']
    
    converted_count = 0
    filled_count = 0
    
    for col in numeric_columns:
        if col in df_clean.columns:
            initial_missing = df_clean[col].isnull().sum()
            
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].str.replace(',', '', regex=False)
                df_clean[col] = df_clean[col].str.replace('$', '', regex=False)
                df_clean[col] = df_clean[col].str.strip()
                df_clean[col] = df_clean[col].replace(['nan', 'NaN', 'None', ''], np.nan)
            
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            converted_count += 1
            
            after_missing = df_clean[col].isnull().sum()
            if after_missing > 0:
                df_clean[col] = df_clean[col].fillna(0)
                filled_count += after_missing
                if after_missing != initial_missing:
                    print(f"   {col}: Filled {after_missing:,} missing values")
    
    if 'YEAR' in df_clean.columns:
        df_clean['YEAR'] = df_clean['YEAR'].astype(int)
    
    missing_after = df_clean.isnull().sum().sum()
    stats['missing_after'] = missing_after
    
    print(f"   Converted {converted_count} columns")
    print(f"   Filled {filled_count:,} missing values with 0")
    
    # Step 5: Create features
    if config['create_features']:
        print("\n5. Creating new features...")
        
        # Calculate TOTAL GROSS if missing
        if 'TOTAL GROSS' in df_clean.columns:
            missing_total = df_clean['TOTAL GROSS'].isnull().sum()
            if missing_total > 0:
                print(f"   Calculating TOTAL GROSS for {missing_total:,} records...")
                salary_components = ['REGULAR', 'OVERTIME', 'RETRO', 'OTHER', 'DETAIL', 'QUINN_EDUCATION', 'INJURED']
                available = [col for col in salary_components if col in df_clean.columns]
                mask = df_clean['TOTAL GROSS'].isnull()
                df_clean.loc[mask, 'TOTAL GROSS'] = df_clean.loc[mask, available].sum(axis=1)
        
        # Overtime percentage
        if 'OVERTIME' in df_clean.columns and 'TOTAL GROSS' in df_clean.columns:
            df_clean['OVERTIME_PCT'] = (df_clean['OVERTIME'] / df_clean['TOTAL GROSS'] * 100).fillna(0)
            df_clean['OVERTIME_PCT'] = df_clean['OVERTIME_PCT'].replace([np.inf, -np.inf], 0)
            stats['columns_created'] += 1
            print("   Created: OVERTIME_PCT")
        
        if 'OVERTIME' in df_clean.columns:
            df_clean['HAS_OVERTIME'] = (df_clean['OVERTIME'] > 0).astype(int)
            stats['columns_created'] += 1
            print("   Created: HAS_OVERTIME")
        
        bonus_cols = ['RETRO', 'QUINN_EDUCATION', 'OTHER']
        available_bonus = [col for col in bonus_cols if col in df_clean.columns]
        if available_bonus:
            df_clean['TOTAL_BONUSES'] = df_clean[available_bonus].sum(axis=1)
            stats['columns_created'] += 1
            print("   Created: TOTAL_BONUSES")
        
        if 'TOTAL GROSS' in df_clean.columns:
            df_clean['PAY_CATEGORY'] = pd.cut(df_clean['TOTAL GROSS'], 
                                              bins=[0, 50000, 100000, 150000, float('inf')],
                                              labels=['<50K', '50K-100K', '100K-150K', '>150K'])
            stats['columns_created'] += 1
            print("   Created: PAY_CATEGORY")
    
    stats['final_rows'] = len(df_clean)
    stats['final_columns'] = len(df_clean.columns)
    
    print(f"\n{'='*80}")
    print("PREPROCESSING COMPLETE")
    print(f"   Rows: {stats['initial_rows']:,} -> {stats['final_rows']:,}")
    print(f"   Columns: {stats['initial_columns']} -> {stats['final_columns']}")
    print(f"{'='*80}\n")
    
    return df_clean, stats


def evaluate_and_compare_models(models_dict, X_train, X_test, y_train, y_test, model_names=None):
    """
    Train, evaluate, and compare multiple regression models.
    """
    print("="*80)
    print("MODEL TRAINING AND EVALUATION")
    print("="*80)
    
    results = []
    trained_models = {}
    predictions = {}
    
    for name, model in models_dict.items():
        print(f"\nTraining {name}...")
        
        try:
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            results.append({
                'Model': name,
                'Train_R2': train_r2,
                'Test_R2': test_r2,
                'Train_RMSE': train_rmse,
                'Test_RMSE': test_rmse,
                'Train_MAE': train_mae,
                'Test_MAE': test_mae,
                'Overfit_Score': train_r2 - test_r2
            })
            
            trained_models[name] = model
            predictions[name] = {'train': y_train_pred, 'test': y_test_pred}
            
            print(f"   R2: {test_r2:.4f}, RMSE: ${test_rmse:,.2f}")
            
        except Exception as e:
            print(f"   Error: {str(e)}")
            continue
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("\nNo models trained successfully!")
        return pd.DataFrame(), {}, {}
    
    results_df = results_df.sort_values('Test_R2', ascending=False).reset_index(drop=True)
    
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    print(f"{'='*80}\n")
    
    return results_df, trained_models, predictions


def generate_data_quality_report(df):
    """
    Generate comprehensive data quality report.
    """
    print("="*80)
    print("DATA QUALITY REPORT")
    print("="*80)
    
    report = {}
    
    print("\nMissing Values:")
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Pct': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    missing_data = missing_data[missing_data['Missing_Count'] > 0]
    
    if len(missing_data) > 0:
        print(missing_data.to_string(index=False))
    else:
        print("   No missing values!")
    
    report['missing_values'] = missing_data
    
    print(f"\nDuplicates: {df.duplicated().sum():,}")
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"{'='*80}\n")
    
    return report


def create_comprehensive_eda_plots(df, target_col='TOTAL GROSS', figsize=(15, 10)):
    """
    Generate comprehensive EDA visualizations.
    """
    print("="*80)
    print("CREATING EDA VISUALIZATIONS")
    print("="*80)
    
    figures = {}
    
    # Distribution
    print("\n1. Distribution plots...")
    fig1, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].hist(df[target_col], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Total Gross Pay ($)', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('Distribution', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].boxplot(df[target_col], vert=True)
    axes[1].set_ylabel('Total Gross Pay ($)', fontweight='bold')
    axes[1].set_title('Box Plot', fontweight='bold')
    plt.tight_layout()
    figures['distribution'] = fig1
    
    # Trends
    if 'YEAR' in df.columns:
        print("2. Time series trends...")
        fig2, ax = plt.subplots(figsize=(12, 6))
        yearly_avg = df.groupby('YEAR')[target_col].mean()
        ax.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, markersize=10, color='darkgreen')
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Average Pay ($)', fontweight='bold')
        ax.set_title('Earnings Trend', fontweight='bold')
        ax.grid(True, alpha=0.3)
        for x, y in zip(yearly_avg.index, yearly_avg.values):
            ax.text(x, y, f'${y:,.0f}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        figures['trends'] = fig2
    
    # Correlation
    print("3. Correlation heatmap...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        fig3, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax)
        ax.set_title('Correlation Heatmap', fontweight='bold')
        plt.tight_layout()
        figures['correlation'] = fig3
    
    # Department
    if 'DEPARTMENT_NAME' in df.columns:
        print("4. Department comparison...")
        fig4, ax = plt.subplots(figsize=(14, 8))
        dept_avg = df.groupby('DEPARTMENT_NAME')[target_col].mean().sort_values(ascending=False).head(10)
        ax.barh(range(len(dept_avg)), dept_avg.values, color='teal', alpha=0.7)
        ax.set_yticks(range(len(dept_avg)))
        ax.set_yticklabels(dept_avg.index)
        ax.set_xlabel('Average Pay ($)', fontweight='bold')
        ax.set_title('Top 10 Departments', fontweight='bold')
        plt.tight_layout()
        figures['department'] = fig4
    
    # Overtime
    if 'OVERTIME' in df.columns:
        print("5. Overtime analysis...")
        fig5, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].scatter(df['OVERTIME'], df[target_col], alpha=0.3, s=10, color='coral')
        axes[0].set_xlabel('Overtime ($)', fontweight='bold')
        axes[0].set_ylabel('Total Pay ($)', fontweight='bold')
        axes[0].set_title('Overtime vs Total Pay', fontweight='bold')
        
        has_ot = df[df['OVERTIME'] > 0]
        axes[1].hist(has_ot['OVERTIME'], bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1].set_xlabel('Overtime ($)', fontweight='bold')
        axes[1].set_ylabel('Frequency', fontweight='bold')
        axes[1].set_title(f'Overtime Distribution ({len(has_ot):,} employees)', fontweight='bold')
        plt.tight_layout()
        figures['overtime'] = fig5
    
    print(f"\n{'='*80}")
    print(f"VISUALIZATIONS COMPLETE ({len(figures)} figures)")
    print(f"{'='*80}\n")
    
    return figures


def perform_statistical_tests(df, group_col='DEPARTMENT_NAME', value_col='TOTAL GROSS', top_n=10):
    """
    Perform statistical significance tests.
    """
    print("="*80)
    print("STATISTICAL TESTING")
    print("="*80)
    
    results = {}
    
    top_groups = df[group_col].value_counts().head(top_n).index
    df_filtered = df[df[group_col].isin(top_groups)].copy()
    
    print(f"\nAnalyzing top {top_n} groups")
    print(f"Total samples: {len(df_filtered):,}")
    
    print(f"\nDescriptive Statistics:")
    group_summary = df_filtered.groupby(group_col)[value_col].agg([
        ('Count', 'count'),
        ('Mean', 'mean'),
        ('Median', 'median'),
        ('StdDev', 'std')
    ]).round(2)
    print(group_summary.to_string())
    results['group_summary'] = group_summary
    
    print(f"\nANOVA Test:")
    groups = [df_filtered[df_filtered[group_col] == group][value_col].dropna() 
              for group in top_groups]
    
    try:
        f_stat, p_val = stats.f_oneway(*groups)
        results['anova'] = {'f_statistic': f_stat, 'p_value': p_val}
        print(f"   F-statistic: {f_stat:.4f}")
        print(f"   P-value: {p_val:.6f}")
        print(f"   Result: {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'}")
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    print("="*80)
    print("CAPSTONE PROJECT - EMPLOYEE EARNINGS ANALYSIS")
    print("Authors: Dev Panchal, Om Patel")
    print("="*80)
    print("\nThis module contains 7 functions")
    print("="*80)