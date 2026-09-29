import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_corruption_comparison(df_cells: pd.DataFrame, save_path: str):
    """
    Plots the average error for each corruption family.
    """
    family_errors = df_cells.groupby('family')['error_fraction'].mean().reset_index()
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=family_errors, x='family', y='error_fraction', palette='viridis')
    plt.title('Average Error by Corruption Family')
    plt.ylabel('Error Rate')
    plt.xlabel('Family')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_severity_degradation(df_cells: pd.DataFrame, save_path: str):
    """
    Plots the error rate increasing across severities.
    """
    severity_errors = df_cells.groupby('severity')['error_fraction'].mean().reset_index()
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=severity_errors, x='severity', y='error_fraction', marker='o', linewidth=2)
    plt.title('Error Degradation by Severity Level')
    plt.ylabel('Average Error Rate')
    plt.xlabel('Severity Level (1-5)')
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
