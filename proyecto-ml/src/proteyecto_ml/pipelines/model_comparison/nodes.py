"""
Model Comparison Pipeline - Nodes for comparing classification and regression models.

This pipeline consolidates results from both classification and regression pipelines,
generates comprehensive comparison tables with mean±std metrics, and creates 
visualizations as required by the evaluation criteria.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import json
import os
import logging


def consolidate_classification_metrics(
    logistic_metrics: Dict,
    knn_metrics: Dict, 
    svm_metrics: Dict,
    dt_clf_metrics: Dict,
    rf_clf_metrics: Dict
) -> Dict[str, Any]:
    """Consolidate all classification model metrics into a comparison table."""
    
    models_data = {
        'Logistic Regression': logistic_metrics,
        'K-Nearest Neighbors': knn_metrics,
        'Support Vector Machine': svm_metrics,
        'Decision Tree': dt_clf_metrics,
        'Random Forest': rf_clf_metrics
    }
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, metrics in models_data.items():
        # Defensive access for optional keys
        best_score = metrics.get('best_score')
        if best_score is None:
            logging.getLogger(__name__).warning("'best_score' missing for model %s; using 'N/A'", model_name)
            best_score_str = 'N/A'
        else:
            try:
                best_score_str = f"{best_score:.4f}"
            except Exception:
                best_score_str = str(best_score)

        best_params = metrics.get('best_params', {})

        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'CV Mean': f"{metrics['cv_mean']:.4f}",
            'CV Std': f"{metrics['cv_std']:.4f}",
            'CV Mean±Std': f"{metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}",
            'Best CV Score': best_score_str,
            'Best Parameters': str(best_params)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Find best models
    best_accuracy = comparison_df.loc[comparison_df['Accuracy'].astype(float).idxmax(), 'Model']
    best_cv = comparison_df.loc[comparison_df['CV Mean'].astype(float).idxmax(), 'Model']
    best_f1 = comparison_df.loc[comparison_df['F1-Score'].astype(float).idxmax(), 'Model']
    
    result = {
        'comparison_table': comparison_data,
        'best_models': {
            'best_accuracy': best_accuracy,
            'best_cv_score': best_cv,
            'best_f1_score': best_f1
        },
        'summary_stats': {
            'mean_accuracy': float(comparison_df['Accuracy'].astype(float).mean()),
            'std_accuracy': float(comparison_df['Accuracy'].astype(float).std()),
            'mean_cv_score': float(comparison_df['CV Mean'].astype(float).mean()),
            'std_cv_score': float(comparison_df['CV Mean'].astype(float).std())
        },
        'raw_metrics': models_data
    }
    
    return result


def consolidate_regression_metrics(
    linear_metrics: Dict,
    multiple_linear_metrics: Dict,
    dt_metrics: Dict,
    rf_metrics: Dict, 
    xgb_metrics: Dict,
) -> Dict[str, Any]:
    """Consolidate all regression model metrics into a comparison table."""
    
    models_data = {
        'Linear Regression': linear_metrics,
        'Multiple Linear Regression': multiple_linear_metrics,
        'Decision Tree': dt_metrics,
        'Random Forest': rf_metrics,
        'XGBoost': xgb_metrics,
    }
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, metrics in models_data.items():
        # Defensive access for optional keys
        best_score = metrics.get('best_score')
        if best_score is None:
            logging.getLogger(__name__).warning("'best_score' missing for regression model %s; using 'N/A'", model_name)
            best_score_str = 'N/A'
        else:
            try:
                best_score_str = f"{best_score:.4f}"
            except Exception:
                best_score_str = str(best_score)

        best_params = metrics.get('best_params', {})

        comparison_data.append({
            'Model': model_name,
            'R²': f"{metrics['r2']:.4f}",
            'RMSE': f"{metrics['rmse']:.4f}",
            'MAE': f"{metrics['mae']:.4f}",
            'MSE': f"{metrics['mse']:.4f}",
            'CV Mean': f"{metrics['cv_mean']:.4f}",
            'CV Std': f"{metrics['cv_std']:.4f}",
            'CV Mean±Std': f"{metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}",
            'Best CV Score': best_score_str,
            'Best Parameters': str(best_params)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Find best models (higher R² is better, lower RMSE is better)
    best_r2 = comparison_df.loc[comparison_df['R²'].astype(float).idxmax(), 'Model']
    best_cv = comparison_df.loc[comparison_df['CV Mean'].astype(float).idxmax(), 'Model'] 
    best_rmse = comparison_df.loc[comparison_df['RMSE'].astype(float).idxmin(), 'Model']
    
    result = {
        'comparison_table': comparison_data,
        'best_models': {
            'best_r2': best_r2,
            'best_cv_score': best_cv,
            'best_rmse': best_rmse
        },
        'summary_stats': {
            'mean_r2': float(comparison_df['R²'].astype(float).mean()),
            'std_r2': float(comparison_df['R²'].astype(float).std()),
            'mean_cv_score': float(comparison_df['CV Mean'].astype(float).mean()),
            'std_cv_score': float(comparison_df['CV Mean'].astype(float).std())
        },
        'raw_metrics': models_data
    }
    
    return result


def create_classification_visualization(classification_results: Dict[str, Any]) -> plt.Figure:
    """Create comprehensive visualization for classification model comparison."""
    
    # Extract data for visualization
    df = pd.DataFrame(classification_results['comparison_table'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Classification Models Comparison', fontsize=16, fontweight='bold')
    
    # 1. Accuracy comparison
    ax1 = axes[0, 0]
    accuracy_values = df['Accuracy'].astype(float)
    bars1 = ax1.bar(df['Model'], accuracy_values, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_title('Model Accuracy Comparison', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Cross-validation scores with error bars
    ax2 = axes[0, 1]
    cv_means = df['CV Mean'].astype(float)
    cv_stds = df['CV Std'].astype(float)
    bars2 = ax2.bar(df['Model'], cv_means, yerr=cv_stds, capsize=5, 
                    color='lightcoral', alpha=0.7, edgecolor='black')
    ax2.set_title('Cross-Validation Scores (Mean ± Std)', fontweight='bold')
    ax2.set_ylabel('CV Score')
    ax2.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Multiple metrics comparison (radar-like bar plot)
    ax3 = axes[1, 0]
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(df))
    width = 0.15
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for i, metric in enumerate(metrics_to_plot):
        values = df[metric].astype(float)
        ax3.bar(x + i*width, values, width, label=metric, 
                color=colors[i], alpha=0.8, edgecolor='black')
    
    ax3.set_title('All Metrics Comparison', fontweight='bold')
    ax3.set_ylabel('Score')
    ax3.set_xlabel('Models')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Best model summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    best_models = classification_results['best_models']
    summary_text = f"""
    🏆 BEST MODELS SUMMARY
    
    🎯 Best Accuracy: {best_models['best_accuracy']}
    📊 Best CV Score: {best_models['best_cv_score']}
    ⚖️ Best F1-Score: {best_models['best_f1_score']}
    
    📈 OVERALL STATISTICS
    Average Accuracy: {classification_results['summary_stats']['mean_accuracy']:.4f}
    Std Accuracy: {classification_results['summary_stats']['std_accuracy']:.4f}
    
    Average CV Score: {classification_results['summary_stats']['mean_cv_score']:.4f}
    Std CV Score: {classification_results['summary_stats']['std_cv_score']:.4f}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
             facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    return fig


def create_regression_visualization(regression_results: Dict[str, Any]) -> plt.Figure:
    """Create comprehensive visualization for regression model comparison."""
    
    # Extract data for visualization
    df = pd.DataFrame(regression_results['comparison_table'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Regression Models Comparison', fontsize=16, fontweight='bold')
    
    # 1. R² comparison
    ax1 = axes[0, 0]
    r2_values = df['R²'].astype(float)
    bars1 = ax1.bar(df['Model'], r2_values, color='lightgreen', alpha=0.7, edgecolor='black')
    ax1.set_title('Model R² Comparison', fontweight='bold')
    ax1.set_ylabel('R² Score')
    ax1.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. RMSE comparison (lower is better)
    ax2 = axes[0, 1]
    rmse_values = df['RMSE'].astype(float)
    bars2 = ax2.bar(df['Model'], rmse_values, color='salmon', alpha=0.7, edgecolor='black')
    ax2.set_title('Model RMSE Comparison (Lower is Better)', fontweight='bold')
    ax2.set_ylabel('RMSE')
    ax2.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cross-validation scores with error bars
    ax3 = axes[1, 0]
    cv_means = df['CV Mean'].astype(float)
    cv_stds = df['CV Std'].astype(float)
    bars3 = ax3.bar(df['Model'], cv_means, yerr=cv_stds, capsize=5,
                    color='gold', alpha=0.7, edgecolor='black')
    ax3.set_title('Cross-Validation R² Scores (Mean ± Std)', fontweight='bold')
    ax3.set_ylabel('CV R² Score')
    ax3.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Best model summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    best_models = regression_results['best_models']
    summary_text = f"""
    🏆 BEST MODELS SUMMARY
    
    📈 Best R²: {best_models['best_r2']}
    📊 Best CV Score: {best_models['best_cv_score']}
    📉 Best RMSE: {best_models['best_rmse']}
    
    📈 OVERALL STATISTICS
    Average R²: {regression_results['summary_stats']['mean_r2']:.4f}
    Std R²: {regression_results['summary_stats']['std_r2']:.4f}
    
    Average CV Score: {regression_results['summary_stats']['mean_cv_score']:.4f}
    Std CV Score: {regression_results['summary_stats']['std_cv_score']:.4f}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
             facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    return fig


def generate_final_report(classification_results: Dict, regression_results: Dict) -> Dict[str, Any]:
    """Generate final consolidated report with all model results."""
    
    final_report = {
        'project_summary': {
            'total_models_trained': 10,
            'classification_models': 5,
            'regression_models': 5,
            'cross_validation_folds': 5,
            'hyperparameter_tuning': 'GridSearchCV'
        },
        'classification_results': classification_results,
        'regression_results': regression_results,
        'best_overall_models': {
            'classification': {
                'model': classification_results['best_models']['best_cv_score'],
                'score': max([float(row['CV Mean']) for row in classification_results['comparison_table']]),
                'metric': 'Cross-Validation Accuracy'
            },
            'regression': {
                'model': regression_results['best_models']['best_cv_score'], 
                'score': max([float(row['CV Mean']) for row in regression_results['comparison_table']]),
                'metric': 'Cross-Validation R²'
            }
        },
        'methodology': {
            'data_preparation': 'Complete preprocessing pipeline with outlier removal and feature engineering',
            'model_validation': '5-fold stratified cross-validation for classification, 5-fold cross-validation for regression',
            'hyperparameter_optimization': 'GridSearchCV with comprehensive parameter grids',
            'evaluation_metrics': {
                'classification': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'regression': ['R²', 'RMSE', 'MAE', 'MSE']
            }
        },
        'recommendations': {
            'classification': f"Use {classification_results['best_models']['best_cv_score']} for best overall performance",
            'regression': f"Use {regression_results['best_models']['best_cv_score']} for best overall performance",
            'deployment': 'Models are saved with joblib and include all preprocessing components'
        }
    }
    
    return final_report
