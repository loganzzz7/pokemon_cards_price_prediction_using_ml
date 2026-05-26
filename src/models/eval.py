import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from baseline import LinearRegressionModel, MovingAverageModel
from advanced import RandomForestModel, XGBoostModel
from train_pipeline import load_data, prepare_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)

def load_results():
    """Load predictions and metrics"""
    predictions = pd.read_csv('../../models/predictions.csv')
    metrics = pd.read_csv('../../models/metrics.csv')
    return predictions, metrics

def plot_actual_vs_pred():
    """Plot actual vs predicted for all models"""
    predictions, _ = load_results()
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    models = ['LinearRegression_pred', 'MovingAverage_pred', 'RandomForest_pred', 'XGBoost_pred']
    model_names = ['Linear Regression', 'Moving Average', 'Random Forest', 'XGBoost']
    
    for ax, pred_col, name in zip(axes, models, model_names):
        ax.scatter(predictions['y_actual'], predictions[pred_col], alpha=0.6, s=50)
        
        # Add perfect prediction line
        min_val = min(predictions['y_actual'].min(), predictions[pred_col].min())
        max_val = max(predictions['y_actual'].max(), predictions[pred_col].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Price ($)', fontsize=11)
        ax.set_ylabel('Predicted Price ($)', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../models/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison():
    """Plot metrics comparison across models"""
    _, metrics = load_results()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # rmse
    axes[0].bar(metrics['Model'], metrics['RMSE'], color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('RMSE ($)', fontsize=11)
    axes[0].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(metrics['RMSE']):
        axes[0].text(i, v + max(metrics['RMSE'])*0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # mae
    axes[1].bar(metrics['Model'], metrics['MAE'], color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('MAE ($)', fontsize=11)
    axes[1].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(metrics['MAE']):
        axes[1].text(i, v + max(metrics['MAE'])*0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # r^2
    axes[2].bar(metrics['Model'], metrics['R2'], color='seagreen', alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('R² Score', fontsize=11)
    axes[2].set_title('Coefficient of Determination', fontsize=12, fontweight='bold')
    axes[2].set_ylim([metrics['R2'].min() - 0.1, 1.05])
    axes[2].tick_params(axis='x', rotation=45)
    for i, v in enumerate(metrics['R2']):
        axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../../models/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance():
    """Plot feature importance from Random Forest and XGBoost"""
    # Retrain models to get feature importance
    enhanced_path = '../../data/processed/pokemon_sv_151_enhanced.csv'
    current_path = '../../data/processed/pokemon_sv_151_cleaned.csv'
    
    if os.path.exists(enhanced_path):
        df = load_data(enhanced_path)
    else:
        df = load_data(current_path)
    
    X, y = prepare_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fill NaNs
    fill_values = X_train.median()
    X_train = X_train.fillna(fill_values)
    X_test = X_test.fillna(fill_values)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    rf_model = RandomForestModel()
    rf_model.fit(X_train_scaled, y_train)
    
    xgb_model = XGBoostModel()
    xgb_model.fit(X_train_scaled, y_train)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Random Forest
    rf_importance = rf_model.model.feature_importances_
    indices_rf = np.argsort(rf_importance)[-15:]  # Top 15
    axes[0].barh(range(len(indices_rf)), rf_importance[indices_rf], color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_yticks(range(len(indices_rf)))
    axes[0].set_yticklabels([X.columns[i] for i in indices_rf], fontsize=10)
    axes[0].set_xlabel('Importance Score', fontsize=11)
    axes[0].set_title('Random Forest - Top 15 Features', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # XGBoost
    xgb_importance = xgb_model.model.feature_importances_
    indices_xgb = np.argsort(xgb_importance)[-15:]  # Top 15
    axes[1].barh(range(len(indices_xgb)), xgb_importance[indices_xgb], color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_yticks(range(len(indices_xgb)))
    axes[1].set_yticklabels([X.columns[i] for i in indices_xgb], fontsize=10)
    axes[1].set_xlabel('Importance Score', fontsize=11)
    axes[1].set_title('XGBoost - Top 15 Features', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('../../models/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run all evaluations"""
    plot_actual_vs_pred()
    plot_metrics_comparison()
    plot_feature_importance()

if __name__ == '__main__':
    main()
