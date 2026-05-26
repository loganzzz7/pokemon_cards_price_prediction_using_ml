import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# use my defined modules
from baseline import LinearRegressionModel, MovingAverageModel
from advanced import RandomForestModel, XGBoostModel

# init loggings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(filepath):
    """Ld cleaned data"""
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} records from {filepath}")
    return df


def prepare_features(df):
    """Prep features n target - use historical price trends + card traits"""
    df_copy = df.copy()
    
    # Rarity to ordinal (encoding for model so no str)
    rarity_map = {
        'Common': 1,
        'Uncommon': 2,
        'Rare': 3,
        'Double Rare': 4,
        'Ultra Rare': 5,
        'Illustration Rare': 6,
        'Special Illustration Rare': 7,
        'Hyper Rare': 8
    }
    df_copy['rarity'] = df_copy['rarity'].map(rarity_map)
    
    # thought about this for a while, then finalized on using non historical avgs to predict
    # Before had the avgs but realized that in a more stable market--as opposed to the stock market--
    # the "recent" pricing avgs played too much of a role in affecting the model's predictions
    # Removed: price_7d_avg, price_30d_avg, price_90d_avg, price_1y_avg (too close to current price)
    # So I only kept: momentum & volatility from historical data
    # and used other card characteristic data for training the models.
    keep_cols = [
        'rarity',
        'popularity_rank',
        'popularity_score',
        'total_sales',
        'price_momentum',
        'price_volatility'
    ]
    
    # Keep only available columns (in case enhanced data not ready yet)
    available_cols = [col for col in keep_cols if col in df_copy.columns]
    X = df_copy[available_cols].copy()
    y = df_copy['market_avg_price']
    
    # Drop rows where target is NaN
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    logger.info(f"Features: {X.shape[1]}, Target: {len(y)}")
    logger.info(f"Features used: {list(X.columns)}")
    return X, y


def train_models(X_train, X_test, y_train, y_test):
    """Train all models n ret predictions"""
    models = {
        'LinearRegression': LinearRegressionModel(),
        'MovingAverage': MovingAverageModel(),
        'RandomForest': RandomForestModel(),
        'XGBoost': XGBoostModel()
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"{name}--RMSE: {rmse:.3f}, MAE: {mae:.3f}, r^2: {r2:.3f}")
    
    return results


def save_results(results, y_test):
    """Save predictions and metrics to CSV"""
    df_results = pd.DataFrame({
        'y_actual': y_test.values
    })
    
    for name, data in results.items():
        df_results[f'{name}_pred'] = data['y_pred']
    
    df_results.to_csv('../../models/predictions.csv', index=False)
    
    # Metrics sum
    metrics = {
        'Model': list(results.keys()),
        'RMSE': [results[m]['rmse'] for m in results.keys()],
        'MAE': [results[m]['mae'] for m in results.keys()],
        'R2': [results[m]['r2'] for m in results.keys()]
    }
    
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv('../../models/metrics.csv', index=False)
    
    print("\n")
    print("MODEL COMPARISON")
    print(df_metrics.to_string(index=False))


def main():
    """Run full pipeline"""
    data_path = '../../data/processed/pokemon_sv_151_cleaned.csv'

    df = load_data(data_path)
    
    X, y = prepare_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Train-Test Split--Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Fill NaNs with training set median (prevent data leakage)
    fill_values = X_train.median()
    X_train = X_train.fillna(fill_values)
    X_test = X_test.fillna(fill_values)
    
    # Scale features for linear regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    save_results(results, y_test)


if __name__ == '__main__':
    main()
