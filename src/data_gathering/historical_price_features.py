import pandas as pd
import numpy as np
from datetime import timedelta


def process_price_history(history_data: list) -> dict:
    """Extract 7d/30d/90d/1y averages, momentum, volatility from price history"""
    if not history_data:
        return {
            'price_7d_avg': np.nan,
            'price_30d_avg': np.nan,
            'price_90d_avg': np.nan,
            'price_1y_avg': np.nan,
            'price_momentum': np.nan,
            'price_volatility': np.nan
        }
    
    df = pd.DataFrame(history_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', ascending=False)
    
    current_price = df.iloc[0]['avg'] if len(df) > 0 else np.nan
    today = df['date'].max()
    
    # Calculate period averages
    price_7d = df[df['date'] >= today - timedelta(days=7)]['avg'].mean()
    price_30d = df[df['date'] >= today - timedelta(days=30)]['avg'].mean()
    price_90d = df[df['date'] >= today - timedelta(days=90)]['avg'].mean()
    price_1y = df[df['date'] >= today - timedelta(days=365)]['avg'].mean()
    
    # % change from 30d avg
    momentum = ((current_price - price_30d) / price_30d * 100) if price_30d > 0 else np.nan
    
    # std dev of 30d prices
    volatility = df[df['date'] >= today - timedelta(days=30)]['avg'].std()
    
    return {
        'price_7d_avg': price_7d,
        'price_30d_avg': price_30d,
        'price_90d_avg': price_90d,
        'price_1y_avg': price_1y,
        'price_momentum': momentum,
        'price_volatility': volatility
    }
