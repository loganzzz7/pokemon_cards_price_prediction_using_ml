import os
import requests
import pandas as pd
import time
import logging
from typing import Optional, Dict
from dotenv import load_dotenv
from historical_price_features import process_price_history

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
POKE_TRACE_API_KEY = os.getenv('POKE_TRACE_API_KEY')


class HistoricalDataEnhancer:
    """Fetches price history and extracts features for each card"""
    
    BASE_URL = "https://api.poketrace.com"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        }
    
    def get_price_history(self, card_id: str, period: str = '1y', tier: str = 'NEAR_MINT') -> Optional[Dict]:
        """Fetch price history from API"""
        try:
            response = requests.get(
                f'{self.BASE_URL}/v1/cards/{card_id}/prices/{tier}/history',
                headers=self.headers,
                params={'period': period},
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching history for {card_id}: {e}")
            return None
    
    def extract_features_from_history(self, history_data: Optional[Dict]) -> Dict:
        """Extract features from price history response"""
        if not history_data or 'data' not in history_data:
            return {
                'price_7d_avg': None,
                'price_30d_avg': None,
                'price_90d_avg': None,
                'price_1y_avg': None,
                'price_momentum': None,
                'price_volatility': None
            }
        
        return process_price_history(history_data['data'])
    
    def enhance_dataset(self, input_csv: str, output_csv: str, delay: float = 2.0) -> pd.DataFrame:
        """Fetch history for all cards and add features to dataset"""
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df)} cards from {input_csv}")
        
        # Add feature cols
        df['price_7d_avg'] = None
        df['price_30d_avg'] = None
        df['price_90d_avg'] = None
        df['price_1y_avg'] = None
        df['price_momentum'] = None
        df['price_volatility'] = None
        
        # Fetch history for each card
        for idx, row in df.iterrows():
            card_id = row['card_id']
            logger.info(f"[{idx+1}/{len(df)}] Fetching history for {card_id}...")
            
            history_data = self.get_price_history(card_id, period='1y')
            features = self.extract_features_from_history(history_data)
            
            for feature_name, feature_value in features.items():
                df.at[idx, feature_name] = feature_value
            
            time.sleep(delay)
        
        df.to_csv(output_csv, index=False)
        logger.info(f"Enhanced dataset saved to {output_csv}")
        
        return df


def main():
    """Enhance cleaned dataset with historical price features (standalone usage)"""
    enhancer = HistoricalDataEnhancer(POKE_TRACE_API_KEY)
    
    input_path = '../../data/processed/pokemon_sv_151_cleaned.csv'
    output_path = '../../data/processed/pokemon_sv_151_cleaned.csv'
    
    logger.info("Starting historical data enhancement...")
    enhancer.enhance_dataset(input_path, output_path, delay=2.0)


if __name__ == '__main__':
    main()
