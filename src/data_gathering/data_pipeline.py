import os
import sys
import logging
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

from data_collection import PokemonCardDataCollector
from popularity_rankings import PopularityRanker
from data_preparation import DataCleaner
from enhance_with_history import HistoricalDataEnhancer

# setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# load .env
load_dotenv()
API_KEY = os.getenv('POKE_TRACE_API_KEY')


def main():
    """Run full pipeline: raw -> popularity -> cleaned -> enhanced with historical features"""
    
    # Collect raw data
    collector = PokemonCardDataCollector(API_KEY)
    df_raw = collector.fetch_151_pokemon_cards(set_slug='sv-scarlet-and-violet-151', market='US', delay=2.0)
    collector.save_to_csv('../../data/raw/pokemon_sv_151_cards_raw.csv')
    
    # Add popularity rankings
    ranker = PopularityRanker()
    df_with_popularity = ranker.merge_with_cards(df_raw)
    df_with_popularity.to_csv('../../data/raw/pokemon_sv_151_with_popularity.csv', index=False)
    
    # Clean Data
    cleaner = DataCleaner()
    df_cleaned = cleaner.finalize('../../data/raw/pokemon_sv_151_with_popularity.csv', 
                                '../../data/processed/pokemon_sv_151_cleaned.csv')
    
    # Enhance with historical price features
    logger.info("Enhancing dataset with historical price features...")
    enhancer = HistoricalDataEnhancer(API_KEY)
    enhancer.enhance_dataset('../../data/processed/pokemon_sv_151_cleaned.csv',
                            '../../data/processed/pokemon_sv_151_cleaned.csv',
                            delay=2.0)
    
    # Summary
    df_final = pd.read_csv('../../data/processed/pokemon_sv_151_cleaned.csv')
    print(f"Dataset: {df_final.shape}")
    print(f"Columns: {len(df_final.columns)}")
    print(f"\nHead:")
    print(df_final[['pokemon_name', 'rarity', 'popularity_rank', 'market_avg_price']].head())


if __name__ == "__main__":
    main()