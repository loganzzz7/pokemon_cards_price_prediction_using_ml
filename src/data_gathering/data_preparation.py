import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Finalize columns for model training"""
    
    def __init__(self):
        self.data = None
    
    def finalize(self, merged_csv_path: str, output_csv_path: str) -> pd.DataFrame:
        """Load merged data and select final columns"""
        self.data = pd.read_csv(merged_csv_path)
        logger.info(f"Loaded {len(self.data)} records")
        
        # Create market features
        self.data['market_avg_price'] = self.data[['ebay_avg_price', 'tcgplayer_avg_price']].mean(axis=1)
        self.data['market_avg_7d_price'] = self.data[['ebay_avg_7d', 'tcgplayer_avg_7d']].mean(axis=1)
        self.data['market_avg_30d_price'] = self.data[['ebay_avg_30d', 'tcgplayer_avg_30d']].mean(axis=1)
        self.data['total_sales'] = self.data['ebay_sale_count'] + self.data['tcgplayer_sale_count']
        self.data['ebay_premium'] = self.data['ebay_avg_price'] - self.data['tcgplayer_avg_price']
        
        # Select columns as per data_collection.md
        cols = [
            'pokemon_name', 'card_id', 'card_number', 'set_name', 'rarity', 'variant',
            'popularity_rank', 'popularity_score',
            'market_avg_price', 'ebay_avg_price', 'tcgplayer_avg_price',
            'market_avg_7d_price', 'market_avg_30d_price',
            'ebay_avg_7d', 'ebay_avg_30d', 'tcgplayer_avg_7d', 'tcgplayer_avg_30d',
            'total_sales', 'ebay_sale_count', 'tcgplayer_sale_count',
            'ebay_premium',
            'ebay_low_price', 'ebay_median_3d', 'ebay_median_7d', 'ebay_median_30d', 'ebay_high_price',
            'tcgplayer_low_price', 'tcgplayer_median_3d', 'tcgplayer_median_7d', 'tcgplayer_median_30d', 'tcgplayer_high_price',
            'price_7d_avg', 'price_30d_avg', 'price_90d_avg', 'price_1y_avg', 'price_momentum', 'price_volatility'
        ]
        
        # Keep only available columns
        available_cols = [c for c in cols if c in self.data.columns]
        self.data = self.data[available_cols]
        
        self.data.to_csv(output_csv_path, index=False)
        logger.info(f"Saved {len(self.data)} records to {output_csv_path}")
        logger.info(f"Columns: {len(self.data.columns)}")
        
        return self.data