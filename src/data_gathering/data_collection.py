import os
import requests
import pandas as pd
import json
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv
import logging

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# load env
load_dotenv()
POKE_TRACE_API_KEY = os.getenv('POKE_TRACE_API_KEY')


# og 151 referenced from wikipedia: https://en.wikipedia.org/wiki/List_of_generation_I_Pok%C3%A9mon
ORIGINAL_151_POKEMON = [
    'Bulbasaur', 'Ivysaur', 'Venusaur', 'Charmander', 'Charmeleon', 'Charizard',
    'Squirtle', 'Wartortle', 'Blastoise', 'Caterpie', 'Metapod', 'Butterfree',
    'Weedle', 'Kakuna', 'Beedrill', 'Pidgey', 'Pidgeotto', 'Pidgeot',
    'Rattata', 'Raticate', 'Spearow', 'Fearow', 'Ekans', 'Arbok',
    'Pikachu', 'Raichu', 'Sandshrew', 'Sandslash', 'Nidoran F', 'Nidorina',
    'Nidoqueen', 'Nidoran M', 'Nidorino', 'Nidoking', 'Clefairy', 'Clefable',
    'Vulpix', 'Ninetales', 'Jigglypuff', 'Wigglytuff', 'Zubat', 'Golbat',
    'Oddish', 'Gloom', 'Vileplume', 'Paras', 'Parasect', 'Venonat',
    'Venomoth', 'Diglett', 'Dugtrio', 'Meowth', 'Persian', 'Psyduck',
    'Golduck', 'Mankey', 'Primeape', 'Growlithe', 'Arcanine', 'Poliwag',
    'Poliwhirl', 'Poliwrath', 'Abra', 'Kadabra', 'Alakazam', 'Machop',
    'Machoke', 'Machamp', 'Bellsprout', 'Weepinbell', 'Victreebel', 'Tentacool',
    'Tentacruel', 'Geodude', 'Graveler', 'Golem', 'Ponyta', 'Rapidash',
    'Slowpoke', 'Slowbro', 'Magnemite', 'Magneton', "Farfetch'd", 'Doduo',
    'Dodrio', 'Seel', 'Dewgong', 'Grimer', 'Muk', 'Shellder',
    'Cloyster', 'Gastly', 'Haunter', 'Gengar', 'Onix', 'Drowzee',
    'Hypno', 'Krabby', 'Kingler', 'Voltorb', 'Electrode', 'Exeggcute',
    'Exeggutor', 'Cubone', 'Marowak', 'Hitmonlee', 'Hitmonchan', 'Lickitung',
    'Koffing', 'Weezing', 'Rhyhorn', 'Rhydon', 'Chansey', 'Tangela',
    'Kangaskhan', 'Horsea', 'Seadra', 'Goldeen', 'Seaking', 'Staryu',
    'Starmie', 'Mr. Mime', 'Scyther', 'Jynx', 'Electabuzz', 'Magmar',
    'Pinsir', 'Taurus', 'Magikarp', 'Gyarados', 'Lapras', 'Ditto',
    'Eevee', 'Vaporeon', 'Jolteon', 'Flareon', 'Porygon', 'Omanyte',
    'Omastar', 'Kabuto', 'Kabutops', 'Aerodactyl', 'Snorlax', 'Articuno',
    'Zapdos', 'Moltres', 'Dratini', 'Dragonair', 'Dragonite', 'Mewtwo',
    'Mew'
]


class PokeTraceAPIClient:
    BASE_URL = "https://api.poketrace.com"
    
    def __init__(self, api_key: str):
        """
        init api client
        """
        self.api_key = api_key
        self.headers = {
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        }
    
    def search_cards(self, 
                    name: Optional[str] = None,
                    market: str = 'US',
                    set_slug: Optional[str] = None,
                    limit: int = 10) -> Optional[List[Dict]]:
        """
        Search for cards in the PokeTrace API
        """
        params = {
            'market': market,
            'limit': limit
        }
        
        if name:
            params['search'] = name
        if set_slug:
            params['set'] = set_slug
        
        try:
            response = requests.get(
                f'{self.BASE_URL}/v1/cards',
                headers=self.headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data:
                return data['data']
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching cards for {name}: {e}")
            return None
    
    def get_card_by_id(self, card_id: str) -> Optional[Dict]:
        """
        Get full card data by ID -> poketrace id
        
        Returns:
            Card data dictionary or None on fail
        """
        try:
            response = requests.get(
                f'{self.BASE_URL}/cards/{card_id}',
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching card {card_id}: {e}")
            return None
    
    def get_price_history(self, card_id: str, tier: str = 'NEAR_MINT') -> Optional[Dict]:
        """
        Get price history for a card -> defaulting to tracking near_mint
        
        Returns:
            Price history data or None
        """
        try:
            # f string from api docs
            response = requests.get(
                f'{self.BASE_URL}/cards/{card_id}/prices/{tier}/history',
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching price history for {card_id}: {e}")
            return None


class PokemonCardDataCollector:
    """Collects and processes Pokemon card data for the 151 original Pokemon"""
    
    def __init__(self, api_key: str):
        """Initialize collector with API client"""
        self.client = PokeTraceAPIClient(api_key)
        self.cards_data = []
    
    def fetch_151_pokemon_cards(self, set_slug: Optional[str] = None, 
                               market: str = 'US',
                               delay: float = 0.5) -> pd.DataFrame:
        """
        get card data for all 151 original Pokemon
        
        Returns:
            pd.Dataframe with fetched data
        """        
        for idx, pokemon_name in enumerate(ORIGINAL_151_POKEMON):
            logger.info(f"[{idx+1}/{len(ORIGINAL_151_POKEMON)}] Searching for {pokemon_name}...")
            
            # search for the card
            results = self.client.search_cards(
                name=pokemon_name,
                market=market,
                set_slug=set_slug,
                limit=10
            )
            
            if results and len(results) > 0:
                # get all variants (all rarities)
                for card in results:
                    card_data = self._extract_card_data(card, pokemon_name)
                    self.cards_data.append(card_data)
                logger.info(f"Found: {len(results)} variants of {pokemon_name}")
            else:
                logger.warning(f"{pokemon_name} not found!")
            
            # Add delay to avoid rate limiting
            time.sleep(delay)
        
        df = pd.DataFrame(self.cards_data)
        logger.info(f"Successfully fetched {len(df)} cards")
        return df
    
    def _extract_card_data(self, card: Dict, pokemon_name: str) -> Dict:
        """
        Get relevant fields from card API response
        
        Returns:
            Dict w extracted data
        """
        # Extract market data for NEAR_MINT condition
        ebay_data = card.get('prices', {}).get('ebay', {}).get('NEAR_MINT', {})
        tcgplayer_data = card.get('prices', {}).get('tcgplayer', {}).get('NEAR_MINT', {})
        
        extracted = {
            'pokemon_name': pokemon_name,
            'card_id': card.get('id'),
            'card_number': card.get('cardNumber'),
            'set_name': card.get('set', {}).get('name'),
            'set_slug': card.get('set', {}).get('slug'),
            'variant': card.get('variant'),
            'rarity': card.get('rarity'),
            'currency': 'USD',
            'market': 'US',
            
            # eBay
            'ebay_avg_price': ebay_data.get('avg'),
            'ebay_low_price': ebay_data.get('low'),
            'ebay_high_price': ebay_data.get('high'),
            'ebay_sale_count': ebay_data.get('saleCount'),
            'ebay_avg_1d': ebay_data.get('avg1d'),
            'ebay_avg_7d': ebay_data.get('avg7d'),
            'ebay_avg_30d': ebay_data.get('avg30d'),
            'ebay_median_3d': ebay_data.get('median3d'),
            'ebay_median_7d': ebay_data.get('median7d'),
            'ebay_median_30d': ebay_data.get('median30d'),
            
            # TCGPlayer
            'tcgplayer_avg_price': tcgplayer_data.get('avg'),
            'tcgplayer_low_price': tcgplayer_data.get('low'),
            'tcgplayer_high_price': tcgplayer_data.get('high'),
            'tcgplayer_sale_count': tcgplayer_data.get('saleCount'),
            'tcgplayer_avg_1d': tcgplayer_data.get('avg1d'),
            'tcgplayer_avg_7d': tcgplayer_data.get('avg7d'),
            'tcgplayer_avg_30d': tcgplayer_data.get('avg30d'),
            'tcgplayer_median_3d': tcgplayer_data.get('median3d'),
            'tcgplayer_median_7d': tcgplayer_data.get('median7d'),
            'tcgplayer_median_30d': tcgplayer_data.get('median30d'),
            
            'last_updated': card.get('lastUpdated'),
            'image_url': card.get('image')
        }
        
        return extracted
    
    def save_to_csv(self, filepath: str) -> None:
        """
        Save to CSV
        """
        df = pd.DataFrame(self.cards_data)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} to {filepath}")


def main():
    """Main function to collect data"""
    # init collector
    collector = PokemonCardDataCollector(POKE_TRACE_API_KEY)
    
    # get 151 og Pokemon
    df = collector.fetch_151_pokemon_cards(set_slug='sv-scarlet-and-violet-151', market='US', delay=0.5)
    
    # Save to CSV
    output_path = '../data/raw/pokemon_sv_151_cards.csv'
    collector.save_to_csv(output_path)
    
    print(f"Total cards collected: {len(df)}")
    print(f"\nPeek head:")
    print(df[['pokemon_name', 'card_id', 'rarity', 'ebay_avg_price', 'tcgplayer_avg_price']].head())


if __name__ == "__main__":
    main()
