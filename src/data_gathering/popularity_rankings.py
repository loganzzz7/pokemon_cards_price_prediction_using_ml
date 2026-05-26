import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Popularity rankings based on Ranker.com
# https://www.ranker.com/list/best-generation-1-pokemon/ranker-pokemon
# These are based on community voting (top-to-bottom = highest-to-lowest rank)
POKEMON_POPULARITY_RANKINGS = {
    'Gengar': 1,
    'Gyarados': 2,
    'Arcanine': 3,
    'Mewtwo': 4,
    'Dragonite': 5,
    'Blastoise': 6,
    'Mew': 7,
    'Charizard': 8,
    'Squirtle': 9,
    'Articuno': 10,
    'Alakazam': 11,
    'Charmander': 12,
    'Lapras': 13,
    'Zapdos': 14,
    'Snorlax': 15,
    'Bulbasaur': 16,
    'Venusaur': 17,
    'Ninetales': 18,
    'Jolteon': 19,
    'Dragonair': 20,
    'Scyther': 21,
    'Eevee': 22,
    'Growlithe': 23,
    'Vaporeon': 24,
    'Haunter': 25,
    'Charmeleon': 26,
    'Pikachu': 27,
    'Raichu': 28,
    'Aerodactyl': 29,
    'Moltres': 30,
    'Ivysaur': 31,
    'Cubone': 32,
    'Dratini': 33,
    'Pidgeot': 34,
    'Flareon': 35,
    'Rapidash': 36,
    'Wartortle': 37,
    'Kabutops': 38,
    'Nidoking': 39,
    'Vulpix': 40,
    'Marowak': 41,
    'Kadabra': 42,
    'Machamp': 43,
    'Onix': 44,
    'Golem': 45,
    'Ditto': 46,
    'Electabuzz': 47,
    'Kangaskhan': 48,
    'Starmie': 49,
    'Rhydon': 50,
    'Poliwrath': 51,
    'Sandslash': 52,
    'Hitmonchan': 53,
    'Nidoqueen': 54,
    'Gastly': 55,
    'Butterfree': 56,
    'Cloyster': 57,
    'Abra': 58,
    'Primeape': 59,
    'Hitmonlee': 60,
    'Magmar': 61,
    'Tauros': 62,
    'Slowbro': 63,
    'Golduck': 64,
    'Rhyhorn': 65,
    'Seadra': 66,
    'Kingler': 67,
    'Muk': 68,
    'Pidgeotto': 69,
    'Dewgong': 70,
    'Machoke': 71,
    'Nidorino': 72,
    'Sandshrew': 73,
    'Arbok': 74,
    'Pinsir': 75,
    'Psyduck': 76,
    'Ponyta': 77,
    'Tentacruel': 78,
    'Vileplume': 79,
    'Chansey': 80,
    'Beedrill': 81,
    'Porygon': 82,
    'Poliwhirl': 83,
    'Venonat': 84,
    'Meowth': 85,
    'Magneton': 86,
    'Persian': 87,
    'Poliwag': 88,
    'Magikarp': 89,
    'Jigglypuff': 90,
    'Nidoran M': 91,
    'Staryu': 92,
    'Victreebel': 93,
    'Exeggutor': 94,  # Note: Also spelled 'Exeggcute' (evo of 'Exeggcute')
    'Machop': 95,
    'Omastar': 96,
    'Dugtrio': 97,
    'Oddish': 98,
    "Farfetch'd": 99,
    'Lickitung': 100,
    'Horsea': 101,
    'Hypno': 102,
    'Geodude': 103,
    'Nidorina': 104,
    'Nidoran F': 105,
    'Dodrio': 106,
    'Weezing': 107,
    'Slowpoke': 108,
    'Omanyte': 109,
    'Kabuto': 110,
    'Krabby': 111,
    'Caterpie': 112,
    'Pidgey': 113,
    'Mankey': 114,
    'Clefairy': 115,
    'Parasect': 116,
    'Clefable': 117,
    'Venomoth': 118,
    'Mr. Mime': 119,
    'Golbat': 120,
    'Graveler': 121,
    'Diglett': 122,
    'Seel': 123,
    'Tangela': 124,
    'Wigglytuff': 125,
    'Shellder': 126,
    'Magnemite': 127,
    'Ekans': 128,
    'Koffing': 129,
    'Fearow': 130,
    'Grimer': 131,
    'Seaking': 132,
    'Electrode': 133,
    'Voltorb': 134,
    'Gloom': 135,
    'Bellsprout': 136,
    'Doduo': 137,
    'Drowzee': 138,
    'Tentacool': 139,
    'Weepinbell': 140,
    'Jynx': 141,
    'Goldeen': 142,
    'Weedle': 143,
    'Exeggcute': 144,
    'Metapod': 145,
    'Raticate': 146,
    'Spearow': 147,
    'Paras': 148,
    'Kakuna': 149,
    'Rattata': 150,
    'Zubat': 151,
}


class PopularityRanker:
    """Manages Pokemon popularity rankings"""
    
    def __init__(self):
        """
        Init w popularity rankings
        """
        self.rankings = POKEMON_POPULARITY_RANKINGS
        self.df = self._create_rankings_dataframe()
    
    def _create_rankings_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from rankings dictionary"""
        df = pd.DataFrame([
            {'pokemon_name': name, 'popularity_rank': rank}
            for name, rank in self.rankings.items()
        ])
        
        # Normalize score to 1-100 scale (100 = most popular, 1 = least popular)
        df['popularity_score'] = self._normalize_rankings(df['popularity_rank'].values)
        
        return df.sort_values('popularity_rank')
    
    @staticmethod
    def _normalize_rankings(ranks: np.ndarray) -> np.ndarray:
        """
        Returns:
            Array of normalized scores (100 = most popular, 1 = least popular)
        """
        max_rank = ranks.max()
        
        # Linear normalization: (max_rank - rank) / (max_rank - 1) * 99 + 1
        normalized = ((max_rank - ranks) / (max_rank - 1)) * 99 + 1
        return np.round(normalized, 2)
    
    def get_popularity(self, pokemon_name: str) -> dict:
        """
        Get popularity info for a Pokemon

        Takes name and return rank n score
        """
        result = self.df[self.df['pokemon_name'] == pokemon_name]
        if len(result) > 0:
            row = result.iloc[0]
            return {
                'pokemon_name': pokemon_name,
                'popularity_rank': int(row['popularity_rank']),
                'popularity_score': round(float(row['popularity_score']), 2)
            }
        return None
    
    def merge_with_cards(self, cards_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge popularity data with card data from data_coll
        """
        merged = cards_df.merge(
            self.df[['pokemon_name', 'popularity_rank', 'popularity_score']],
            on='pokemon_name',
            how='left'
        )
        
        return merged


def create_popularity_dataframe() -> pd.DataFrame:
    """Create and return popularity rankings as DataFrame"""
    ranker = PopularityRanker()
    return ranker.df


if __name__ == "__main__":
    # Test the popularity rankings and normalization
    ranker = PopularityRanker()
    
    print("\nTop 10 Pokemons:")
    print(ranker.df.head(10)[['pokemon_name', 'popularity_rank', 'popularity_score']])
    
    print("\nBottom 10 Pokemon:")
    print(ranker.df.tail(10)[['pokemon_name', 'popularity_rank', 'popularity_score']])
