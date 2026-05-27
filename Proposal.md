# [Pokemon Card Price Prediction Using Machine Learning](https://github.com/loganzzz7/pokemon_cards_price_prediction_using_ml)

## Introduction

The market for collectible assets such as Pokémon trading cards has grown significantly over the past decade, with rare cards often appreciating substantially due to scarcity, nostalgia, grading, and public popularity. However, predicting card price movements remains challenging because prices are influenced by multiple dynamic factors including historical sales trends, character popularity, card rarity, condition grading, and broader hobbist sentiment.

This project aims to design and evaluate a machine learning framework to predict Pokémon card prices using historical pricing data combined with self-framed popularity metrics.

### The Problem:
Can machine learning models predict future Pokémon card prices (non-graded, aka "raw" copy) using historical price data and card-specific popularity guages--i.e. character on card, card rarity?

---

## Background

Machine learning has been widely used in traditional financial markets--[stocks](https://www.nature.com/articles/s41599-024-02807-x)--including:

- Linear Regression
- Random Forest
- Gradient Boosting (XGBoost)
- Neural Networks (LSTM)

Alternative asset classes such as collectibles cards have seen less modeling due to more difficult data gathering and nicheness.

[Recent growth in Pokémon card investing](https://innotechtoday.com/why-pokemon-cards-went-crazy-during-the-pandemic/) presents an opportunity to apply machine learning to:

- Asset valuation
- Consumer sentiment analysis
- Feature engineering for niche markets

### State of the Art:
While established existing marketplaces--i.e. [Card Ladder](https://www.cardladder.com/ladder?category=Pokemon)--provide price tracking, there is not a platform utilizing multi-dimensional features to predict future pricing using both market and popularity variables.

This project aims to be an innovative price prediction analysis on alternative assets centered on Pokemon cards. 

---

## Methodology / Approach

### Data Collection

The project will gather data from multiple verified sources:

#### Price Data:
- [PokeTrace API](https://poketrace.com/)

#### Card Feature Data (i.e. rarity):
- [PokeTrace API](https://poketrace.com/)

#### Popularity Features:
- [Pokémon character popularity rankings](https://www.ranker.com/list/best-generation-1-pokemon/ranker-pokemon)

### Data Verification:
To avoid invalid assumptions:

- Cross-source validation of pricing data--eBay and TCGplayer

---

### Feature Engineering

Predictive features include:

- Card rarity
- Price momentum (% change from 30d avg)
- Price volatility (std dev of 30d prices)
- Sale volume
- Popularity rating
~~- Previous 7-day / 30-day price averages~~(in a more stable market the recent price averages were minimally different from the current price, so I decided to stick with non-highly-indicative features)
~~- Set release age ~~(decided to focus on the original 151 pokemons and narrow down to 1 set--sv-151--instead)

---

### Models to Implement

#### Baseline:
- Linear Regression
- Moving Average

#### Advanced:
- Random Forest Regressor
- XGBoost

---

### Evaluation Metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- $R^2$ Score

---

### Software / Packages:
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn
- Google Trends API (pytrends)
- Potentially Web scraping tools (BeautifulSoup / Selenium / Headless Browser--I will not violate robot.txt if crawling is forbidden)

---

## Deliverables

### Cleaned Dataset
- Historical Pokémon card prices
- Popularity metrics
- Chosen predictive features

### Predictive Models
- Baseline models
- Advanced machine learning models

### Benchmark Analysis
- Model performance comparison
- Error analysis

### Visualizations
- Price trend predictions
- Feature correlation heatmaps
- Predicted pricing overlayed with real
- Comparative model performance

### Final Report
- Problem definition
- State of the art review
- Dataset methodology
- Model results
- Discussion of findings and limitations
- Future improvements

---

## Potential Challenges

- Market manipulation or false price movements
- Limited long-term data availability
- Popularity metric correlation

---

## Conclusion

This project aims to show whether collectible asset prices can be meaningfully predicted machine learning techniques. Combining historical pricing data with popularity features. Through data gathering, model benchmarking, and scientific methodology.