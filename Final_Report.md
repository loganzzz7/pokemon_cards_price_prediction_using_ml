# [Pokemon Card Price Prediction Using Machine Learning](https://github.com/loganzzz7/pokemon_cards_price_prediction_using_ml)

## Introduction

The market for collectible assets such as Pokémon trading cards has grown significantly over the past decade, with rare cards often appreciating substantially due to scarcity, nostalgia, grading, and public popularity. However, predicting card price movements remains challenging because prices are influenced by multiple dynamic factors including historical sales trends, character popularity, card rarity, condition grading, and broader hobbist sentiment.

This project aims to design and evaluate a machine learning framework to predict Pokémon card prices using historical pricing trends combined with self-framed popularity metrics.

### The Question:
Can machine learning models predict Pokémon card prices (non-graded, aka "raw" copy) using historical price movement trends and card-specific metrics--i.e. card rarity--as well as popularity guages--i.e. character on card?

---

## Background

Machine learning has been widely used in traditional financial markets--[stocks](https://www.nature.com/articles/s41599-024-02807-x)--including:

- Linear Regression
- Random Forest
- Gradient Boosting (XGBoost)
- Neural Networks (LSTM)

Alternative asset classes such as collectibles cards have seen less modeling due to more difficult data gathering processes and nicheness.

[Recent growth in Pokémon card investing](https://innotechtoday.com/why-pokemon-cards-went-crazy-during-the-pandemic/) presents an opportunity to apply machine learning to:

- Alternative asset valuation
- Consumer sentiment analysis
- Feature engineering for niche markets

### State of the Art:
While established existing marketplaces--i.e. [Card Ladder](https://www.cardladder.com/ladder?category=Pokemon)--provide price tracking, there is not a platform utilizing multi-dimensional features to predict future pricing using both market and popularity variables.

This project aims to be an innovative price prediction analysis on alternative assets centered on Pokemon cards. 

---

## Methodology / Approach

### Data Collection

The project will gather data from multiple verified sources:

#### Past Price Data:
- [PokeTrace API](https://poketrace.com/)

#### Card Feature Data (i.e. rarity, character on card):
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

---

## Findings

### Artifacts

**Final Dataset:**
- [Training Data: pokemon_sv_151_cleaned.csv](/data/processed/pokemon_sv_151_cleaned.csv) 
    - 381 Pokémon cards from SV-151 set
    - 356 valid samples (after removing NaN targets)
    - Train/Test Split: 80/20 (284 train, 72 test)

**Model Outputs:**
| Model | $r^2$ | RMSE | MAE |
|-------|------|------|------|
| XGBoost | 0.832 | $6.26 | $1.88 |
| LinearRegression | 0.878 | $5.34 | $2.88 |
| RandomForest | 0.772 | $7.29 | $1.82 |
| MovingAverage | -0.022 | $15.43 | $8.88 |
- [Predictions: predictions.csv](/models/predictions.csv) - All model predictions on test set
- [Metrics: metrics.csv](/models/metrics.csv) - RMSE, MAE, R² scores

### Visualizations

#### Actual vs Predicted Prices
![Actual vs Predicted](/models/actual_vs_predicted.png)
Scatter plots showing predicted prices vs actual prices for all four models (LinearRegression, MovingAverage, RandomForest, XGBoost).

#### Metrics Comparison
![Metrics Comparison](/models/metrics_comparison.png)
Performance comparison across models: RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and $r^2$ Score.

#### Feature Importance
![Feature Importance](/models/feature_importance.png)
Most influential features from RandomForest and XGBoost models, showing which features drive price predictions. 

### Results
The XGBoost model performed the best as expected when accounting for both $r^2$ (0.832) and MAE (1.88). The second model to follow was not Random Forest but rather Linear Regression--most likely due to the limit amount of training data and features. Given more time alloted for data gathering and predicting across multiple sets with more viable indicator features, I believe that both the XGBoost and Random Forest models should come out on top.

---

## Challenges Encountered

- Direct pricing data was not an honest indicator for the models as in a less volatile Pokemon TCG market, the limited **long-term** past pricing data meant that they often directly translated to the present day price. 
    - This was especially seen when the linear regression model performed the best when I still had past pricing data--7d avg, 30d avg--as predictive features.

### Solution
- Removed absolute price levels, kept only derived metrics:
    - Price momentum (% change direction)
    - Price volatility (standard deviation)

---

## Conclusion

This project showed that it is highly possible to predict the prices of pokemon cards with non-pricing related data as rarity, total sales, and the self-designated popularity metric ranked top 4 following price volatility as the most influential prediction features in the XGBoost model. Given a larger dataset and more than one set of cards to learn from, features such as set popularity and set sales should also be heavily influential to the models.