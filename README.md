# Pokemon Card Price Prediction Using Machine Learning

Predicting Pokémon trading card prices using machine learning models trained on historical price data, card rarity, and popularity metrics.

---

## Quick Start

### Setup

1. **Install Python 3.9+** and create a conda environment:
   ```bash
   conda create -n your_env python=3.9
   conda activate your_env
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API key:**
   Create a `.env` file in the project root:
   ```
   POKE_TRACE_API_KEY=your_api_key_here
   ```
   Get your API key from [PokeTrace](https://poketrace.com/)

---

## Running the Code

### Option 1: Run the Full Pipeline (Collect -> Train -> Evaluate)

```bash
cd src/data_gathering
python data_pipeline.py
```

This will:
- Fetch 381 Pokémon cards from the API (~6 min)
- Merge popularity rankings
- Fetch 1-year price history for each card (~15 min)
- Generate `data/processed/pokemon_sv_151_cleaned.csv`

**Total runtime: ~20 minutes**

Then train models:
```bash
cd ../models
python train_pipeline.py
```

Finally, evaluate and visualize:
```bash
python eval.py
```

### Option 2: Use Existing Data

If you already have the cleaned dataset--feel free to use the ones I used in data, just train and evaluate:

```bash
cd src/models
python train_pipeline.py
python eval.py
```

This will:
- Load pre-collected data
- Train 4 models (Linear Regression, Random Forest, XGBoost, Moving Average)
- Generate predictions and performance metrics
- Create 3 visualization PNGs

---

## Output Files

After running the pipeline:

| File | Description |
|------|-------------|
| `pokemon_sv_151_cleaned.csv` | 381 cards × 37 features (training data) |
| `predictions.csv` | Predictions from all 4 models |
| `metrics.csv` | Performance metrics: RMSE, MAE, $r^2$ |
| `actual_vs_predicted.png` | Visual comparison of predictions |
| `metrics_comparison.png` | Model performance bar charts |
| `feature_importance.png` | Top features from RandomForest & XGBoost |

---

## Features Used (6 total)

1. **rarity** — Card rarity level (1-8 ordinal scale)
2. **popularity_rank** — Character popularity rank (1-151)
3. **popularity_score** — Normalized popularity (1-100)
4. **total_sales** — Total historical sales volume
5. **price_momentum** — % price change from 30-day average
6. **price_volatility** — Standard deviation of 30-day prices
