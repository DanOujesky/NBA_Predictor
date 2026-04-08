# NBA Predictor

Machine learning system for predicting NBA game outcomes. Uses multiple data sources, advanced feature engineering, and a model comparison framework to find the best predictor.

## Architecture

```
nba_predictor/
    app.py                  Flask web server entry point
    config.py               Central configuration
    pipeline.py             Data collection + training pipeline
    requirements.txt        Python dependencies

    data/
        basketball_ref.py   Source 1: Basketball Reference scraper
        nba_stats.py        Source 2: Official NBA Stats API
        processor.py        Data cleaning and merging

    features/
        team_features.py    Rolling team stats, form, streaks, rest
        player_features.py  Player value scores, roster strength, injuries
        builder.py          Combines all features into training dataset

    models/
        base.py             Abstract base class for all models
        logistic.py         Logistic Regression (linear baseline)
        xgboost_model.py    XGBoost gradient-boosted trees
        random_forest.py    Random Forest ensemble
        evaluator.py        Side-by-side model comparison

    web/
        routes.py           Flask API endpoints
        templates/
            index.html      Dashboard frontend
```

## Data Sources

1. **Basketball Reference** (basketball-reference.com): Historical game logs going back 5 seasons. Provides box scores, shooting percentages, and schedule data.

2. **NBA Stats API** (stats.nba.com via nba_api): Current-season team game logs, advanced stats (offensive/defensive ratings, pace), player per-game averages, and injury availability derived from games played.

## Features

The prediction model uses 12 engineered features computed as differentials between the two teams in each matchup:

| Feature      | Description                                 |
| ------------ | ------------------------------------------- |
| diff_pts     | Rolling scoring average differential        |
| diff_opp_pts | Rolling points allowed differential         |
| diff_fg_pct  | Rolling field goal percentage differential  |
| diff_reb     | Rolling rebounding differential             |
| diff_ast     | Rolling assists differential                |
| diff_tov     | Rolling turnover differential (inverted)    |
| diff_form    | Recent win rate differential (last 5 games) |
| diff_streak  | Win/loss streak differential                |
| diff_rest    | Rest days differential                      |
| diff_roster  | Roster availability ratio differential      |
| is_home      | Home court advantage flag                   |
| is_b2b       | Back-to-back game flag                      |

All rolling statistics use a 10-game window and are shifted by one game to prevent data leakage.

## Models

Three models are trained and compared automatically:

- **Logistic Regression**: Linear baseline with built-in feature scaling. Fast to train, highly interpretable coefficients.
- **XGBoost**: Gradient-boosted decision trees with automated hyperparameter tuning via RandomizedSearchCV.
- **Random Forest**: Ensemble of decision trees, also auto-tuned. Good at capturing non-linear feature interactions.

The evaluator trains all three on the same train/test split and compares them on accuracy, ROC-AUC, F1 score, log loss, and cross-validated accuracy. The best model (by ROC-AUC) is saved automatically.

## Setup

```bash
py -m pip install -r requirements.txt
```

## Usage

### 1. Run the full pipeline (collect data, build features, train models):

```bash
py pipeline.py
```

This takes time on first run due to data collection. Use `--skip-scrape` on subsequent runs to reuse existing data:

```bash
py pipeline.py --skip-scrape
```

To only retrain models without re-processing data:

```bash
py pipeline.py --train-only
```

### 2. Start the web dashboard:

```bash
py app.py
```

Open http://localhost:5000 in your browser. The dashboard shows game predictions, model comparison metrics, and the injury report.

### API Endpoints

| Endpoint             | Description                                            |
| -------------------- | ------------------------------------------------------ |
| GET /api/predictions | Game predictions with win probabilities                |
| GET /api/models      | Model comparison table                                 |
| GET /api/injuries    | Injury/availability report (optional ?team=BOS filter) |
| GET /api/teams       | List of all NBA teams                                  |
| GET /api/status      | System status (model trained, data available)          |

## Adding a New Model

1. Create a new file in `models/` (e.g., `models/neural_net.py`)
2. Subclass `BaseModel` from `models/base.py`
3. Implement `name`, `train`, `predict`, `predict_proba`, and `get_params`
4. Add an instance to the models list in `pipeline.py` line ~68:

```python
models = [
    LogisticModel(),
    XGBoostModel(auto_tune=True),
    RandomForestModel(auto_tune=True),
    YourNewModel(),
]
```

5. Run `python pipeline.py --train-only` to compare all models

## Accuracy Notes

NBA game prediction is inherently uncertain due to the high variance in basketball outcomes. The theoretical upper bound for prediction accuracy with historical team stats alone is approximately 70-75%. The key factors that limit accuracy are in-game variance (hot/cold shooting nights), referee decisions, and player motivation/effort which cannot be quantified from box scores.

This system maximizes signal by combining offensive and defensive metrics, player availability impact, rest/scheduling effects, and momentum indicators. The model comparison framework makes it easy to test whether adding new features or trying new algorithms improves performance.
