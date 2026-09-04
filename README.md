# NFL Draft Undervaluation Analysis

This project uses NFL Combine, draft, and player performance data to find players who may have been undervalued or overvalued in the NFL Draft.

I used Python and scikit-learn to build a model that predicts player performance based on draft position. Then, I used the difference between predicted and actual performance to find players who performed better or worse than expected.

## Project Goal

The goal of this project was to answer:

> Which players outperformed expectations based on where they were drafted?

A player drafted late who performs much better than expected may be considered undervalued. A highly drafted player who performs worse than expected may be considered overvalued.

## Data Used

This project combines:

- NFL Combine data
- Pre-draft player data
- NFL Draft data
- Post-draft player performance data

The data was cleaned, standardized, and merged into master datasets before modeling.

## What I Did

- Cleaned NFL Combine and draft data.
- Standardized player names across different datasets.
- Handled missing values using preprocessing and imputation.
- Combined combine, draft, and player performance data.
- Split the data into training, validation, and test sets.
- Built a linear regression baseline model.
- Tested additional models.
- Used residuals to identify players who performed above or below expectations.

## Model Results

The baseline model used draft position to predict player performance.

- **R²:** Approximately 0.35
- Draft position explained about **35% of the variation** in player performance.

This means draft position is useful, but it does not explain everything. NFL success can also depend on injuries, coaching, opportunity, team situation, development, and many other factors.

## Residual Analysis

Residuals were calculated using:

```text
Residual = Actual Performance - Predicted Performance
```

- A **positive residual** means a player performed better than expected based on draft position.
- A **negative residual** means a player performed worse than expected based on draft position.

Players with large positive residuals can be viewed as possible draft steals or undervalued players.

## Files

- `nfl_combine_data_cleaning.ipynb` — Cleaning and preparing NFL Combine data.
- `standardize_nfl_draft_data.py` — Standardizes NFL draft data.
- `build_nfl_master_datasets.py` — Merges datasets into master files.
- `standardized_combine_data.csv` — Cleaned and standardized Combine data.
- `standardized_pre_draft_player_data.csv` — Standardized pre-draft player data.
- `standardized_post_draft_player_data.csv` — Standardized post-draft player data.
- `nfl_master_dataset_a_2014_2018.csv` — Master dataset for 2014–2018.
- `nfl_master_dataset_b_2000_2018.csv` — Master dataset for 2000–2018.
- `nfl_master_dataset_c_2006_2018.csv` — Master dataset for 2006–2018.

## Tools Used

- Python
- pandas
- NumPy
- scikit-learn
- Jupyter Notebook
- Linear Regression
- Missing-value imputation
- Train/validation/test splits
- Residual analysis

## Limitations

- Draft position cannot capture every reason a player succeeds or struggles in the NFL.
- Player performance can be affected by injuries, playing time, coaching, team fit, and opportunity.
- Some NFL Combine information is missing for certain players.
- Performance statistics can be different across positions.
- Being identified as undervalued does not prove a team made a mistake; it means the player performed better than the model expected based on draft position.

## Conclusion

This project shows that draft position has value for predicting NFL performance, but it only explained about 35% of performance differences. By looking at residuals, I was able to identify players who outperformed or underperformed expectations based on where they were drafted.
