# APSAPT - Automated Pipeline for Sentiment Analysis of Political Tweets

## Directory Structure

```bash
├── README.md               <- The top-level README for developers using this project.
├── data
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
│
├── models                  <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks               <- Jupyter notebooks.
│
├── references              <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated graphics and figures to be used in reporting
│
├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
│                              generated with `pip freeze > requirements.txt`
│
├── src                     <- Source code for use in this project.
│   └──
│      ├── __init__.py     <- Makes src a Python module
│      │
│      ├── config.py       <- Configuration file for some global variables
│      │
│      ├── data            <- Scripts to download or generate data
│      │   └── make_dataset.py
│      │
│      ├── features        <- Scripts to turn raw data into features for modeling
│      │   └── build_features.py
│      │
│      ├── models          <- Scripts to train models and then use trained models to make
│      │   │                 predictions
│      │   ├── predict.py
│      │   └── train.py
│      │
│
└── .gitignore

```
