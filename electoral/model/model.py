"""
The goal of this notebook is to train a model to predict the abstension rate of municipalities.
We will need to perform feature engineering carefully to account for spatial relations.
"""

import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

def prepare_splits(
    data: pd.DataFrame, 
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Splits the dataset into train, validation, and test sets based on blocks."""
    
    target_col = config["model"]["target_col"]
    feature_cols = config["model"]["feature_cols"]
    group_col = config["model"]["group_col"]
    cat_cols = config["model"]["categorical_cols"]
    
    Y = data[target_col].copy()
    X = data[feature_cols].copy()

    # Handle categoricals dynamically
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")

    # First Split: Train vs Temp (Val + Test)
    gss = GroupShuffleSplit(
        n_splits=1, 
        test_size=config["training"]["split_1_test_size"], 
        random_state=config["training"]["random_state"]
    )
    
    train_idx, temp_idx = next(gss.split(X, Y, groups=data[group_col]))

    # Second Split: Validation vs Test
    gss2 = GroupShuffleSplit(
        n_splits=1, 
        test_size=config["training"]["split_2_test_size"], 
        random_state=config["training"]["random_state"]
    )
    
    val_rel, test_rel = next(
        gss2.split(
            X.iloc[temp_idx], 
            Y.iloc[temp_idx], 
            groups=data[group_col].iloc[temp_idx]
        )
    )

    # Resolve indices
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    y_train, y_val, y_test = Y.iloc[train_idx], Y.iloc[val_idx], Y.iloc[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test





def train_and_evaluate(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,
    config: dict
) -> Tuple[Pipeline, Dict[str, float]]:
    """Defines the model, trains it, and evaluates its performance."""
    
    xgb_params = config["training"]["xgboost"]

    early_stopping = xgb_params.pop("early_stopping_rounds")
    
    # Initialize Pipeline
    pipe = Pipeline([
        ("XGB", XGBRegressor(random_state=config["training"]["random_state"], **xgb_params))
    ])

    print("-- Model Starting --")

    
    pipe.fit(
        X_train, y_train, 
        XGB__eval_set=[(X_val, y_val)], 
        XGB__verbose=False,
        
    )

    # Evaluation
    score = pipe.score(X_test, y_test)
    y_pred = pipe.predict(X_test)
    rmse = root_mean_squared_error(y_true=y_test, y_pred=y_pred)
    
    metrics = {
        "r_squared": score,
        "rmse": rmse
    }
    
    print(f"-- Model Score (R-squared) --\n{score:.4f}")
    print(f"-- Model RMSE --\n{rmse:.4f}")

    return pipe, metrics