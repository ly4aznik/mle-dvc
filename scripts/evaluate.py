# scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

import joblib
import json
import yaml
import os

# оценка качества модели
def evaluate_model():
    # прочитайте файл с гиперпараметрами params.yaml
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    
    target_col = params["target_col"]
    index_col = params["index_col"]

    n_jobs = params["n_jobs"]
    n_splits = params["n_splits"]
    metrics = params["metrics"]
    one_hot_drop = params["one_hot_drop"]
    C = params["C"]
    penalty = params["penalty"]
    # загрузите результат прошлого шага: fitted_model.pkl
    model = joblib.load("models/fitted_model.pkl")
    # реализуйте основную логику шага с использованием прочтённых гиперпараметров
    data = pd.read_csv("data/initial_data.csv")

    X = data.drop(columns=[target_col])
    y = data[target_col]
    cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cat_features = X.select_dtypes(include="object")
    potential_binary_features = cat_features.nunique() == 2

    binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
    other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
    num_features = X.select_dtypes(include=["float", "int"])

    preprocessor = ColumnTransformer(
        [
            (
                "binary",
                OneHotEncoder(drop=one_hot_drop, handle_unknown="ignore"),
                binary_cat_features.columns.tolist()
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                other_cat_features.columns.tolist()
            ),
            (
                "num",
                StandardScaler(),
                num_features.columns.tolist()
            )
        ],
    remainder="drop",
    verbose_feature_names_out=False
)

    model = LogisticRegression(
        C=C,
        penalty=penalty,
        random_state=42,
        max_iter=1000
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X, y)

    cv_res = cross_validate(
        estimator=pipeline,
        X=X,
        y=y,
        cv=cv_strategy,
        n_jobs=n_jobs,
        scoring=metrics
    )

    for key, value in cv_res.items():
        cv_res[key] = round(value.mean(), 3) 
        # сохраните результата кросс-валидации в cv_res.json
    os.makedirs("cv_results", exist_ok=True)

    with open("cv_results/cv_res.json", "w") as f:
        json.dump(cv_res, f)

if __name__ == '__main__':
    evaluate_model()