# scripts/fit.py

import os
import joblib
import yaml
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier


def fit_model():
    # Прочитайте файл с гиперпараметрами params.yaml
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    index_col = params["index_col"]
    target_col = params["target_col"]
    one_hot_drop = params["one_hot_drop"]
    auto_class_weights = params["auto_class_weights"]

    # загрузите результат предыдущего шага: initial_data.csv
    data = pd.read_csv("data/initial_data.csv")

    # разделите данные на признаки и целевую переменную
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # обучение модели
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
                CatBoostEncoder(return_df=False),
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

    model = CatBoostClassifier(
        auto_class_weights=auto_class_weights,
        verbose=0,
        random_state=42
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X, y)

    # сохраните обученную модель в models/fitted_model.pkl
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/fitted_model.pkl")


if __name__ == "__main__":
    fit_model()