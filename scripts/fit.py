import pandas as pd
import yaml
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

def fit_model():
    # Загрузка гиперпараметров
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    # Загрузка данных
    data = pd.read_csv('data/initial_data.csv')
    target = 'target'
    X = data.drop(columns=target)
    y = data[target]

    # Категориальные и числовые признаки
    cat_features = X.select_dtypes(include='object')
    potential_binary_features = cat_features.nunique() == 2

    # Разделение категориальных признаков
    binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
    other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
    num_features = X.select_dtypes(include=['int', 'float'])

    # Предобработка
    preprocessor = ColumnTransformer(
        transformers=[
            ('binary', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False),
             binary_cat_features.columns.tolist()),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             other_cat_features.columns.tolist()),
            ('num', StandardScaler(), num_features.columns.tolist()),
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    # Модель
    # model = CatBoostClassifier(auto_class_weights='Balanced', verbose=0)
    # model = LogisticRegression(C=1.0, penalty='l2', solver='liblinear')
    model_params = params.get("model", {})
    model = LogisticRegression(
        C=model_params.get("C", 1.0),
        penalty=model_params.get("penalty", "l2"),
        solver=model_params.get("solver", "liblinear")
)

    # Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X, y)

    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    with open('models/fitted_model.pkl', 'wb') as fd:
        joblib.dump(pipeline, fd)

if __name__ == '__main__':
    fit_model()