import pandas as pd
import yaml
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostRegressor
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Обучение модели
def fit_model():
    # Загрузка гиперпараметров из файла params.yaml
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    target = params.get('target', 'price')  # Выбор целевой функции
    model_path = params.get('model_output', 'models/price_predictor_model.pkl')
    input_path = params.get('input_data', 'data/initial_data.csv')

    # Загрузка данных
    data = pd.read_csv(input_path)
    logger.info(f"📥 Загружено {len(data)} строк из {input_path}")

    # Подготовка признаков
    if target not in data.columns:
        raise ValueError(f"❌ Целевой столбец '{target}' не найден в данных.")

    X = data.drop(target, axis=1)
    y = data[target]

    # Разделение признаков по типам
    cat_features = X.select_dtypes(include='object')
    potential_binary_features = cat_features.nunique() == 2

    binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
    other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
    num_features = X.select_dtypes(include=['float64', 'int64'])

    preprocessor = ColumnTransformer(
        [
            ('binary', OneHotEncoder(drop='if_binary'), binary_cat_features.columns.tolist()),
            ('cat', CatBoostEncoder(return_df=False), other_cat_features.columns.tolist()),
            ('num', StandardScaler(), num_features.columns.tolist())
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    # Конфигурация и обучение модели
    model = CatBoostRegressor(silent=True)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X, y)
    logger.info("✅ Модель успешно обучена")

    # Сохранение модели
    OUT_PATH = 'Progect1/mle-project-sprint-1-v001/part2_dvc/models'
    os.makedirs(OUT_PATH, exist_ok=True)

    joblib.dump(model, os.path.join(OUT_PATH, 'price_predictor_model.pkl'))
    logger.info(f"📦 Модель сохранена в файл: {os.path.join(OUT_PATH, 'price_predictor_model.pkl')}")

if __name__ == '__main__':
    fit_model()