import pandas as pd
import os
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import CatBoostEncoder
from catboost import CatBoostRegressor
from dotenv import load_dotenv
from sqlalchemy import create_engine


# настройка логирования (если ранее не задано):
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_connection():
    host = 'rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net'
    port = 6432
    db = 'playground_mle_20250530_d398f4a560'
    username = 'mle_20250530_d398f4a560' #mle_ro
    password = '83284b762ddc448d83912fd1d83a40fd'

    connection_str = f'postgresql://{username}:{password}@{host}:{port}/{db}'
    logger.info(f"📡 Подключение к БД: {connection_str}")
    
    engine = create_engine(connection_str, connect_args={'sslmode': 'require'})
    return engine

def get_clean_data() -> pd.DataFrame:
    """Загрузка очищенных данных напрямую из БД и сохранение csv."""

    engine = create_connection()
    query = "SELECT * FROM real_estate_dataset_clean"

    # Загружаем данные напрямую из БД через Pandas
    df = pd.read_sql(query, engine)
    logger.info(f"📥 Загружено строк из БД: {len(df)}")

    # Сохраняем в нужный CSV (относительный путь ../../../data)
    data_dir = '/home/mle-user/mle_projects/mle-dvc/Progect1/mle-project-sprint-1-v001/part2_dvc/data'
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, 'clean_data.csv')
    df.to_csv(data_path, index=False)
    logger.info(f"💾 Данные сохранены в {data_path}")

    return df

def train_model(df: pd.DataFrame):
    """Обучение модели с разделением на выборки и оценкой."""
    target = 'price'
    X = df.drop(target, axis=1)
    y = df[target]

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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

    model = CatBoostRegressor(silent=True)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Обучение на train-выборке
    pipeline.fit(X_train, y_train)

    # Оцениваем модель на тестовой выборке
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    logger.info(f"✅ MAE модели на тестовой выборке: {mae:.2f}")
    logger.info(f"✅ RMSE модели на тестовой выборке: {rmse:.2f}")
    logger.info(f"✅ R2 модели на тестовой выборке: {r2:.2f}")

    # Сохраняем модель
    model_dir = '/home/mle-user/mle_projects/mle-dvc/Progect1/mle-project-sprint-1-v001/part2_dvc/models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'price_predictor_model.pkl')
    joblib.dump(pipeline, model_path)
    logger.info(f"📦 Модель сохранена в {model_path}")

def main():
    logger.info("🚀 Скрипт начал работу. Извлекаем данные.")
    df = get_clean_data()  # загружаем данные из БД
    logger.info(f"✅ Извлечено строк данных: {len(df)}")

    # обучаем модель
    train_model(df)

    logger.info("🏁 Скрипт завершил работу.")

if __name__ == "__main__":
    main()