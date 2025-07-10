import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
import joblib
import json
import yaml
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Функция оценки модели
def evaluate_model():
    # Загрузка гиперпараметров из params.yaml
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    target = params.get('target', 'price')
    input_path = params.get('input_data', 'data/initial_data.csv')
    model_path = params.get('model_output', 'models/price_predictor_model.pkl')

    # Загрузка данных и модели
    data = pd.read_csv(input_path)
    logger.info(f"📥 Прочитано {len(data)} строк из {input_path}")

    if target not in data.columns:
        raise ValueError(f"❌ Целевой столбец '{target}' не найден в данных.")

    X = data.drop(columns=[target])
    y = data[target]

    model = joblib.load(model_path)
    logger.info(f"✅ Модель загружена из {model_path}")

    # Подготовка кросс-валидации и кастомных метрик
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        'mae': make_scorer(mean_absolute_error),
        'r2': make_scorer(r2_score)
    }

    logger.info("🔄 Начало кросс-валидации...")
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    avg_results = {
        'mae': round(cv_results['test_mae'].mean(), 3),
        'r2': round(cv_results['test_r2'].mean(), 3)
    }

    logger.info(f"📊 MAE: {avg_results['mae']}")
    logger.info(f"📈 R²: {avg_results['r2']}")

    #  Сохранение результатов в JSON
    OUT_DIR = 'Progect1/mle-project-sprint-1-v001/part2_dvc/cv_results'
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(OUT_DIR, 'evaluation_results.json'), 'w') as fd:
        json.dump(avg_results, fd)

    logger.info(f"✅ Результаты сохранены в {os.path.join(OUT_DIR, 'evaluation_results.json')}")


if __name__ == '__main__':
    evaluate_model()