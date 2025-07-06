import pandas as pd
import yaml
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Обучение модели
def fit_model():
    # Загрузка гиперпараметров из файла params.yaml
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    # Загрузка данных
    data = pd.read_csv('data/initial_data.csv')

    # Подготовка данных
    target = 'target'
    X = data.drop(target, axis=1)
    y = data[target]

    # Выделение типов признаков
    cat_features = data.select_dtypes(include='object').columns
    num_features = data.select_dtypes(include=['float']).columns

    # Создание преобразователя признаков
    preprocessor = ColumnTransformer(
        [
            # OneHotEncoder для всех категориальных признаков
            ('cat', OneHotEncoder(), cat_features),
            ('num', StandardScaler(), num_features)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    # Создание модели LogisticRegression с параметрами из YAML
    model = LogisticRegression(
        C=params['model']['C'],
        penalty=params['model']['penalty'],
        solver=params['model']['solver'],
        max_iter=params['model']['max_iter'],
        random_state=params['model']['random_state']
    )

    # Создание и обучение пайплайна
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X, y)

    # Сохранение обученной модели
    os.makedirs('models', exist_ok=True)
    model_path = 'models/fitted_model.pkl' 
    with open(model_path, 'wb') as fd:
        joblib.dump(pipeline, fd)

if __name__ == '__main__':
    fit_model()