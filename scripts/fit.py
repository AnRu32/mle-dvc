import pandas as pd
import yaml
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier

# Обучение модели
def fit_model():
    # Загрузка гиперпараметров из файла params.yaml
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    # Загрузка данных
    data = pd.read_csv('data/initial_data.csv')

    # Подготовка данных
    target = 'target' # Предполагаем, что ваш целевой столбец называется "target"
    X = data.drop(target, axis=1)  # Выделение признаков
    y = data[target]  # Целевая переменная

    cat_features = data.select_dtypes(include='object')
    potential_binary_features = cat_features.nunique() == 2

    binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
    other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
    num_features = data.select_dtypes(['float'])

    preprocessor = ColumnTransformer(
        [
            ('binary', OneHotEncoder(drop='if_binary'), binary_cat_features.columns.tolist()),
            ('cat', CatBoostEncoder(return_df=False), other_cat_features.columns.tolist()),
            ('num', StandardScaler(), num_features.columns.tolist())
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    # Конфигурация модели
    model = CatBoostClassifier(auto_class_weights='Balanced')  # Предполагается, что гиперпараметры модели хранятся под ключом 'model'

    # Создание и обучение пайплайна
    pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
     ])
    
    pipeline.fit(X, y)

    # Сохранение обученной модели
    os.makedirs('models', exist_ok=True) # создание директории, если её ещё нет
    model_path = 'models/fitted_model.pkl' 
    with open(model_path, 'wb') as fd:
        joblib.dump(pipeline, fd)

if __name__ == '__main__':
    fit_model()