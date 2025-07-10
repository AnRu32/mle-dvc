import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_connection():
    load_dotenv()
    host = os.environ.get('DB_DESTINATION_HOST')
    port = os.environ.get('DB_DESTINATION_PORT')
    db = os.environ.get('DB_DESTINATION_NAME')
    username = os.environ.get('DB_DESTINATION_USER')
    password = os.environ.get('DB_DESTINATION_PASSWORD')

    connection_str = f'postgresql://{username}:{password}@{host}:{port}/{db}'
    logger.info(f"📡 Подключение к базе данных: {connection_str}")
    
    engine = create_engine(connection_str, connect_args={'sslmode': 'require'})  # Можно убрать sslmode при подключениях по локалке
    return engine

def get_data() -> pd.DataFrame:
    # загрузка параметров
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    table_name = params.get('data_table', 'real_estate_dataset_clean')
    index_col = params.get('index_col', None)

    # подключение и извлечение
    engine = create_connection()
    logger.info(f"📥 Загрузка данных из таблицы {table_name}")
    df = pd.read_sql(f"SELECT * FROM {table_name}", con=engine) #, index_col=index_col)
    engine.dispose()

    logger.info(f"✅ Извлечено строк: {len(df)}")
    
    # # сохранение в CSV
    # os.makedirs('data', exist_ok=True)
    # df.to_csv('data/initial_data.csv', index=False)
    # logger.info("💾 Данные сохранены в data/initial_data.csv")

    data_dir = Path('/home/mle-user/mle_projects/mle-dvc/Progect1/mle-project-sprint-1-v001/part2_dvc/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / 'initial_data.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"💾 Данные сохранены в {output_path}")

    return df

if __name__ == '__main__':
    df = get_data()
    print(df.head())