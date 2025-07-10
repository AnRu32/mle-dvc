import logging
import pandas as pd
import pendulum
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Настройка логирования 25
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dag(
    schedule_interval="@once",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["real_estate"],
)
def real_estate_etl():

    @task
    def create_table():
        """Создание таблицы, если она не существует."""
        hook = PostgresHook('destination_db')
        conn = None
        try:
            conn = hook.get_conn()
            logger.info("✅ Успешное подключение к БД")
        except Exception as e:
            logger.error(f"Ошибка подключения к БД: {e}")
            raise
        
        engine = create_engine(hook.get_uri())
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS real_estate_dataset (
                    flat_id INT PRIMARY KEY,
                    building_id INT,
                    floor INT,
                    kitchen_area FLOAT,
                    living_area FLOAT,
                    rooms INT,
                    is_apartment BOOLEAN,
                    studio BOOLEAN,
                    total_area FLOAT,
                    price FLOAT,
                    build_year INT,
                    building_type_int INT,
                    latitude FLOAT,
                    longitude FLOAT,
                    ceiling_height FLOAT,
                    flats_count INT,
                    floors_total INT,
                    has_elevator BOOLEAN
                );
            """))
            logger.info("✅ Таблица создана")

    @task
    def extract_data() -> pd.DataFrame:
        """Извлечение данных из базы данных."""
        hook = PostgresHook('destination_db')
        conn = None
        try:
            conn = hook.get_conn()
            logger.info("✅ Успешное подключение к БД для извлечения данных")
        except Exception as e:
            logger.error(f"Ошибка подключения к БД для извлечения данных: {e}")
            raise

        query = """
            SELECT 
                f.id as flat_id,
                f.building_id,
                f.floor,
                f.kitchen_area,
                f.living_area,
                f.rooms,
                f.is_apartment,
                f.studio,
                f.total_area,
                f.price,
                b.build_year,
                b.building_type_int,
                b.latitude,
                b.longitude,
                b.ceiling_height,
                b.flats_count,
                b.floors_total,
                b.has_elevator
            FROM flats f
            LEFT JOIN buildings b ON f.building_id = b.id
        """
        df = pd.read_sql(query, conn)
        logger.info("📥 Извлечено строк: %d", len(df))
        return df

    @task
    def transform_data(df: pd.DataFrame) -> pd.DataFrame:
        """Очистка и базовая трансформация данных."""
        df = df.convert_dtypes()
        df['has_elevator'] = df['has_elevator'].fillna(False)
        logger.info("🔧 Трансформация завершена")
        return df

    @task
    def load_data(df: pd.DataFrame):
        """Загрузка очищенных данных в базу данных."""
        hook = PostgresHook('destination_db')
        engine = None
        try:
            engine = create_engine(hook.get_uri())
            logger.info("✅ Успешное подключение к БД для загрузки данных")
        except Exception as e:
            logger.error(f"Ошибка подключения к БД для загрузки данных: {e}")
            raise

        df.to_sql(
            'real_estate_dataset',
            engine,
            if_exists='replace',
            index=False,
            method='multi',
            chunksize=1000
        )
        logger.info("📤 Данные загружены в БД")

    # Последовательность задач в DAG
    create_table_task = create_table()
    df = extract_data()
    cleaned_df = transform_data(df)
    load_data(cleaned_df)

# Создание DAG
dag = real_estate_etl()