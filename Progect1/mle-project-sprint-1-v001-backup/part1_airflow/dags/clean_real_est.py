import logging
import pandas as pd
import pendulum
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sqlalchemy import create_engine
from sqlalchemy import text
from steps.CleanData import EmptyRR, Dublikates, EmptyCells, Vibros, Vibros2
from steps.messages import send_telegram_success_message, send_telegram_failure_message


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dag(
    schedule_interval="@once",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["real_estate_cleaning"],
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message,   
)
def real_estate_data_cleaning():

    @task
    def extract_data() -> pd.DataFrame:
        """Извлечение данных из таблицы real_estate_dataset."""
        hook = PostgresHook('destination_db')
        try:
            conn = hook.get_conn()
            logger.info("✅ Успешное подключение к БД для извлечения данных")
        except Exception as e:
            logger.error(f"Ошибка подключения к БД для извлечения данных: {e}")
            raise

        query = "SELECT * FROM real_estate_dataset"
        df = pd.read_sql(query, conn)
        logger.info("📥 Извлечено строк: %d", len(df))
        return df

    @task
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Очистка и трансформация данных."""
        cols_b1 = df.shape[1]
        initial_lenb1 = len(df)
    
        df = EmptyRR(df)
        df = Dublikates(df)
        df = EmptyCells(df)
        df = Vibros2(df)
        #df = Vibros(df)
    
        cols_after1 = df.shape[1]
        logger.info("🗑️ Всего Удалено столбцов: %d", cols_b1 - cols_after1)
        logger.info("🗑️ Всего Удалено строк: %d", initial_lenb1 - len(df))
        logger.info("🔧 Трансформация и очистка завершены")
        return df

    @task
    def load_cleaned_data(df: pd.DataFrame):
        """Полная перезапись таблицы real_estate_dataset_clean новыми данными"""
        try:
            logger.info("🔄 Начало загрузки данных (полная перезапись таблицы)")

            # Получаем подключение через PostgresHook
            hook = PostgresHook(postgres_conn_id='destination_db')
            conn = hook.get_connection('destination_db')
            
            # Формируем строку подключения
            connection_uri = f"postgresql+psycopg2://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema}"
            engine = create_engine(connection_uri)

            # Удаление старой таблицы (если существует)
            with engine.begin() as connection:
                connection.execute(text("DROP TABLE IF EXISTS real_estate_dataset_clean CASCADE"))
                logger.info("🗑️ Старая таблица удалена (если существовала)")

                # Создание новой таблицы и загрузка данных
                df.to_sql(
                    name='real_estate_dataset_clean',
                    con=connection,
                    if_exists='fail',  # Гарантируем, что таблицы не существует
                    index=False,
                    method='multi',
                    chunksize=1000
                )
                logger.info("✅ Новая таблица создана и данные загружены")

        except Exception as e:
            logger.error(f"❌ Критическая ошибка: {str(e)}")
            raise

    # Последовательность задач в DAG
    df = extract_data()
    cleaned_df = clean_data(df)
    load_cleaned_data(cleaned_df)

# Создание DAG
dag = real_estate_data_cleaning()