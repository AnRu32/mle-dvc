import pandas as pd
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def EmptyRR(df: pd.DataFrame) -> pd.DataFrame:
    """Удаление строк с NaN или нулевым 'price', удаление полностью пустых столбцов и 
    столбцов с одинаковыми значениями."""
    cols_before1 = df.shape[1]

    # Удаление строк с NaN или нулевым 'price'
    initial_len = len(df)
    df = df.dropna(subset=['price'])
    df = df[df['price'] != 0]
    logger.info("🧹 Удалено строк с отсутствующей или нулевой ценой: %d", initial_len - len(df))
    
    # Удаление полностью пустых столбцов
    cols_before = df.shape[1]
    df = df.dropna(axis=1, how='all')
    cols_after = df.shape[1]
    logger.info("🗑️ Удалено пустых столбцов: %d", cols_before - cols_after)
    
    # Удаление столбцов с одинаковыми значениями
    cols_before = df.shape[1]
    identical_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=identical_cols)
    cols_after = df.shape[1]
    logger.info("🗑️ Удалено столбцов с одинаковыми значениями: %d", cols_before - cols_after)

    # Приведение столбцов к формату 0,00
    for column in ['kitchen_area', 'living_area', 'total_area', 'ceiling_height']:
        if column in df.columns:
            df[column] = df[column].round(2)
            logger.info(f"🔹 Столбец '{column}' округлён до двух знаков после запятой.")
    
   
    return df


def Dublikates(df: pd.DataFrame) -> pd.DataFrame:        
    # Проверка и удаление столбца flat_id, если он существует
    if 'flat_id' in df.columns:
        df = df.drop(columns=['flat_id'])
        logger.info("🔍 Столбец 'flat_id' был удалён перед удалением дубликатов.")
    
    initial_len = len(df)
    df = df.drop_duplicates()
    # Логирование количества удалённых дубликатов
    logger.info("🔍 Удалено дубликатов: %d", initial_len - len(df))
    
    return df

def EmptyCells(df: pd.DataFrame) -> pd.DataFrame:
    """Заполнение пропущенных значений по логике."""
    missing = df.isna().sum()
    logger.info("📉 Пропущенные значения:\n%s", missing[missing > 0])
    df = df.fillna({
        'has_elevator': False,
        'kitchen_area': df['kitchen_area'].median(),
        'living_area': df['living_area'].median(),
        'ceiling_height': df['ceiling_height'].median(),
        'flats_count': df['flats_count'].median(),
        'floors_total': df['floors_total'].median(),
        'build_year': df['build_year'].mode().iloc[0] if not df['build_year'].mode().empty else None,
    })
    return df

def Vibros(df: pd.DataFrame) -> pd.DataFrame:
    e1 = 4
    """Обработка выбросов с использованием IQR."""
    for column in ['total_area', 'price', 'ceiling_height']:
        if column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - e1 * IQR
            upper_bound = Q3 + e1 * IQR
            before_len = len(df)
            
            # Фильтрация выбросов
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            removed_outliers_count = before_len - len(df)
            
            # Корректный logger.info с корректным количеством параметров
            logger.info("🔍 Удалено выбросов в столбце %s: %d", column, removed_outliers_count)
            logger.info("🔹 Границы для столбца %s: нижняя %f, верхняя %f", column, lower_bound, upper_bound)
    
    return df

def Vibros2(df: pd.DataFrame) -> pd.DataFrame:
    """Обработка выбросов с использованием 3 стандартных отклонений от среднего."""
    for column in ['total_area', 'price', 'ceiling_height']:
        if column in df.columns:
            std_dev = df[column].std()
            mean = df[column].mean()
            upper_limit = mean + 5 * std_dev
            lower_limit = mean - 3 * std_dev
            before_len = len(df)
            
            # Фильтрация выбросов
            df = df[(df[column] <= upper_limit) & (df[column] >= lower_limit)]
            removed_outliers_count = before_len - len(df)
            
            # Лог о количестве удаленных выбросов
            logger.info("🔍 Удалено выбросов в столбце Метод2 %s: %d", column, removed_outliers_count)
            logger.info("🔹 Границы для столбца %s: нижняя %f, верхняя %f", column, lower_limit, upper_limit)

    return df