# Указываем базовый образ
FROM apache/airflow:2.7.3-python3.10

# Копируем файл requirements.txt в контейнер
COPY requirements.txt /tmp/requirements.txt

# Обновляем pip и устанавливаем зависимости
RUN pip install -U pip && \
    pip install -r /tmp/requirements.txt

