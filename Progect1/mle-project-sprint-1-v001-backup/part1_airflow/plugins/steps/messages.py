from airflow.providers.telegram.hooks.telegram import TelegramHook
import os
import logging

from airflow.models import Variable

# TELEGRAM_BOT_TOKEN = Variable.get("TELEGRAM_BOT_TOKEN")
# TELEGRAM_CHAT_ID = Variable.get("TELEGRAM_CHAT_ID")

def send_telegram_success_message(context):
    print("🟢 CALLBACK SUCCESS TRIGGERED")
    """
    Send success notification to Telegram
    """
    # Получаем переменные из окружения
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Telegram credentials not set in environment variables!")

    hook = TelegramHook(
        token=TELEGRAM_BOT_TOKEN,
        chat_id=TELEGRAM_CHAT_ID
    )
    dag_id = context['dag'].dag_id
    run_id = context['run_id']
    message = f"✅ DAG {dag_id} с id={run_id} успешно выполнен!"
    hook.send_message({'chat_id': TELEGRAM_CHAT_ID, 'text': message})
    logger.info("🔹 Токкены %s: нижняя %f, верхняя %f", TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

def send_telegram_failure_message(context):
    print("🔴 CALLBACK FAILURE TRIGGERED")
    # Получаем переменные из окружения
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Telegram credentials not set in environment variables!")

    hook = TelegramHook(
        token=TELEGRAM_BOT_TOKEN,
        chat_id=TELEGRAM_CHAT_ID
    )
    dag_id = context['dag'].dag_id
    task_instance_key_str = context['task_instance_key_str']
    run_id = context['run_id']
    message = f"🔥 Ошибка в DAG {dag_id}, задача {task_instance_key_str}, run_id={run_id}!"
    hook.send_message({'chat_id': TELEGRAM_CHAT_ID, 'text': message})