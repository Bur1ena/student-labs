import time
import pandas as pd
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
MODEL_NAME = "qwen2.5-7b-instruct" # Используем Qwen!

# Наш тестовый запрос (событие ИБ)
user_input = "Событие: в 03:17 зафиксировано 25 неудачных попыток входа в учётную запись admin с одного IP, после чего произошёл успешный вход. Проанализируй ситуацию."

prompts = {
    "v1_базовый": "Ты помощник. Ответь на вопрос пользователя.",
    
    "v2_структурированный": """Ты аналитик центра мониторинга ИБ. 
    Задача: Классифицируй событие и оцени риск.
    Ограничения: не придумывай отсутствующие факты; если данных недостаточно, перечисли недостающие.
    Формат ответа: 1. Категория. 2. Риск (low/medium/high/critical). 3. Доказательства. 4. Недостающие данные.""",
    
    "v3_few_shot_json": """Ты аналитик ИБ. Верни строго JSON.
    Пример 1:
    Вход: "5 неудачных входов, затем успешный" 
    Выход: {"category": "brute_force", "risk": "medium", "evidence": ["5 failed", "success"], "missing_data": []}
    
    Пример 2:
    Вход: "Вход с нового IP в 3 утра"
    Выход: {"category": "suspicious_login", "risk": "high", "evidence": ["new IP", "night time"], "missing_data": ["user location"]}
    
    Твой вход: """ + user_input
}

results = []
print("Начинаем тестирование промптов на модели Qwen 7B...\n")

for version, system_text in prompts.items():
    print(f"Тестируем {version}...")
    start = time.perf_counter()
    
    messages = [
        {"role": "system", "content": system_text if version != "v3_few_shot_json" else "Ты аналитик ИБ. Отвечай строго в формате JSON."},
        {"role": "user", "content": user_input if version != "v3_few_shot_json" else system_text.split("Твой вход: ")[1]}
    ]
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=500
    )
    
    latency = round(time.perf_counter() - start, 2)
    answer = response.choices[0].message.content
    
    results.append({
        "Версия промпта": version,
        "Время (сек)": latency,
        "Ответ модели": answer
    })

df = pd.DataFrame(results)
df.to_csv("results_lab3.csv", index=False, encoding='utf-8-sig')
print("\nГотово! Файл results_lab3.csv создан.")