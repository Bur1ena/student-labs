import time
import pandas as pd
from openai import OpenAI

# Подключение к LM Studio
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio"
)
MODEL_NAME = "llama-3.2-3b-instruct"

# Запрос, который мы будем тестировать (из методички)
USER_PROMPT = "Событие: в 03:17 зафиксировано 25 неудачных попыток входа в учётную запись admin с одного IP, после чего произошёл успешный вход. Проанализируй ситуацию."

# Три системные инструкции из методички
system_prompts = {
    "sys_1_базовый": "Отвечай на запрос пользователя.",
    "sys_2_структура": "Ты ассистент аналитика ИБ. Отвечай не более чем в 5 пунктах. Для каждого пункта укажи риск и рекомендуемое действие.",
    "sys_3_факты_гипотезы": "Ты ассистент аналитика ИБ. Не выдумывай недостающие данные. Явно отделяй факты от предположений. Ответ верни в формате: Факты / Гипотезы / Действия."
}

results = []

print("--- ЭТАП 1: Исследование системных промптов ---")
for sys_name, sys_text in system_prompts.items():
    print(f"Отправка запроса с промптом: {sys_name}...")
    start = time.perf_counter()
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": sys_text},
            {"role": "user", "content": USER_PROMPT}
        ],
        temperature=0.2 # Фиксируем температуру
    )
    
    latency = round(time.perf_counter() - start, 2)
    answer = response.choices[0].message.content
    
    results.append({
        "Эксперимент": "System Prompt",
        "Параметр": sys_name,
        "Повтор": 1,
        "Время (сек)": latency,
        "Ответ модели": answer
    })

print("\n--- ЭТАП 2: Исследование Temperature ---")
# Запрос на креативность (из методички)
CREATIVE_PROMPT = "Предложи пять названий сервиса для автоматического анализа журналов событий информационной безопасности. Для каждого названия дай пояснение в одном предложении."

# Тестируем 3 температуры по 2 повтора
temperatures = [0.1, 0.7, 1.2]

for temp in temperatures:
    for repeat in range(1, 3): # 2 повтора
        print(f"Запрос с temperature={temp}, повтор {repeat}...")
        start = time.perf_counter()
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Ты креативный помощник."},
                {"role": "user", "content": CREATIVE_PROMPT}
            ],
            temperature=temp
        )
        
        latency = round(time.perf_counter() - start, 2)
        answer = response.choices[0].message.content
        
        results.append({
            "Эксперимент": "Temperature",
            "Параметр": f"temp={temp}",
            "Повтор": repeat,
            "Время (сек)": latency,
            "Ответ модели": answer
        })

# Сохраняем всё в CSV
df = pd.DataFrame(results)
df.to_csv("results_lab2.csv", index=False, encoding='utf-8-sig')
print("\nГотово! Результаты сохранены в results_lab2.csv")