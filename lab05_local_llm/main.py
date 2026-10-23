import requests
import time
import json

URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

prompts = [
    "Объясни в 5 предложениях разницу между хэшированием и шифрованием.",
    "Определи категорию события: 'пять неуспешных входов, затем успешный вход с нового IP'. Ответь: норма / подозрительно / инцидент.",
    "Напиши функцию Python, проверяющую, содержит ли строка IPv4-адрес.",
    "Верни JSON с полями risk_level, reason, recommended_action для события: 'Скачивание 10 ГБ данных за 5 минут'.",
    "Суммаризируй в 3 пункта: 'Сотрудник забыл пароль. Обратился в поддержку. Ему выдали временный пароль. Система попросила его сменить.'"
]

results = []

print(f"Начинаем тестирование локальной модели {MODEL} через Ollama...\n")

for i, prompt in enumerate(prompts, start=1):
    print(f"[{i}/5] Отправка запроса...")
    
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    start_time = time.perf_counter()
    
    try:
        response = requests.post(URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "Ошибка: пустой ответ")
    except Exception as e:
        answer = f"ОШИБКА: {e}"
        
    end_time = time.perf_counter()
    latency = round(end_time - start_time, 2)
    
    print(f"    Готово! Время: {latency} сек.\n")
    
    results.append({
        "Номер": i,
        "Запрос": prompt,
        "Ответ": answer,
        "Время (сек)": latency
    })

with open("results_lab5.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("Все готово! Результаты сохранены в results_lab5.json")