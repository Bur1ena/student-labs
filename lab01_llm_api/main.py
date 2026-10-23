import time
import pandas as pd
from openai import OpenAI

# Подключаемся к вашему локальному серверу LM Studio
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio" 
)

MODEL_NAME = "qwen2.5-7b-instruct"

prompts = [
    "Объясни в двух предложениях разницу между хэшированием и шифрованием.",
    "Определи тип события ИБ: 'После 15 неудачных попыток входа с одного IP выполнен успешный вход администратора'. Ответь одним словом: норма / подозрительно / критично.",
    "Напиши простую функцию на Python, которая складывает два числа.",
    "Суммаризируй текст в 3 пункта: 'Сотрудник забыл пароль. Он обратился в поддержку. Ему выдали временный пароль. При первом входе система попросила его сменить пароль.'",
    "Выдели из текста имя и возраст в формате JSON: 'Меня зовут Иван, мне 25 лет, я работаю в ИТ.'"
]

results = []

print(f"Отправка запросов к модели {MODEL_NAME}...\n")

for i, prompt in enumerate(prompts, start=1):
    print(f"[{i}/{len(prompts)}] Отправка запроса...")
    
    start_time = time.perf_counter()
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Ты полезный ассистент."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"ОШИБКА: {e}"
    
    end_time = time.perf_counter()
    latency = round(end_time - start_time, 2)
    
    results.append({
        "Модель": MODEL_NAME,
        "Номер запроса": i,
        "Текст запроса": prompt,
        "Ответ модели": answer,
        "Время (сек)": latency
    })
    
    print(f"Готово! Время: {latency} сек.\n")

df = pd.DataFrame(results)
df.to_csv("results_qwen.csv", index=False, encoding='utf-8-sig')

print("Все готово! Результаты сохранены в файл results_llama.csv")