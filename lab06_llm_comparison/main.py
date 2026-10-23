import time
import pandas as pd
from openai import OpenAI

# Подключаемся к LM Studio (он у нас будет играть роль "облака" и "локалки")
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

# Наши модели: Llama 3B (быстрая) и Qwen 7B (умная)
models = ["llama-3.2-3b-instruct", "qwen2.5-7b-instruct"]

prompt = "Напиши подробный план из 5 пунктов по реагированию на инцидент ИБ: утечка базы данных клиентов. Каждый пункт должен содержать не менее 2 предложений."

results = []

for model_name in models:
    print(f"\nТестируем модель: {model_name}...")
    start_time = time.perf_counter()
    ttft = None
    full_text = ""
    token_count = 0
    
    try:
        # Включаем stream=True для замера TTFT
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = round(time.perf_counter() - start_time, 2)
                full_text += chunk.choices[0].delta.content
                token_count += 1 # Примерное количество токенов
        
        total_time = round(time.perf_counter() - start_time, 2)
        
        # Примерная стоимость (для облака)
        # Допустим, входной токен стоит $0.001, выходной $0.002 (условно)
        cost = (50 * 0.001) + (token_count * 0.002) 
        
        results.append({
            "Модель": model_name,
            "TTFT (сек)": ttft,
            "Общее время (сек)": total_time,
            "Токенов": token_count,
            "Примерная стоимость ($)": round(cost, 4)
        })
        print(f"  Готово! TTFT: {ttft} сек, Всего: {total_time} сек.")
        
    except Exception as e:
        print(f"  Ошибка: {e}")

# Итоговая таблица
df = pd.DataFrame(results)
print("\n--- ИТОГОВАЯ ТАБЛИЦА ---")
print(df.to_string(index=False))
df.to_csv("results_lab6.csv", index=False, encoding='utf-8-sig')
print("\nСохранено в results_lab6.csv")