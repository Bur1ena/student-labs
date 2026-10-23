import time
import pandas as pd
import json
import re
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Подключение к LM Studio
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
MODEL_NAME = "llama-3.2-3b-instruct" # Убедись, что название точное!

# Загружаем тестовые данные
try:
    df_tests = pd.read_csv("tests.csv")
except Exception as e:
    print(f"Ошибка чтения tests.csv: {e}")
    exit()

system_prompt = """Ты аналитик ИБ. Определи тип события.
Верни ответ СТРОГО в формате JSON: 
{"label": "normal" или "suspicious", "reason": "краткое объяснение"}
Не пиши ничего кроме JSON."""

results = []
print(f"Запуск тестов для модели: {MODEL_NAME}")
print(f"Всего тестов: {len(df_tests)}\n")

for index, row in df_tests.iterrows():
    test_id = row['id']
    prompt_text = row['input']
    expected = row['expected_label']
    
    print(f"[{test_id}/{len(df_tests)}] Отправка: {prompt_text[:40]}...")
    
    start = time.perf_counter()
    predicted = "error"
    raw_answer = ""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.1,
            max_tokens=150
        )
        raw_answer = response.choices[0].message.content
        
        # Ищем JSON в ответе (на случай, если модель добавила текст вокруг)
        json_match = re.search(r'\{.*\}', raw_answer, re.DOTALL)
        
        if json_match:
            parsed = json.loads(json_match.group(0))
            predicted = parsed.get("label", "parse_error").strip().lower()
        else:
            predicted = "parse_error"
            
    except Exception as e:
        print(f"  Ошибка: {e}")
        predicted = "api_error"
        raw_answer = str(e)
    
    latency = round(time.perf_counter() - start, 2)
    print(f"  Ответ: {predicted} (Ожидалось: {expected}) | Время: {latency} с")
    
    results.append({
        "id": test_id,
        "expected": expected,
        "predicted": predicted,
        "latency": latency,
        "raw_answer": raw_answer
    })

# Сохраняем сырые ответы
df_results = pd.DataFrame(results)
df_results.to_csv(f"raw_results_{MODEL_NAME}.csv", index=False)

# Считаем метрики (только для нормально распарсенных ответов)
valid_results = df_results[df_results['predicted'].isin(['normal', 'suspicious'])]
y_true = valid_results['expected']
y_pred = valid_results['predicted']

print("\n=========================================")
print("--- РЕЗУЛЬТАТЫ ОЦЕНКИ КАЧЕСТВА ---")
print(f"Всего тестов: {len(df_tests)}")
print(f"Успешно распарсено: {len(valid_results)}")
print(f"Ошибки формата (parse_error/api_error): {len(df_results) - len(valid_results)}")

if len(valid_results) > 0:
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.2f}")
    print(f"Precision: {precision_score(y_true, y_pred, pos_label='suspicious', zero_division=0):.2f}")
    print(f"Recall:    {recall_score(y_true, y_pred, pos_label='suspicious', zero_division=0):.2f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, pos_label='suspicious', zero_division=0):.2f}")
    print(f"Средняя задержка: {df_results['latency'].mean():.2f} сек.")
    
    print("\n--- CONFUSION MATRIX ---")
    cm = confusion_matrix(y_true, y_pred, labels=['normal', 'suspicious'])
    print(f"True Normal, Pred Normal: {cm[0][0]}")
    print(f"True Normal, Pred Suspicious: {cm[0][1]}")
    print(f"True Suspicious, Pred Normal: {cm[1][0]}")
    print(f"True Suspicious, Pred Suspicious: {cm[1][1]}")
else:
    print("Не удалось распарсить ни одного ответа!")

print("=========================================")