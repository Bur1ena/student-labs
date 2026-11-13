import csv
import json
import time
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
MODEL_NAME = "qwen2.5-7b-instruct"

# 1. Загружаем факты о компании
facts_text = ""
with open("data/facts.csv", "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        facts_text += f"- {row['Утверждение']} (Источник: {row['Источник']})\n"

# 2. Загружаем вопросы
queries = []
with open("data/queries.csv", "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        queries.append(row)

# 3. Собираем ответы
results = []
print(f"Начинаем сбор ответов для {len(queries)} вопросов...")

for q in queries:
    print(f"Вопрос {q['query_id']}: {q['query_text']}")
    
    system_prompt = f"""Ты — ИИ-поисковая система. Отвечай на вопрос пользователя, используя ТОЛЬКО следующие проверенные факты о компании «Лаборатория Касперского»:
{facts_text}

Если информации в фактах недостаточно, скажи «Недостаточно информации».
В конце ответа укажи источник (URL), если он есть в фактах."""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q['query_text']}
            ],
            temperature=0.2,
            max_tokens=300
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"ERROR: {e}"
    
    # Автоматическая разметка (упрощенная)
    brand_mentioned = 1 if "Касперск" in answer else 0
    brand_cited = 1 if "kaspersky.ru" in answer else 0
    
    results.append({
        "query_id": q['query_id'],
        "intent": q['intent'],
        "system": "Qwen-Local",
        "run": 1,
        "status": "ok" if "ERROR" not in answer else "error",
        "brand_mentioned": brand_mentioned,
        "brand_cited": brand_cited,
        "claims_true": 0, # Заполним вручную
        "claims_false": 0,
        "claims_unknown": 0,
        "answer": answer
    })
    
    print(f"  Упоминание: {brand_mentioned}, Ссылка: {brand_cited}")
    time.sleep(1)

# 4. Сохраняем в CSV
with open("results/responses.csv", "w", encoding="utf-8-sig", newline="") as f:
    fieldnames = ["query_id", "intent", "system", "run", "status", "brand_mentioned", "brand_cited", "claims_true", "claims_false", "claims_unknown", "answer"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nГотово! Сохранено в results/responses.csv")