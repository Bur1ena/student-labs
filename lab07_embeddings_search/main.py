import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Загрузка модели эмбеддингов (это может занять минуту при первом запуске)...")
# Легкая мультиязычная модель, понимает русский
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Загружаем корпус документов
documents = []
with open('corpus.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        documents.append(json.loads(line))

texts = [doc['text'] for doc in documents]
print(f"Загружено {len(texts)} документов.")

# Получаем эмбеддинги для всех документов
print("Построение векторного индекса...")
doc_embeddings = model.encode(texts, normalize_embeddings=True)

# Наши поисковые запросы (не совпадают с текстами дословно!)
queries = [
    "что делать если украли пароль",
    "как защитить серверную комнату",
    "подозрительное письмо в почте",
    "восстановление данных после сбоя"
]

results_data = []

print("\n--- ВЫПОЛНЕНИЕ ПОИСКА ---")
for query in queries:
    print(f"\nЗапрос: '{query}'")
    # Получаем вектор запроса
    query_embedding = model.encode([query], normalize_embeddings=True)
    
    # Считаем косинусное сходство между запросом и всеми документами
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Сортируем по убыванию и берем топ-3
    top_indices = np.argsort(similarities)[::-1][:3]
    
    for rank, idx in enumerate(top_indices, start=1):
        score = round(float(similarities[idx]), 3)
        doc = documents[idx]
        print(f"  {rank}. [{doc['id']}] (score: {score}) - {doc['text']}")
        
        results_data.append({
            "Запрос": query,
            "Ранг": rank,
            "ID документа": doc['id'],
            "Score": score,
            "Текст документа": doc['text']
        })

# Сохраняем результаты
df = pd.DataFrame(results_data)
df.to_csv("results_lab7.csv", index=False, encoding='utf-8-sig')
print("\nГотово! Результаты сохранены в results_lab7.csv")