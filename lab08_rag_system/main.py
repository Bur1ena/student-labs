import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# 1. Подключаемся к LM Studio (Qwen 7B)
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
MODEL_NAME = "qwen2.5-7b-instruct"

# 2. Загружаем модель эмбеддингов для поиска
print("Загрузка модели эмбеддингов...")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 3. Загружаем базу знаний
documents = []
with open('corpus.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        documents.append(json.loads(line))

texts = [doc['text'] for doc in documents]
doc_embeddings = embedder.encode(texts, normalize_embeddings=True)

# 4. Функция поиска (Retrieval)
def retrieve(query, k=2):
    query_emb = embedder.encode([query], normalize_embeddings=True)
    similarities = cosine_similarity(query_emb, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_indices:
        results.append({
            "id": documents[idx]['id'],
            "text": documents[idx]['text'],
            "score": round(float(similarities[idx]), 3)
        })
    return results

# 5. Функция формирования контекста
def build_context(results):
    blocks = []
    for i, item in enumerate(results, start=1):
        blocks.append(f"[S{i}] id={item['id']}\n{item['text']}")
    return "\n\n".join(blocks)

# 6. Системная инструкция для LLM
SYSTEM_PROMPT = """Ты отвечаешь по базе знаний. 
Используй только факты из блока CONTEXT. 
После каждого существенного утверждения указывай источник вида [S1].
Если данных для ответа нет, напиши: «В предоставленных источниках недостаточно информации».
Не придумывай факты, которых нет в контексте."""

# 7. Полный цикл RAG
def rag_answer(question):
    print(f"\nВопрос: {question}")
    
    # Шаг А: Поиск
    retrieved = retrieve(question, k=2)
    print("Найденные фрагменты:")
    for r in retrieved:
        print(f"  [{r['id']}] score={r['score']} - {r['text'][:50]}...")
    
    # Шаг Б: Формирование контекста
    context = build_context(retrieved)
    
    # Шаг В: Генерация ответа
    print("Генерация ответа...")
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"}
        ],
        temperature=0.1,
        max_tokens=400
    )
    latency = round(time.perf_counter() - start, 2)
    answer = response.choices[0].message.content
    
    print(f"Ответ (за {latency} сек):\n{answer}\n")
    
    return {
        "question": question,
        "retrieved_ids": [r['id'] for r in retrieved],
        "context": context,
        "answer": answer,
        "latency": latency
    }

# 8. Тестовые вопросы
questions = [
    "Что делать при компрометации учетной записи?", # Ответ есть в базе (d01)
    "Как часто нужно менять пароли?",              # Ответ есть в базе (d06)
    "Как настроить VPN-сервер?",                  # Ответа НЕТ в базе (проверка галлюцинаций)
    "Куда сообщать об инциденте ИБ?"              # Ответ есть в базе (d07)
]

all_results = []
for q in questions:
    res = rag_answer(q)
    all_results.append(res)

# Сохраняем в JSON
with open("results_lab8.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

print("Готово! Все результаты сохранены в results_lab8.json")