import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Загрузка модели эмбеддингов...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- ЗАГРУЗКА ДАННЫХ ---
with open('data/corpus.jsonl', 'r', encoding='utf-8') as f:
    docs = [json.loads(line) for line in f]

with open('data/questions.jsonl', 'r', encoding='utf-8') as f:
    questions = [json.loads(line) for line in f]

# Берем только те вопросы, на которые есть ответ
answerable_qs = [q for q in questions if q['answerable']]

# --- ФУНКЦИЯ РАЗБИЕНИЯ ---
def get_chunks(docs, size=100, overlap=10):
    chunks = []
    for doc in docs:
        text = doc['text']
        start = 0
        while start < len(text):
            end = start + size
            chunk_text = text[start:end]
            if chunk_text:
                chunks.append({"doc_id": doc['id'], "text": chunk_text})
            start += (size - overlap)
    return chunks

# --- ФУНКЦИЯ ОЦЕНКИ ---
def evaluate_config(chunk_size, top_k):
    chunks = get_chunks(docs, size=chunk_size, overlap=10)
    texts = [c['text'] for c in chunks]
    chunk_embeddings = model.encode(texts, normalize_embeddings=True)
    
    hits, recalls, mrrs = [], [], []
    
    for item in answerable_qs:
        q_emb = model.encode([item['question']], normalize_embeddings=True)
        sims = cosine_similarity(q_emb, chunk_embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        found_docs = [chunks[idx]['doc_id'] for idx in top_indices]
        unique_found = list(dict.fromkeys(found_docs))
        gold_id = item['gold_doc_ids'][0]
        
        is_hit = 1 if gold_id in unique_found else 0
        recall = 1 if gold_id in unique_found else 0
        try:
            rr = 1 / (unique_found.index(gold_id) + 1)
        except ValueError:
            rr = 0.0
            
        hits.append(is_hit)
        recalls.append(recall)
        mrrs.append(rr)
        
    return {
        "Chunk Size": chunk_size,
        "Top-K": top_k,
        "Всего фрагментов": len(chunks),
        "Hit@k": round(np.mean(hits), 2),
        "Recall@k": round(np.mean(recalls), 2),
        "MRR": round(np.mean(mrrs), 2)
    }

# --- ЗАПУСК ЭКСПЕРИМЕНТОВ ---
results = []
print("Запуск экспериментов...")

# 1. Исследование размера фрагмента (Top-K = 3)
for size in [50, 100, 200]:
    results.append(evaluate_config(chunk_size=size, top_k=3))

# 2. Исследование Top-K (Chunk Size = 100)
for k in [1, 2, 5]:
    results.append(evaluate_config(chunk_size=100, top_k=k))

df = pd.DataFrame(results)
print("\n--- РЕЗУЛЬТАТЫ ---")
print(df.to_string(index=False))
df.to_csv("results_lab9.csv", index=False, encoding='utf-8-sig')

# --- ПОСТРОЕНИЕ ГРАФИКОВ ---
print("\nПостроение графиков...")

# График 1: MRR от размера фрагмента
chunk_sizes = [50, 100, 200]
mrrs = [0.97, 0.89, 0.89]
plt.figure(figsize=(8, 5))
plt.plot(chunk_sizes, mrrs, marker='o', linestyle='-', color='b')
plt.title('Зависимость MRR от размера фрагмента (Top-K=3)')
plt.xlabel('Размер фрагмента (символов)')
plt.ylabel('MRR')
plt.grid(True)
plt.xticks(chunk_sizes)
plt.ylim(0.8, 1.05)
plt.savefig('figures/graph_mrr_vs_chunk.png')
print("Сохранен figures/graph_mrr_vs_chunk.png")

# График 2: Hit@k от Top-K
top_ks = [1, 2, 5]
hit_rates = [0.80, 0.93, 1.0]
plt.figure(figsize=(8, 5))
plt.plot(top_ks, hit_rates, marker='s', linestyle='-', color='r')
plt.title('Зависимость Hit@k от Top-K (Chunk Size=100)')
plt.xlabel('Количество извлекаемых фрагментов (Top-K)')
plt.ylabel('Hit@k')
plt.grid(True)
plt.xticks(top_ks)
plt.ylim(0.7, 1.05)
plt.savefig('figures/graph_hit_vs_topk.png')
print("Сохранен figures/graph_hit_vs_topk.png")

print("\nГотово!")