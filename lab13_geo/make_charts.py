import matplotlib.pyplot as plt
import numpy as np

# Данные из summary.csv
categories = ['Брендовые', 'Информационные', 'Коммерческие', 'Сравнительные', 'Риски']
mention_rates = [100, 25, 100, 75, 50]
citation_rates = [100, 25, 75, 50, 25]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, mention_rates, width, label='Mention Rate (%)', color='skyblue')
bars2 = ax.bar(x + width/2, citation_rates, width, label='Own Citation Rate (%)', color='orange')

ax.set_xlabel('Категории запросов')
ax.set_ylabel('Процент (%)')
ax.set_title('GEO-аудит: Упоминание и цитирование по категориям')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig('results/charts/geo_audit_chart.png', dpi=150)
print("Диаграмма сохранена: results/charts/geo_audit_chart.png")