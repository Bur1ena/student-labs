import csv
from collections import defaultdict
from pathlib import Path

REQUIRED = {"query_id", "intent", "system", "run", "status", "brand_mentioned", "brand_cited", "claims_true", "claims_false", "claims_unknown"}

def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or REQUIRED - set(reader.fieldnames):
            raise ValueError("Нет столбцов: " + ", ".join(sorted(REQUIRED - set(reader.fieldnames or []))))
        return list(reader)

def safe_pct(numerator, denominator):
    return round(100 * numerator / denominator, 1) if denominator else "н/д"

def calculate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["system"], row["intent"])].append(row)
        groups[(row["system"], "ИТОГО")].append(row)
        
    result = []
    for (system, intent), block in sorted(groups.items()):
        ok = [r for r in block if r["status"] == "ok"]
        counts = {c: sum(int(r[c]) for r in ok) for c in ("claims_true", "claims_false", "claims_unknown")}
        inspected = sum(counts.values())
        
        result.append({
            "system": system, "intent": intent,
            "attempts": len(block), "answered": len(ok),
            "completion_%": safe_pct(len(ok), len(block)),
            "mention_%": safe_pct(sum(int(r["brand_mentioned"]) for r in ok), len(ok)),
            "own_citation_%": safe_pct(sum(int(r["brand_cited"]) for r in ok), len(ok)),
            "verified_claim_%": safe_pct(counts["claims_true"], inspected),
            "false_claim_%": safe_pct(counts["claims_false"], inspected),
        })
    return result

if __name__ == "__main__":
    rows = read_rows("results/responses.csv")
    summary = calculate(rows)
    with open("results/summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
        
    for row in summary:
        if row["intent"] == "ИТОГО":
            print(row)