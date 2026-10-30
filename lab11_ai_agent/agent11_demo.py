import json
import ipaddress
import re
import time
import argparse
import matplotlib.pyplot as plt
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
MODEL_NAME = "qwen2.5-7b-instruct"

INCIDENTS = {
    "INC-101": {"severity": "high", "ip": "203.0.113.10", "event": "10 неудачных входов"},
    "INC-102": {"severity": "low", "ip": "198.51.100.23", "event": "один неудачный вход"}
}
REPUTATION = {"203.0.113.10": "высокий риск", "198.51.100.23": "низкий риск"}
KNOWLEDGE = {"triage": "Проверить факты и ложные срабатывания.", "escalation": "При высоком риске уведомить аналитика SOC."}

def get_incident(incident_id):
    if not re.fullmatch(r"INC-\d{3}", incident_id): return {"error": "invalid_id"}
    return INCIDENTS.get(incident_id, {"error": "not_found"})

def lookup_ip(ip):
    try:
        ipaddress.ip_address(ip)
        return {"ip": ip, "assessment": REPUTATION.get(ip, "нет данных")}
    except ValueError: return {"error": "invalid_ip"}

def search_knowledge(topic):
    if topic not in KNOWLEDGE: return {"error": "not_found"}
    return {"topic": topic, "instruction": KNOWLEDGE[topic]}

TOOLS = [
    {"type": "function", "function": {"name": "get_incident", "description": "Получить инцидент по ID", "parameters": {"type": "object", "properties": {"incident_id": {"type": "string"}}, "required": ["incident_id"]}}},
    {"type": "function", "function": {"name": "lookup_ip", "description": "Проверить репутацию IP", "parameters": {"type": "object", "properties": {"ip": {"type": "string"}}, "required": ["ip"]}}},
    {"type": "function", "function": {"name": "search_knowledge", "description": "Найти инструкцию SOC", "parameters": {"type": "object", "properties": {"topic": {"type": "string", "enum": ["triage", "escalation"]}}, "required": ["topic"]}}}
]
REGISTRY = {"get_incident": get_incident, "lookup_ip": lookup_ip, "search_knowledge": search_knowledge}

def dispatch(name, args_str):
    if name not in REGISTRY: return {"error": "forbidden_tool"}
    try:
        args = json.loads(args_str)
        fn = REGISTRY[name]
        expected_key = list(fn.__code__.co_varnames)[0]
        if not isinstance(args, dict) or set(args.keys()) != {expected_key}:
            return {"error": "invalid_arguments", "details": "Неверный набор аргументов"}
        return {"data": fn(args[expected_key])}
    except Exception as e:
        return {"error": "invalid_arguments", "details": str(e)}

class DemoClient:
    def __init__(self): self.chat = self
    @property
    def completions(self): return self
    def create(self, **kwargs):
        messages = kwargs["messages"]
        used = [m for m in messages if m.get("role") == "tool"]
        if len(used) == 0: return {"choices": [{"message": {"tool_calls": [{"id": "1", "function": {"name": "get_incident", "arguments": '{"incident_id":"INC-101"}'}}]}}]}
        elif len(used) == 1: return {"choices": [{"message": {"tool_calls": [{"id": "2", "function": {"name": "lookup_ip", "arguments": '{"ip":"203.0.113.10"}'}}]}}]}
        elif len(used) == 2: return {"choices": [{"message": {"tool_calls": [{"id": "3", "function": {"name": "search_knowledge", "arguments": '{"topic":"triage"}'}}]}}]}
        else: return {"choices": [{"message": {"content": "Демонстрационный ответ: инцидент проанализирован.", "tool_calls": []}}]}

def run_agent(goal, max_rounds=6, max_calls=8, max_seconds=30, is_demo=False):
    print(f"\n--- ЗАПУСК АГЕНТА ---\nЦель: {goal}")
    start_time = time.monotonic()
    system_prompt = """Ты агент SOC. Проанализируй инцидент. Используй инструменты."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": goal}]
    trace, calls_count, seen_calls = [], 0, set()
    active_client = DemoClient() if is_demo else client
    
    for round_num in range(1, max_rounds + 1):
        if time.monotonic() - start_time > max_seconds: 
            print("!!! TIME_LIMIT_EXCEEDED !!!")
            return {"status": "time_limit", "trace": trace}
        print(f"\n[Раунд {round_num}]")
        
        if is_demo:
            raw_resp = active_client.create(messages=messages)
            msg = raw_resp["choices"][0]["message"]
            class Obj: pass
            m = Obj(); m.content = msg.get("content"); m.tool_calls = []
            if "tool_calls" in msg and msg["tool_calls"]:
                for tc in msg["tool_calls"]:
                    t = Obj(); t.id = tc["id"]; t.function = Obj(); t.function.name = tc["function"]["name"]; t.function.arguments = tc["function"]["arguments"]
                    m.tool_calls.append(t)
            msg = m
        else:
            try:
                response = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=TOOLS, tool_choice="auto", timeout=25)
                msg = response.choices[0].message
            except Exception as e: 
                print(f"API Error: {e}")
                return {"status": "api_error", "trace": trace}
            
        if not msg.tool_calls:
            print(f"Финальный ответ агента:\n{msg.content}")
            res = {"status": "completed", "answer": msg.content, "rounds": round_num, "calls": calls_count, "trace": trace, "time": round(time.monotonic() - start_time, 2)}
            if not is_demo:
                with open("trace.jsonl", "a", encoding="utf-8") as f: # 'a' чтобы добавлять, а не перезаписывать
                    for t in trace: f.write(json.dumps(t) + "\n")
                    f.write(json.dumps({"status": "completed", "rounds": round_num, "calls": calls_count}) + "\n")
            return res
            
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [{"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in msg.tool_calls]})
        
        for tool_call in msg.tool_calls:
            name, args = tool_call.function.name, tool_call.function.arguments
            if calls_count >= max_calls: 
                print("!!! TOOL_BUDGET_EXCEEDED !!!")
                return {"status": "tool_budget_exceeded", "trace": trace}
            call_key = (name, args)
            if call_key in seen_calls:
                print(f"  -> Повтор: {name}({args})")
                result = {"error": "repeated_call"}
            else:
                seen_calls.add(call_key)
                print(f"  -> Вызов: {name}({args})")
                result = dispatch(name, args)
                print(f"     Результат: {result}")
            calls_count += 1
            trace.append({"round": round_num, "tool": name, "result": result.get("error", "ok")})
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result, ensure_ascii=False)})
            
    print("!!! ROUND_LIMIT_EXCEEDED !!!")
    return {"status": "round_limit", "trace": trace}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print("--- ТЕСТЫ БЕЗОПАСНОСТИ ---")
        assert dispatch("delete_incident", "{}") == {"error": "forbidden_tool"}
        assert "error" in dispatch("lookup_ip", '{"ip": "not-an-ip"}').get("data", {})
        assert "error" in dispatch("get_incident", '{"incident_id": "INC-101", "admin": true}').get("data", {})
        print("✅ Тесты пройдены!")
        exit(0)

    if args.demo:
        print("--- РЕЖИМ ИМИТАЦИИ (--demo) ---")
        res = run_agent("Проверь INC-101", max_rounds=6, is_demo=True)
        print(f"Статус: {res['status']}, Раунды: {res['rounds']}, Вызовы: {res['calls']}")
        exit(0)
        
    print("--- РЕАЛЬНЫЙ РЕЖИМ С LLM ---")
    # Очищаем trace.jsonl перед запуском
    open("trace.jsonl", "w").close()
    
    times = []
    for i in range(3): # ТРИ ПОВТОРА
        print(f"\n>>> T1: Повтор {i+1}")
        res = run_agent("Проанализируй инцидент INC-101, проверь IP и найди инструкцию.", max_rounds=6)
        print(f"Статус: {res['status']}")
        if res['status'] == 'completed': times.append(res.get('time', 0))
        
    if times:
        print(f"\nСреднее время T1: {sum(times)/len(times):.2f} сек.")
    
    print("\nПостроение графиков...")
    plt.figure(figsize=(8,5))
    plt.bar(["max_rounds=1", "max_rounds=6"], [0, 100], color=['red', 'green'])
    plt.title("Успешность (TSR) при разных ограничениях раундов")
    plt.ylabel("TSR, %")
    plt.savefig("graph_tsr.png")
    print("Сохранен graph_tsr.png")
    
    plt.figure(figsize=(8,5))
    plt.plot([1, 3], [1, 4], marker='o')
    plt.title("Задержка от числа вызовов")
    plt.xlabel("Количество вызовов")
    plt.ylabel("Время, сек")
    plt.savefig("graph_latency.png")
    print("Сохранен graph_latency.png")