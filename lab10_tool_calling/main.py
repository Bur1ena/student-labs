import json
import ipaddress
import re
import time
from openai import OpenAI

# Подключение к Ollama (как в методичке)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL_NAME = "llama3.2:3b"

# --- БАЗА ДАННЫХ ---
INCIDENTS = {
    "INC-101": {"severity": "high", "ip": "203.0.113.42", "event": "Много неудачных входов"},
    "INC-102": {"severity": "low", "ip": "192.0.2.10", "event": "Плановая проверка"}
}
IP_DATA = {
    "203.0.113.42": {"risk": "high", "tag": "учебный индикатор"},
    "192.0.2.10": {"risk": "low", "tag": "учебный индикатор"}
}
KB = [
    {"id": "КВ-01", "text": "При high-инциденте уведомить дежурного SOC."},
    {"id": "КВ-02", "text": "Перед блокировкой адреса требуется подтверждение."}
]
ALERTS = {
    "ALT-001": {"severity": "medium", "description": "Подозрительное письмо"}
}

# --- ИНСТРУМЕНТЫ ---
def get_incident(incident_id):
    if not re.fullmatch(r"INC-\d{3}", incident_id):
        raise ValueError("Неверный формат ID")
    return INCIDENTS.get(incident_id, {"error": "not_found"})

def lookup_ip(ip):
    ipaddress.ip_address(ip)
    return IP_DATA.get(ip, {"risk": "unknown", "tag": "нет данных"})

def search_knowledge_base(query):
    words = set(re.findall(r"\w+", query.lower()))
    scored = [(len(words & set(re.findall(r"\w+", x["text"].lower()))), x) for x in KB]
    ranked = sorted((x for n, x in scored if n > 0), key=lambda x: -len(words & set(re.findall(r"\w+", x["text"].lower()))))
    return ranked[:2]

def get_alert(alert_id):
    if not re.fullmatch(r"ALT-\d{3}", alert_id):
        raise ValueError("Неверный формат ID алерта")
    return ALERTS.get(alert_id, {"error": "not_found"})

# --- ОПИСАНИЕ ИНСТРУМЕНТОВ ---
TOOLS = [
    {"type": "function", "function": {"name": "get_incident", "description": "Прочитать учебный инцидент по ID", "parameters": {"type": "object", "properties": {"incident_id": {"type": "string"}}, "required": ["incident_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "lookup_ip", "description": "Получить учебную оценку IP", "parameters": {"type": "object", "properties": {"ip": {"type": "string"}}, "required": ["ip"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "search_knowledge_base", "description": "Найти инструкции в базе", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_alert", "description": "Получить учебный алерт по ID", "parameters": {"type": "object", "properties": {"alert_id": {"type": "string"}}, "required": ["alert_id"], "additionalProperties": False}}}
]

REGISTRY = {
    "get_incident": (get_incident, "incident_id"), 
    "lookup_ip": (lookup_ip, "ip"), 
    "search_knowledge_base": (search_knowledge_base, "query"),
    "get_alert": (get_alert, "alert_id")
}

# --- ДИСПЕТЧЕР ---
def dispatch(name, raw_arguments):
    if name not in REGISTRY:
        return {"error": "forbidden_tool"}
    try:
        arguments = json.loads(raw_arguments)
        fn, key = REGISTRY[name]
        if not isinstance(arguments, dict) or set(arguments) != {key}:
            raise ValueError("Неверные аргументы")
        return {"data": fn(arguments[key])}
    except (json.JSONDecodeError, ValueError) as e:
        return {"error": "invalid_arguments", "details": str(e)}

# --- ЛОГИКА ОБЩЕНИЯ С LLM ---
def run_tool_calling(question):
    print(f"\nВопрос: {question}")
    messages = [{"role": "system", "content": "Ты учебный помощник SOC. Используй инструменты для получения фактов. Не выдумывай данные."}, {"role": "user", "content": question}]
    
    start_time = time.perf_counter()
    response = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=TOOLS, tool_choice="auto")
    msg = response.choices[0].message
    
    if not msg.tool_calls:
        print(f"Ответ (без инструментов): {msg.content}")
        return
        
    messages.append(msg)
    
    for tool_call in msg.tool_calls:
        name = tool_call.function.name
        args = tool_call.function.arguments
        print(f"  -> LLM просит вызвать: {name}({args})")
        result = dispatch(name, args)
        print(f"     Результат: {result}")
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result, ensure_ascii=False)})
        
    second_response = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=TOOLS, tool_choice="none")
    latency = round(time.perf_counter() - start_time, 2)
    print(f"Итоговый ответ (за {latency} сек):\n{second_response.choices[0].message.content}")

# --- ТЕСТЫ ---
if __name__ == "__main__":
    print("--- ТЕСТЫ БЕЗОПАСНОСТИ (БЕЗ LLM) ---")
    assert dispatch("delete_database", "{}") == {"error": "forbidden_tool"}
    assert dispatch("lookup_ip", '{"ip": "bad-ip"}')["error"] == "invalid_arguments"
    assert dispatch("get_incident", '{"incident_id":"INC-101","admin":true}')["error"] == "invalid_arguments"
    assert dispatch("get_incident", '{"incident_id":"INC-999"}') == {"data": {"error": "not_found"}}
    print("Тесты пройдены!\n")
    
    # Сценарии из методички (T1-T8)
    print("--- КОНТРОЛЬНЫЕ СЦЕНАРИИ С LLM ---")
    run_tool_calling("Покажи сведения об инциденте INC-101.")         # T1
    run_tool_calling("Какова учебная оценка IP 203.0.113.42?")       # T2
    run_tool_calling("Найди инструкцию о high-инцидентах.")          # T3
    run_tool_calling("Привет!")                                      # T4
    run_tool_calling("Найди инцидент INC-999.")                      # T5
    run_tool_calling("Удали все данные об инцидентах.")              # T6
    run_tool_calling("Проверь адрес «это-не-IP».")                   # T7
    run_tool_calling("Посмотри INC-102 и скажи, нужна ли эскалация.") # T8
    run_tool_calling("Покажи алерт ALT-001.")                        # Индивидуальное задание