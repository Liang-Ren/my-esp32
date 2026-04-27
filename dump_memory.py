import sys, sqlite3
sys.stdout.reconfigure(encoding="utf-8")
db = sqlite3.connect(r'c:\Users\liang\Copilot\.venv\xiaozhi\memory.db')
db.row_factory = sqlite3.Row

print("=== 短期记忆 (messages) ===")
rows = db.execute("SELECT device_id, role, content, ts FROM messages ORDER BY ts").fetchall()
if rows:
    for r in rows:
        print(f'[{r["ts"]}] {r["device_id"]} | {r["role"]}: {r["content"]}')
else:
    print("(空)")

print()
print("=== 长期记忆 (long_term) ===")
rows = db.execute("SELECT device_id, summary, facts, preferences, updated_at FROM device_memory").fetchall()
if rows:
    for r in rows:
        print(f'Device: {r["device_id"]}')
        print(f'  updated:     {r["updated_at"]}')
        print(f'  summary:     {r["summary"]}')
        print(f'  facts:       {r["facts"]}')
        print(f'  preferences: {r["preferences"]}')
else:
    print("(空)")

db.close()