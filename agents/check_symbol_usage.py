import os
import re

AGENTS_DIR = "agents"  # Modify this if your agents are in a different folder

def find_symbol_feed_usages(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    symbol_feed_found = re.search(r'def __init__\(.*symbol_feed', content)
    symbols_found = re.search(r'def __init__\(.*symbols', content)

    if symbol_feed_found and not symbols_found:
        return True  # Still using symbol_feed, not yet converted
    return False

def main():
    print("🔍 Scanning for agent files still using `symbol_feed` instead of `symbols`...")
    broken_agents = []

    for root, _, files in os.walk(AGENTS_DIR):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if find_symbol_feed_usages(file_path):
                    broken_agents.append(file_path)

    if broken_agents:
        print("\n⚠️ The following agent files still use `symbol_feed`:")
        for path in broken_agents:
            print(f"  - {path}")
        print("\n🛠️ Please refactor them to use `symbols=` in the constructor.")
    else:
        print("✅ All agents are properly using `symbols=` in their constructors.")

if __name__ == "__main__":
    main()
