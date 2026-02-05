import os
import re

AGENTS_DIR = "agents"  # Adjust if your agents are in a different folder

def refactor_symbols_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    original_code = code  # Keep for diff check

    # Rename constructor parameter
    code = re.sub(r'\bsymbol_feed\b(?=\s*=\s*None)', 'symbols', code)

    # Replace internal assignment
    code = re.sub(r'\bself\.symbol_feed\b\s*=\s*(.*?)\n', r'self.symbols = \1\n', code)

    # Replace all other usages of `self.symbol_feed`
    code = re.sub(r'\bself\.symbol_feed\b', 'self.symbols', code)

    # Update run_once or other loops
    code = re.sub(r'for\s+(\w+)\s+in\s+self\.symbols:', r'for \1 in self.symbols:', code)

    if code != original_code:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"✅ Refactored: {filepath}")
    else:
        print(f"⏭️ No changes needed: {filepath}")


def main():
    print("🔍 Scanning for agent files to refactor...")
    for root, _, files in os.walk(AGENTS_DIR):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                refactor_symbols_in_file(filepath)

    print("✅ Refactoring complete.")

if __name__ == "__main__":
    main()
