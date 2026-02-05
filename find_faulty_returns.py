import os
import re

ROOT_DIR = '.'  # Replace with your code folder if needed

pattern = re.compile(r"^\s*return\s+(True|False)\s*(#.*)?$")  # Matches 'return True' or 'return False' alone

def scan_file(filepath):
    with open(filepath, 'r') as f:
        for i, line in enumerate(f, 1):
            if pattern.match(line):
                print(f"⚠️ Found in {filepath}:{i} → {line.strip()}")

def scan_dir(directory):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.py'):
                scan_file(os.path.join(root, filename))

scan_dir(ROOT_DIR)
