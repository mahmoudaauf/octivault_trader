#!/usr/bin/env python3

import sys

if len(sys.argv) != 2:
    print("Usage: python3 fix_tabs.py <file>")
    sys.exit(1)

file_path = sys.argv[1]

with open(file_path, 'r') as f:
    content = f.read()

content = content.replace('\t', '    ')

with open(file_path, 'w') as f:
    f.write(content)

print("Replaced tabs with spaces in", file_path)