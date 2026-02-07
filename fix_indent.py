#!/usr/bin/env python3

import ast
import sys

def fix_indentation(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse the AST to ensure it's valid Python
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"Syntax error in file: {e}")
        return False

    # Use ast.unparse to reformat with proper indentation
    try:
        # Python 3.9+ has ast.unparse
        formatted_content = ast.unparse(tree)
    except AttributeError:
        # Fallback for older Python versions
        import astor
        formatted_content = astor.to_source(tree)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_content)

    print("File reformatted successfully")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_indent.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    success = fix_indentation(file_path)
    sys.exit(0 if success else 1)