import sys
import re

# Usage: python fix_indentation.py <input_file> <output_file>
# This script will fix over-indented class and method blocks in a Python file.

def fix_indentation(lines):
    output = []
    in_class = False
    class_indent = None
    method_indent = None
    for i, line in enumerate(lines):
        # Detect class definition
        class_match = re.match(r'^(\s*)class (\w+)\s*\(?.*\)?:', line)
        if class_match:
            in_class = True
            class_indent = len(class_match.group(1))
            output.append(line.lstrip())
            continue
        # Detect end of class (by dedent)
        if in_class and line.strip() and not line.startswith(' ' * (class_indent + 4)) and not line.startswith(' ' * class_indent):
            in_class = False
            class_indent = None
        # If inside a class, fix indentation
        if in_class and line.strip():
            # Remove one extra indent level (4 spaces)
            fixed_line = line[class_indent + 4:] if line.startswith(' ' * (class_indent + 4)) else line.lstrip()
            output.append(' ' * 4 + fixed_line)
        else:
            output.append(line)
    return output

def main():
    if len(sys.argv) != 3:
        print("Usage: python fix_indentation.py <input_file> <output_file>")
        sys.exit(1)
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
    fixed = fix_indentation(lines)
    with open(sys.argv[2], 'w') as f:
        f.writelines(fixed)
    print(f"Fixed indentation written to {sys.argv[2]}")

if __name__ == "__main__":
    main()
