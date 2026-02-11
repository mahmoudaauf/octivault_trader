import sys
import re

# Usage: python fix_class_decorator_indentation.py <input_file> <output_file>
# This script will fix misplaced decorators and class definitions to the leftmost column.

def fix_class_decorator_indentation(lines):
    output = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Fix decorators that are indented
        if re.match(r"^\s+@", line):
            output.append(line.lstrip())
            # If next line is a class/function, also move it
            if i + 1 < len(lines) and re.match(r"^\s+(class|def) ", lines[i + 1]):
                output.append(lines[i + 1].lstrip())
                i += 1
        # Fix class definitions that are indented
        elif re.match(r"^\s+class ", line):
            output.append(line.lstrip())
        else:
            output.append(line)
        i += 1
    return output

def main():
    if len(sys.argv) != 3:
        print("Usage: python fix_class_decorator_indentation.py <input_file> <output_file>")
        sys.exit(1)
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
    fixed = fix_class_decorator_indentation(lines)
    with open(sys.argv[2], 'w') as f:
        f.writelines(fixed)
    print(f"Fixed decorator/class indentation written to {sys.argv[2]}")

if __name__ == "__main__":
    main()
