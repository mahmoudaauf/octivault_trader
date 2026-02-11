import sys
import re

# Usage: python fix_python_indentation.py <input_file> <output_file>
# This script will attempt to fix global indentation issues for top-level classes, functions, and decorators.

def fix_indentation(lines):
    output = []
    prev_was_decorator = False
    for i, line in enumerate(lines):
        # Fix decorators
        if re.match(r"^\s+@", line):
            output.append(line.lstrip())
            prev_was_decorator = True
            continue
        # Fix class and def at module level
        if re.match(r"^\s*(class|def) ", line):
            output.append(line.lstrip())
            prev_was_decorator = False
            continue
        # Fix async def at module level
        if re.match(r"^\s*async def ", line):
            output.append(line.lstrip())
            prev_was_decorator = False
            continue
        # If previous was decorator, dedent this line
        if prev_was_decorator and line.startswith("    "):
            output.append(line[4:])
            prev_was_decorator = False
            continue
        output.append(line)
    return output

def main():
    if len(sys.argv) != 3:
        print("Usage: python fix_python_indentation.py <input_file> <output_file>")
        sys.exit(1)
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
    fixed = fix_indentation(lines)
    with open(sys.argv[2], 'w') as f:
        f.writelines(fixed)
    print(f"Fixed global indentation written to {sys.argv[2]}")

if __name__ == "__main__":
    main()
