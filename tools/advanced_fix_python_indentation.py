import sys
import re

# Usage: python advanced_fix_python_indentation.py <input_file> <output_file>
# This script attempts to fix global indentation issues for top-level classes, functions, decorators, and code blocks.

def fix_indentation(lines):
    output = []
    block_stack = []  # Track (indent_level, block_type)
    prev_line = ""
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        # Fix decorators
        if re.match(r"^@", stripped):
            output.append(stripped)
            prev_line = stripped
            continue
        # Fix class/def/async def at module level
        if re.match(r"^(class|def|async def) ", stripped):
            output.append(stripped)
            block_stack = [(0, 'block')]
            prev_line = stripped
            continue
        # Fix block headers (if, except, for, while, with, try, else, elif)
        if re.match(r"^(if |except|for |while |with |try:|else:|elif )", stripped):
            # Indent block header if inside another block
            if block_stack:
                output.append(' ' * (block_stack[-1][0] + 4) + stripped)
                block_stack.append((block_stack[-1][0] + 4, 'block'))
            else:
                output.append(stripped)
                block_stack.append((0, 'block'))
            prev_line = stripped
            continue
        # Indent code inside a block
        if block_stack and stripped:
            # If previous line was a block header, indent this line
            if prev_line.endswith((':', '):')) and not re.match(r"^(class|def|async def) ", prev_line):
                output.append(' ' * (block_stack[-1][0] + 4) + stripped)
            else:
                output.append(' ' * block_stack[-1][0] + stripped)
            prev_line = stripped
            continue
        # Default: output as is
        output.append(stripped)
        prev_line = stripped
    return output

def main():
    if len(sys.argv) != 3:
        print("Usage: python advanced_fix_python_indentation.py <input_file> <output_file>")
        sys.exit(1)
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
    fixed = fix_indentation(lines)
    with open(sys.argv[2], 'w') as f:
        f.writelines(line + '\n' if not line.endswith('\n') else line for line in fixed)
    print(f"Advanced indentation fix written to {sys.argv[2]}")

if __name__ == "__main__":
    main()
