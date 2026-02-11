import sys
import re

# Usage: python smart_python_indentation_fixer.py <input_file> <output_file>
# This script attempts to fix indentation for top-level classes, functions, decorators, and code blocks,
# while preserving logic as much as possible. It will not insert 'pass' after docstrings, but will warn about them.

def is_block_header(line):
    return bool(re.match(r"^(class |def |async def |if |elif |else:|try:|except|finally:|for |while |with )", line.strip()))

def fix_indentation(lines):
    output = []
    indent_level = 0
    block_stack = []
    prev_line = ""
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        # Decorators: align with next class/def
        if re.match(r"^@", stripped):
            output.append(' ' * indent_level + stripped)
            prev_line = stripped
            continue
        # Top-level class/def/async def
        if re.match(r"^(class |def |async def )", stripped):
            indent_level = 0
            output.append(stripped)
            block_stack.append(4)
            indent_level = 4
            prev_line = stripped
            continue
        # Block headers (if, elif, else, try, except, finally, for, while, with)
        if re.match(r"^(if |elif |else:|try:|except|finally:|for |while |with )", stripped):
            output.append(' ' * indent_level + stripped)
            block_stack.append(indent_level + 4)
            indent_level += 4
            prev_line = stripped
            continue
        # Dedent on blank line or end of block
        if not stripped:
            output.append(stripped)
            # Dedent if previous line was a block header
            if block_stack and prev_line.endswith(":"):
                indent_level = block_stack[-1] - 4
                if indent_level < 0:
                    indent_level = 0
                block_stack = block_stack[:-1]
            prev_line = stripped
            continue
        # Dedent for return, break, continue, raise at end of block
        if re.match(r"^(return|break|continue|raise|pass)", stripped):
            output.append(' ' * indent_level + stripped)
            prev_line = stripped
            continue
        # Default: indent according to current block
        output.append(' ' * indent_level + stripped)
        prev_line = stripped
    return output

def main():
    if len(sys.argv) != 3:
        print("Usage: python smart_python_indentation_fixer.py <input_file> <output_file>")
        sys.exit(1)
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
    fixed = fix_indentation(lines)
    with open(sys.argv[2], 'w') as f:
        f.writelines(line if line.endswith('\n') else line + '\n' for line in fixed)
    print(f"Smart indentation fix written to {sys.argv[2]}")

if __name__ == "__main__":
    main()
