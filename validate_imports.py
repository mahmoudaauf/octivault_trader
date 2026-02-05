import os
import ast
import sys

# Map of top-level local modules to expected directories
LOCAL_MODULES = {
    'core': 'core',
    'agents': 'agents',
    'utils': 'utils',
    'portfolio': 'portfolio',
}

def get_python_files(root_dir):
    py_files = []
    for root, dirs, files in os.walk(root_dir):
        if 'venv' in root or '.git' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files

def resolve_import_path(base_path, module_name):
    """
    Checks if a module path exists in the project.
    E.g. "core.shared_state" -> "./core/shared_state.py"
    """
    parts = module_name.split('.')
    if parts[0] not in LOCAL_MODULES:
        return True, "External/Stdlib" # Assume valid if not in our local map

    # Check mapping
    relative_path = os.path.join(*parts)
    
    # Check 1: File import (core/shared_state.py)
    file_path = os.path.join(base_path, relative_path + ".py")
    if os.path.exists(file_path):
        return True, file_path

    # Check 2: Package import (core/shared_state/__init__.py)
    pkg_path = os.path.join(base_path, relative_path, "__init__.py")
    if os.path.exists(pkg_path):
        return True, pkg_path
        
    # Check 3: Directory as namespace (Python 3.3+) - check if directory exists at least
    dir_path = os.path.join(base_path, relative_path)
    if os.path.isdir(dir_path):
        return True, dir_path

    return False, f"Missing: {relative_path}.py or dir"

def check_file_imports(file_path, project_root):
    errors = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=file_path)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    except Exception as e:
        return [f"ReadError: {e}"]

    for node in ast.walk(tree):
        module = None
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                valid, msg = resolve_import_path(project_root, module)
                if not valid:
                    errors.append(f"Line {node.lineno}: import {module} -> {msg}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module
                valid, msg = resolve_import_path(project_root, module)
                if not valid:
                    errors.append(f"Line {node.lineno}: from {module} ... -> {msg}")
    
    return errors

def main():
    root = os.getcwd()
    print(f"Scanning project root: {root}")
    files = get_python_files(root)
    print(f"Found {len(files)} Python files.")
    
    all_errors = {}
    
    for py_file in files:
        rel_name = os.path.relpath(py_file, root)
        errs = check_file_imports(py_file, root)
        if errs:
            all_errors[rel_name] = errs

    if not all_errors:
        print("\n✅ PASSED: All local imports resolve to existing files.")
        sys.exit(0)
    else:
        print(f"\n❌ FOUND ISSUES in {len(all_errors)} files:")
        for f, errs in all_errors.items():
            print(f"\n📄 {f}:")
            for e in errs:
                print(f"  - {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
