import sys
import os
import importlib

print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"sys.path: {sys.path}")

try:
    print("\nAttempting to import dashboard_server...")
    import dashboard_server
    print("✅ SUCCESS: dashboard_server imported successfully.")
    
    if hasattr(dashboard_server, "DashboardServer"):
        print("✅ SUCCESS: DashboardServer class found.")
    else:
        print("❌ ERROR: DashboardServer class NOT found in the module.")
        print(f"Module members: {dir(dashboard_server)}")
        
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
except Exception as e:
    print(f"❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\nChecking dependencies...")
    import fastapi
    print("✅ fastapi: OK")
    import uvicorn
    print("✅ uvicorn: OK")
except ImportError as e:
    print(f"❌ MISSING DEPENDENCY: {e}")
