
import json
import os

def load_tuned_params(agent_name: str) -> dict:
    """
    Load tuned parameters from tuned_params/{agent_name}.json.
    Returns empty dict if file not found.
    """
    try:
        path = f"tuned_params/{agent_name}.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}
