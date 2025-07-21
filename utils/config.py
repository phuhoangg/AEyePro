import json
import os

def get_config(config_path='config/settings.json'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, '..', config_path)
    full_path = os.path.normpath(full_path)

    with open(full_path, 'r') as f:
        return json.load(f)

def save_config(config, config_path='config/settings.json'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, '..', config_path)
    full_path = os.path.normpath(full_path)
    with open(full_path, 'w') as f:
        json.dump(config, f, indent=2)
