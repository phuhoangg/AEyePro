import os
import json
import numpy as np
import pandas as pd

from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent.parent/ "data"

def save_data(data, file_path):
    """Save data to JSON file at the specified file path"""
    try:
        def convert_to_json_serializable(obj):
            """Recursively convert NumPy types to JSON-serializable Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_json_serializable(item) for item in obj)
            return obj

        # Convert all data to JSON-serializable format
        json_data = convert_to_json_serializable(data)

        # Tạo thư mục cha nếu chưa tồn tại
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)

        # Lưu file với đường dẫn đã cung cấp
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=2)

    except Exception as e:
        print(f"❌ Error saving data: {e}")

def append_csv_row(row_dict, file_path, fieldnames=None):
    """Append a row (dict) to a CSV file. Nếu file chưa tồn tại sẽ tạo mới với header."""
    import os
    file_exists = os.path.isfile(file_path)
    df = pd.DataFrame([row_dict])
    if not file_exists and fieldnames:
        df = df.reindex(columns=fieldnames)
    df.to_csv(file_path, mode='a', header=not file_exists, index=False)

def read_csv(file_path):
    """Đọc toàn bộ file CSV thành DataFrame."""
    return pd.read_csv(file_path)

def delete_csv_file(file_path):
    """Xóa file CSV nếu tồn tại."""
    import os
    if os.path.isfile(file_path):
        os.remove(file_path)