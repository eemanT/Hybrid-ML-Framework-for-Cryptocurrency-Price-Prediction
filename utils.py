import json
import os
from datetime import datetime

METADATA_FILE = "data/metadata.json"


def get_last_trained_date(crypto):
    if not os.path.exists(METADATA_FILE):
        return None
    with open(METADATA_FILE, 'r') as f:
        meta = json.load(f)
    date_str = meta.get(crypto)
    return datetime.strptime(date_str, "%Y-%m-%d") if date_str else None


def update_last_trained_date(crypto, date):
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            meta = json.load(f)
    else:
        meta = {}
    meta[crypto] = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)[:10]
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, 'w') as f:
        json.dump(meta, f, indent=2)
