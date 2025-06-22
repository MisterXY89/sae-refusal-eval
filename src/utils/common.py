
import os
import json

import numpy as np
import pandas as pd


# Helper
def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):          # arrays → list
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    return obj

def convert_dfs_to_json_serializable(obj):
    """
    Recursively traverses a dictionary or list and converts any
    pandas DataFrame into a JSON serializable format (list of records).

    Args:
        obj: The dictionary, list, or other object to process.

    Returns:
        A new object of the same type with DataFrames converted.
    """
    # If the object is a pandas DataFrame, convert it to a list of dictionaries
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')

    # If the object is a dictionary, recursively process its values
    if isinstance(obj, dict):
        return {key: convert_dfs_to_json_serializable(value) for key, value in obj.items()}

    # If the object is a list, recursively process its items
    if isinstance(obj, list):
        return [convert_dfs_to_json_serializable(item) for item in obj]

    # If it's none of the above, return the object as is (it's likely already serializable)
    return obj