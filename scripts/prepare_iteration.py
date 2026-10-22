"""Automation helper for fine-tuning pipeline configurations and pushing to Kaggle."""

import json
import os
import sys
from pathlib import Path


def update_notebook_config(nb_path: str, params: dict, preset_name: str = "", score_axis: str = ""):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Update cell 8 (environment variables)
    cell8_lines = nb["cells"][8]["source"]
    new_cell8 = []
    for line in cell8_lines:
        matched = False
        if preset_name and line.startswith("BIOHUB_PRESET ="):
            new_cell8.append(f"BIOHUB_PRESET = '{preset_name}'\n")
            continue
        if score_axis and line.startswith("BIOHUB_SCORE_AXIS ="):
            new_cell8.append(f"BIOHUB_SCORE_AXIS = '{score_axis}'\n")
            continue
        for k, v in params.items():
            if line.startswith(f'os.environ["{k}"]'):
                new_cell8.append(f'os.environ["{k}"] = "{v}"\n')
                matched = True
                break
        if not matched:
            new_cell8.append(line)
    nb["cells"][8]["source"] = new_cell8

    # Update cell 10 (guard assertions)
    cell10_lines = nb["cells"][10]["source"]
    new_cell10 = []
    for line in cell10_lines:
        matched = False
        for k, v in params.items():
            if f'"{k}":' in line:
                indent = line[:line.find(f'"{k}"')]
                try:
                    num_val = float(v)
                    new_cell10.append(f'{indent}"{k}": {num_val},\n')
                except ValueError:
                    new_cell10.append(f'{indent}"{k}": "{v}",\n')
                matched = True
                break
        if not matched:
            new_cell10.append(line)
    nb["cells"][10]["source"] = new_cell10

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully updated {nb_path} with {len(params)} parameters.")


if __name__ == "__main__":
    test_params = {
        "BIOHUB_DET_THRESHOLD": "0.9625",
        "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.2",
        "BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": "0.28",
        "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": "0.18"
    }
    # Example dry run
    print("Helper loaded successfully.")
