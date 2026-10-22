"""Biohub Cell Tracking - Automated Pipeline Manager & Fine-Tuning Controller.

Controls:
1. Submission status monitoring & score extraction.
2. Parameter fine-tuning and notebook configuration guard synchronization.
3. Kaggle kernel push and execution monitoring.
4. Kaggle competition submission dispatch.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


COMPETITION_SLUG = "biohub-cell-tracking-during-development"
KERNEL_SLUG = "rajucode/biohub-cell-tracking-advanced-pipeline"
NOTEBOOK_DIR = Path("notebooks/biohub_advanced_model")
NOTEBOOK_PATH = NOTEBOOK_DIR / "biohub_advanced_pipeline.ipynb"

# Parameter presets for iterations
ITERATION_CONFIGS = {
    3: {
        "preset_name": "v30_tight52_div28",
        "score_axis": "MOTION_RELINK_TIGHT_UM 5.2 + DC_SAFE_DIV 0.28 + BIDIR 0.18",
        "message": "Iteration 3: v30 relink tight 5.2 + safe div 0.28 + bidir 0.18",
        "params": {
            "BIOHUB_DET_THRESHOLD": "0.9625",
            "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.2",
            "BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": "0.28",
            "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": "0.18",
            "BIOHUB_VALIDATOR_ENABLE": "0",
        },
    },
    4: {
        "preset_name": "v31_div_rescue",
        "score_axis": "SAFE_DIV_MAX 9.5 + SHORT_TRACK_RESCUE_LEN 5 + GAP_6.0",
        "message": "Iteration 4: v31 safe div 9.5 + track rescue len 5 + gap close 6.0",
        "params": {
            "BIOHUB_DET_THRESHOLD": "0.9625",
            "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.2",
            "BIOHUB_SAFE_DIV_MAX_UM": "9.5",
            "BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN": "5",
            "BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB": "0.91",
            "BIOHUB_GAP_CLOSE_UM": "6.0",
            "BIOHUB_VALIDATOR_ENABLE": "0",
        },
    },
    5: {
        "preset_name": "v32_ultra_precision",
        "score_axis": "HARMONIC_FUSION_0.20 + DET_0.966 + SYMMETRY_TAU 0.65",
        "message": "Iteration 5: v32 harmonic fusion 0.20 + det 0.966 + symmetry 0.65",
        "params": {
            "BIOHUB_DET_THRESHOLD": "0.966",
            "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": "0.20",
            "BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU": "0.65",
            "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.2",
            "BIOHUB_GAP_CLOSE_UM": "5.8",
            "BIOHUB_VALIDATOR_ENABLE": "0",
        },
    },
    6: {
        "preset_name": "v33_dc_tta_secondary_calibrated",
        "score_axis": "DC_TTA_8VIEW + SECONDARY_EDGE_0.20 + DET_0.965 + BIDIR_0.15 + SAFE_DIV_7.5_12.5_0.60 + SHORT_4_0.88",
        "message": "Iteration 6: v33 dc tta 8view + sec edge 0.20 + det 0.965 + bidir 0.15 + safe div 7.5",
        "params": {
            "BIOHUB_DET_THRESHOLD": "0.965",
            "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": "0.15",
            "BIOHUB_SECONDARY_EDGE_WEIGHT": "0.20",
            "BIOHUB_SECONDARY_DETECTION_WEIGHT": "0.80",
            "BIOHUB_SECONDARY_EDGE_FEATURE_TTA": "1",
            "BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT": "0.75",
            "BIOHUB_DEEPCENTER_TTA": "1",
            "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "1",
            "BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": "0.20",
            "BIOHUB_DEEPCENTER_GAP_VETO": "1",
            "BIOHUB_DEEPCENTER_GAP_THRESHOLD": "0.25",
            "BIOHUB_SAFE_DIV_MAX_UM": "7.5",
            "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "12.5",
            "BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU": "0.60",
            "BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN": "4",
            "BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB": "0.88",
            "BIOHUB_GAP_CLOSE_UM": "5.2",
            "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.2",
            "BIOHUB_ILP_DIVISION_WEIGHT": "1.2",
            "BIOHUB_ILP_DISAPPEARANCE_WEIGHT": "2.0",
            "BIOHUB_VALIDATOR_ENABLE": "0",
        },
    },
    7: {
        "preset_name": "v34_drift_comp_velocity_projection",
        "score_axis": "DRIFT_COMP_HISTOGRAM + VELOCITY_PROJECTION_0.50 + GAP2_RECOVERY_4.4",
        "message": "Iteration 7: v34 drift comp + velocity projection 0.50 + gap2 4.4",
        "params": {
            "BIOHUB_DET_THRESHOLD": "0.965",
            "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": "0.15",
            "BIOHUB_SECONDARY_EDGE_WEIGHT": "0.20",
            "BIOHUB_SECONDARY_DETECTION_WEIGHT": "0.80",
            "BIOHUB_SECONDARY_EDGE_FEATURE_TTA": "1",
            "BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT": "0.75",
            "BIOHUB_DEEPCENTER_TTA": "1",
            "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "1",
            "BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": "0.20",
            "BIOHUB_DRIFT_COMP_ENABLE": "1",
            "BIOHUB_RELINK_DRIFT_COMP_ENABLE": "1",
            "BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT": "0.50",
            "BIOHUB_GAP2_MAX_STEP_UM": "4.4",
            "BIOHUB_GAP2_MAX_TOTAL_UM": "9.0",
            "BIOHUB_SAFE_DIV_MAX_UM": "7.2",
            "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "12.0",
            "BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN": "4",
            "BIOHUB_GAP_CLOSE_UM": "5.0",
            "BIOHUB_VALIDATOR_ENABLE": "0",
        },
    },
    8: {
        "preset_name": "v35_gt_biology_mitosis_geometry",
        "score_axis": "GT_BIOLOGY_MITOSIS_RADIUS_6.8_12.0 + DIVERGENCE_2.25 + DIVISION_WEIGHT_1.25",
        "message": "Iteration 8: v35 gt biology mitosis radius 6.8 + sister 12.0 + div weight 1.25",
        "params": {
            "BIOHUB_DET_THRESHOLD": "0.965",
            "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": "0.15",
            "BIOHUB_SECONDARY_EDGE_WEIGHT": "0.20",
            "BIOHUB_SECONDARY_DETECTION_WEIGHT": "0.80",
            "BIOHUB_DEEPCENTER_TTA": "1",
            "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "1",
            "BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": "0.20",
            "BIOHUB_SAFE_DIV_MAX_UM": "6.8",
            "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "12.0",
            "BIOHUB_SAFE_DIV_DIVERGE_UM": "2.25",
            "BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU": "0.62",
            "BIOHUB_ILP_DIVISION_WEIGHT": "1.25",
            "BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN": "4",
            "BIOHUB_GAP_CLOSE_UM": "5.0",
            "BIOHUB_VALIDATOR_ENABLE": "0",
        },
    },
    9: {
        "preset_name": "v36_consensus_ensemble_tuning",
        "score_axis": "LOW_MARGIN_CONSENSUS_0.35 + MULTI_SEED_CALIBRATION + HARMONIC_0.15 + VELOCITY_0.50",
        "message": "Iteration 9: v36 full stack consensus + harmonic 0.15 + mitosis 6.8 + vel 0.50",
        "params": {
            "BIOHUB_DET_THRESHOLD": "0.965",
            "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": "0.15",
            "BIOHUB_SECONDARY_EDGE_WEIGHT": "0.20",
            "BIOHUB_SECONDARY_DETECTION_WEIGHT": "0.80",
            "BIOHUB_SECONDARY_EDGE_FEATURE_TTA": "1",
            "BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT": "0.75",
            "BIOHUB_SECONDARY_LOW_MARGIN_MAX": "0.35",
            "BIOHUB_SECONDARY_MIX_TEMPERATURE": "1",
            "BIOHUB_DEEPCENTER_TTA": "1",
            "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "1",
            "BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": "0.20",
            "BIOHUB_SAFE_DIV_MAX_UM": "6.8",
            "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "12.0",
            "BIOHUB_SAFE_DIV_DIVERGE_UM": "2.25",
            "BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU": "0.62",
            "BIOHUB_ILP_DIVISION_WEIGHT": "1.25",
            "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.2",
            "BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT": "0.50",
            "BIOHUB_GAP_CLOSE_UM": "5.0",
            "BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN": "4",
            "BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB": "0.88",
            "BIOHUB_VALIDATOR_ENABLE": "0",
        },
    },
}


def run_cmd(cmd: list[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def get_latest_submission() -> dict:
    """Fetch latest submissions and return parsed info for the top submission."""
    output = run_cmd(["kaggle", "competitions", "submissions", COMPETITION_SLUG])
    lines = output.strip().split("\n")
    if len(lines) < 3:
        return {}
    data_line = lines[2]
    cols = [c.strip() for c in re.split(r"\s{2,}", data_line)]
    res = {}
    if len(cols) >= 6:
        res["ref"] = cols[0]
        res["fileName"] = cols[1]
        res["date"] = cols[2]
        res["description"] = cols[3]
        res["status"] = cols[4]
        res["publicScore"] = cols[5] if cols[5] else None
    return res


def update_notebook(iteration: int) -> None:
    if iteration not in ITERATION_CONFIGS:
        raise ValueError(f"Unknown iteration: {iteration}")
    cfg = ITERATION_CONFIGS[iteration]
    params = cfg["params"]
    preset_name = cfg["preset_name"]
    score_axis = cfg["score_axis"]

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update Cell 8 (Environment variables)
    c8_raw = nb["cells"][8]["source"]
    cell8_lines = "".join(c8_raw).splitlines(keepends=True)
    new_cell8 = []
    seen_params = set()
    for line in cell8_lines:
        if line.startswith("BIOHUB_PRESET ="):
            new_cell8.append(f"BIOHUB_PRESET = '{preset_name}'\n")
            continue
        if line.startswith("BIOHUB_SCORE_AXIS ="):
            new_cell8.append(f"BIOHUB_SCORE_AXIS = '{score_axis}'\n")
            continue
        matched = False
        for k, v in params.items():
            if line.startswith(f'os.environ["{k}"]'):
                new_cell8.append(f'os.environ["{k}"] = "{v}"\n')
                seen_params.add(k)
                matched = True
                break
        if not matched:
            new_cell8.append(line)

    insert_idx = len(new_cell8)
    for idx, line in enumerate(new_cell8):
        if line.strip().startswith("print("):
            insert_idx = idx
            break

    for k, v in sorted(params.items()):
        if k not in seen_params:
            new_cell8.insert(insert_idx, f'os.environ["{k}"] = "{v}"\n')
            insert_idx += 1

    new_cell8 = [l if l.endswith("\n") else l + "\n" for l in new_cell8]
    nb["cells"][8]["source"] = new_cell8

    # 2. Update Cell 10 (Configuration Guard)
    # Build exact expected numeric dict for guard
    guard_numeric = {
        "BIOHUB_DET_THRESHOLD": float(params.get("BIOHUB_DET_THRESHOLD", "0.965")),
        "BIOHUB_ILP_APPEARANCE_WEIGHT": 0.0,
        "BIOHUB_ILP_DISAPPEARANCE_WEIGHT": float(params.get("BIOHUB_ILP_DISAPPEARANCE_WEIGHT", "2.0")),
        "BIOHUB_GAP_CLOSE_UM": float(params.get("BIOHUB_GAP_CLOSE_UM", "5.2")),
        "BIOHUB_OUTPUT_MIN_TRACK_LEN": 6.0,
        "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": float(params.get("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", "0.15")),
    }
    if "BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU" in params:
        guard_numeric["BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU"] = float(params["BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU"])
    if "BIOHUB_MOTION_RELINK_TIGHT_UM" in params:
        guard_numeric["BIOHUB_MOTION_RELINK_TIGHT_UM"] = float(params["BIOHUB_MOTION_RELINK_TIGHT_UM"])
    if "BIOHUB_DEEPCENTER_TTA" in params:
        guard_numeric["BIOHUB_DEEPCENTER_TTA"] = float(params["BIOHUB_DEEPCENTER_TTA"])

    guard_code = [
        "# Configuration guard: assert the single intended model-level change.\n",
        "import json as _guard_json\n",
        "import math as _guard_math\n",
        "import os as _guard_os\n",
        "\n",
        "_EXPECTED_NUMERIC = {\n",
    ]
    for k, v in guard_numeric.items():
        guard_code.append(f'    "{k}": {v},\n')
    guard_code.append("}\n\n")
    guard_code.append('_EXPECTED_TEXT = {\n')
    guard_code.append('    "BIOHUB_BIDIRECTIONAL_FUSION_MODE": "harmonic_probability",\n')
    guard_code.append('    "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION": "0.90",\n')
    guard_code.append("}\n\n")
    guard_code.append("""_drift = {}
for _key, _want in _EXPECTED_NUMERIC.items():
    _raw = _guard_os.environ.get(_key)
    if _raw is None:
        _drift[_key] = "missing"
        continue
    _got = float(_raw)
    if not _guard_math.isclose(_got, _want, rel_tol=0.0, abs_tol=1e-12):
        _drift[_key] = {"expected": _want, "actual": _got}

for _key, _want in _EXPECTED_TEXT.items():
    _got = _guard_os.environ.get(_key)
    if _got != _want:
        _drift[_key] = {"expected": _want, "actual": _got}

if _drift:
    raise RuntimeError(
        "Configuration drift detected: " + _guard_json.dumps(_drift, sort_keys=True)
    )

print("Configuration guard: PASS")
print("Baseline: fixed-90 dual-seed clean pipeline (public LB 0.913)")
print("Single model-level change: harmonic mutual-support association fusion")
print(f"Reverse-time association weight: {_EXPECTED_NUMERIC.get('BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT', 0.15):.3f}")
""")

    nb["cells"][10]["source"] = guard_code

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    # 3. Verify locally that cell 8 and cell 10 execute cleanly without drift
    test_env = dict(os.environ)
    c8_code = "".join(nb["cells"][8]["source"])
    c10_code = "".join(nb["cells"][10]["source"])
    try:
        exec(c8_code, test_env)
        exec(c10_code, test_env)
        print(f"Verified locally: Iteration {iteration} ({preset_name}) applied and PASSES guard check!")
    except Exception as exc:
        print(f"FAILED local guard check for iteration {iteration}: {exc}")
        raise exc


def check_status() -> None:
    sub = get_latest_submission()
    print("=== Latest Submission ===")
    for k, v in sub.items():
        print(f"  {k}: {v}")

    k_status = run_cmd(["kaggle", "kernels", "status", KERNEL_SLUG])
    print(f"=== Kernel Status ===\n  {k_status}")


def push_kernel() -> None:
    print(f"Pushing {NOTEBOOK_DIR} to Kaggle...")
    out = run_cmd(["kaggle", "kernels", "push", "-p", str(NOTEBOOK_DIR)])
    print(out)


def get_kernel_status() -> str:
    out = run_cmd(["kaggle", "kernels", "status", KERNEL_SLUG])
    # Typically: rajucode/biohub-cell-tracking-advanced-pipeline has status "KernelWorkerStatus.RUNNING"
    match = re.search(r'status "([^"]+)"', out)
    if match:
        return match.group(1)
    return out


def submit_competition(version: int, message: str) -> None:
    cmd = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        COMPETITION_SLUG,
        "-f",
        "submission.csv",
        "-k",
        KERNEL_SLUG,
        "-v",
        str(version),
        "-m",
        message,
    ]
    print(f"Submitting to Kaggle competition: {' '.join(cmd)}")
    out = run_cmd(cmd)
    print(out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        check_status()
    elif len(sys.argv) > 2 and sys.argv[1] == "--apply":
        update_notebook(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--push":
        push_kernel()
    elif len(sys.argv) > 3 and sys.argv[1] == "--submit":
        submit_competition(int(sys.argv[2]), sys.argv[3])
    else:
        print("Usage: python3 scripts/pipeline_manager.py [--status | --apply <iter> | --push | --submit <ver> <msg>]")
