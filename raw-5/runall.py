import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

base_dir = Path(__file__).resolve().parent
master_output_dir = base_dir / "ablation_studies"
default_python = Path("/home/sofa/miniconda3/envs/phy/bin/python")

import pandas as pd

python_exec = os.getenv("PIPELINE_PYTHON", str(default_python if default_python.exists() else Path(sys.executable)))

experiments = [
    {"name": "01_Baseline_Full_Pipeline", "weight": "10.0", "threshold": "0.3", "augment": "True"},
    {"name": "02_No_Saliency_Weighting", "weight": "0.0", "threshold": "0.3", "augment": "True"},
    {"name": "03_Hyper_Saliency", "weight": "25.0", "threshold": "0.3", "augment": "True"},
    {"name": "04_Low_Threshold", "weight": "10.0", "threshold": "0.1", "augment": "True"},
    {"name": "05_No_Data_Augmentation", "weight": "10.0", "threshold": "0.3", "augment": "False"},
]

scripts = [
    "1_train_saliency_affine.py",
    "2_train_crossattention_grader.py",
    "3_visualize_pipeline.py",
]

master_output_dir.mkdir(parents=True, exist_ok=True)
master_results = []

print(f"Starting ablation study: {len(experiments)} runs scheduled.\n")

for exp in experiments:
    run_dir = master_output_dir / exp["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"INITIATING RUN: {exp['name']}")
    print(f"Output Directory: {run_dir}")
    print("=" * 60)

    env = os.environ.copy()
    env["RUN_OUT_DIR"] = str(run_dir)
    env["SALIENCY_WEIGHT"] = exp["weight"]
    env["SALIENCY_THRESHOLD"] = exp["threshold"]
    env["DATA_AUGMENT"] = exp["augment"]

    run_failed = False

    for script in scripts:
        print(f"   -> Executing {script}...")
        script_path = base_dir / script
        result = subprocess.run([python_exec, str(script_path)], env=env)

        if result.returncode != 0:
            print(f"\nERROR: {script} failed during {exp['name']}. Skipping to next experiment.")
            run_failed = True
            break

    if not run_failed:
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            row = {**exp, **metrics}
            master_results.append(row)
            print(f"Run {exp['name']} completed. F1 Score: {metrics.get('F1_Weighted', 'N/A')}\n")

if master_results:
    df = pd.DataFrame(master_results)
    csv_path = master_output_dir / f"master_ablation_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(csv_path, index=False)
    print("=" * 60)
    print("ABLATION STUDY COMPLETE")
    print(f"Results compiled and saved to: {csv_path}")
    print("=" * 60)
