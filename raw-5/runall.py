import os
import subprocess
import sys
import json
import pandas as pd
from datetime import datetime

python_exec = "/home/sofa/miniconda3/envs/phy/bin/python"
base_dir = "/home/sofa/host_dir/spatial_alignment/raw-4"
master_output_dir = os.path.join(base_dir, "ablation_studies")

# Define the Ablation Experiments
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
    "3_visualize_pipeline.py"
]

os.makedirs(master_output_dir, exist_ok=True)
master_results = []

print(f"🚀 Starting 10-Hour Ablation Study: {len(experiments)} Runs Scheduled.\n")

for exp in experiments:
    run_dir = os.path.join(master_output_dir, exp["name"])
    os.makedirs(run_dir, exist_ok=True)
    
    print("="*60)
    print(f"🧪 INITIATING RUN: {exp['name']}")
    print(f"📁 Output Directory: {run_dir}")
    print("="*60)
    
    # Set Environment Variables for the child scripts to read
    env = os.environ.copy()
    env["RUN_OUT_DIR"] = run_dir
    env["SALIENCY_WEIGHT"] = exp["weight"]
    env["SALIENCY_THRESHOLD"] = exp["threshold"]
    env["DATA_AUGMENT"] = exp["augment"]
    
    run_failed = False
    
    for script in scripts:
        print(f"   -> Executing {script}...")
        script_path = os.path.join(base_dir, script)
        
        result = subprocess.run([python_exec, script_path], env=env)
        
        if result.returncode != 0:
            print(f"\n❌ ERROR: {script} failed during {exp['name']}! Skipping to next experiment.")
            run_failed = True
            break # Break out of the scripts loop and move to the next experiment
            
    if not run_failed:
        # Read the metrics dumped by script 2
        metrics_file = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                
            # Combine config and results for the master CSV
            row = {**exp, **metrics}
            master_results.append(row)
            print(f"✅ Run {exp['name']} Completed. F1 Score: {metrics.get('F1_Weighted', 'N/A')}\n")

# Wrap up and save Master CSV
if master_results:
    df = pd.DataFrame(master_results)
    csv_path = os.path.join(master_output_dir, f"master_ablation_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
    df.to_csv(csv_path, index=False)
    print("="*60)
    print("🎉 ABLATION STUDY COMPLETE!")
    print(f"📊 Results compiled and saved to: {csv_path}")
    print("="*60)