import subprocess
import sys

python_exec = "/home/sofa/miniconda3/envs/phy/bin/python"
base_dir = "/home/sofa/host_dir/spatial_alignment/raw-4"

scripts = [
    ("1_train_saliency_affine.py", "Training Saliency-Weighted STN"),
    ("2_train_crossattention_grader.py", "Training CNN ResNet50 Grader"),
    ("3_visualize_pipeline.py", "Generating Visual Diagnostics")
]

for script, description in scripts:
    print(f"\n{'='*55}")
    print(f"🚀 STARTING: {description}")
    print(f"{'='*55}")
    
    script_path = f"{base_dir}/{script}"
    
    # Run the script and wait for it to finish
    result = subprocess.run([python_exec, script_path])
    
    # If a script crashes, halt the entire pipeline
    if result.returncode != 0:
        print(f"\n❌ ERROR: {script} failed! Halting pipeline.")
        sys.exit(1)

print("\n" + "="*55)
print("✅ FULL PIPELINE COMPLETE!")
print("="*55)
