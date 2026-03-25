import os
import torch
import pandas as pd
from PIL import Image
import importlib.util
import sys

def dynamic_import(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

BASE_DIR = '/home/sofa/host_dir/spatial_alignment/raw-4'
script1_path = os.path.join(BASE_DIR, '1_train_saliency_affine.py')
script2_path = os.path.join(BASE_DIR, '2_train_crossattention_grader.py')
script3_path = os.path.join(BASE_DIR, '3_visualize_pipeline.py')

try:
    stn_mod = dynamic_import("script1", script1_path)
    grader_mod = dynamic_import("script2", script2_path)
    viz_mod = dynamic_import("script3", script3_path)
except Exception as e:
    print(f"❌ Load Error: {e}")
    sys.exit(1)

def run_checks():
    print("="*60)
    print("🔍 PRE-FLIGHT VALIDATION (FIXED)")
    print("="*60)

    device = torch.device("cpu")
    
    try:
        # 1. Check STN
        stn = stn_mod.RawSpatialTransformer().to(device)
        stn.eval() # <--- CRITICAL FIX
        dummy_cc = torch.randn(1, 1, 224, 224)
        dummy_mlo = torch.randn(1, 1, 224, 224)
        aligned, theta = stn(dummy_cc, dummy_mlo)
        print(f"✅ STN Initialized. Output Shape: {aligned.shape}")
        
        # 2. Check Grader
        grader = grader_mod.CNNCrossAttentionGrader().to(device)
        grader.eval() # <--- CRITICAL FIX: BatchNorm won't crash with size 1 now
        dummy_fused = torch.randn(1, 2, 224, 224)
        logits = grader(dummy_fused)
        print(f"✅ Grader Initialized. Logits Shape: {logits.shape}")
        
        # 3. Check Visualizer
        visualizer = viz_mod.CNN_Visualizer().to(device)
        visualizer.eval() # <--- CRITICAL FIX
        attn = visualizer(dummy_fused)
        print(f"✅ Visualizer Initialized. Attention Map Shape: {attn.shape}")
        
        print("\n🚀 ALL SYSTEMS NOMINAL. VALIDATION PASSED.")
    except Exception as e:
        print(f"❌ ERROR: Architecture mismatch found! Error: {e}")

if __name__ == "__main__":
    run_checks()