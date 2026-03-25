import os
import sys
import torch
import pandas as pd

# 1. Dynamically get the absolute path of the root project folder (unified-1)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

try:
    from data.dataset_spatial import SpatialMammogramDataset
    from models.master_pipeline import CompleteMammogramPipeline
    from torch.utils.data import DataLoader
    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    print(f"❌ ERROR: Could not import modules. Details: {e}")

def check_directories():
    print("\n--- 1. DIRECTORY STRUCTURE CHECK ---")
    # Use absolute paths based on the project root
    expected_dirs = ['data', 'models', 'utils', 'scripts']
    all_good = True
    
    for d in expected_dirs:
        target_path = os.path.join(PROJECT_ROOT, d)
        if os.path.isdir(target_path):
            print(f"✅ Found directory: {target_path}")
        else:
            print(f"❌ Missing directory: {target_path}")
            all_good = False
    return all_good

def check_manifest():
    print("\n--- 2. DATASET MANIFEST CHECK ---")
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'spatial_manifest.csv')
    
    if not os.path.exists(csv_path):
        print(f"❌ Manifest not found at {csv_path}. Did you run build_manifest.py?")
        return False
    
    df = pd.read_csv(csv_path)
    print(f"✅ Manifest loaded successfully. Total records: {len(df)}")
    
    sample_img = df.iloc[0]['cc_image_path']
    if os.path.exists(sample_img):
        print(f"✅ Verified image paths are valid (e.g., {sample_img})")
    else:
        print(f"⚠️ WARNING: The image paths in the CSV point to files that don't exist: {sample_img}")
    return True

def check_dataloader():
    print("\n--- 3. DATALOADER & TENSOR SHAPE CHECK ---")
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'spatial_manifest.csv')
    try:
        dataset = SpatialMammogramDataset(csv_path)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch = next(iter(dataloader))
        
        cc_img, cc_mask, cc_bbox, mlo_img, mlo_mask, mlo_bbox, labels = batch
        
        print(f"✅ DataLoader successfully batched 2 records.")
        print(f"📐 CC Image Shape:  {cc_img.shape} (Expected: [2, 1, 224, 224])")
        return True
    except Exception as e:
        print(f"❌ DataLoader crashed. Details: {e}")
        return False

def check_pipeline_forward_pass():
    print("\n--- 4. MASTER PIPELINE ARCHITECTURE CHECK ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Running diagnostics on device: {device}")
    
    try:
        model = CompleteMammogramPipeline().to(device)
        model.eval() 
        print(f"✅ Pipeline instantiated successfully.")
        
        print("🧪 Injecting dummy tensors to test forward pass dimensions...")
        dummy_cc = torch.randn(2, 1, 224, 224).to(device)
        dummy_mlo = torch.randn(2, 1, 224, 224).to(device)
        
        with torch.no_grad():
            grade_logits, warped_mlo_mask, cc_mask, ddf = model(dummy_cc, dummy_mlo)
            
        print(f"✅ Forward pass complete. No dimension mismatch errors!")
        print(f"📤 Output 1 (Logits):      {grade_logits.shape} (Expected: [2, 5])")
        print(f"📤 Output 2 (Warped Mask): {warped_mlo_mask.shape} (Expected: [2, 1, 224, 224])")
        
    except Exception as e:
        print(f"❌ Pipeline forward pass failed. Details: {e}")

if __name__ == "__main__":
    print("==================================================")
    print("🔍 MAMMOGRAM AI PIPELINE DIAGNOSTIC TOOL 🔍")
    print("==================================================")
    
    check_directories()
    if MODULES_LOADED:
        has_data = check_manifest()
        if has_data:
            check_dataloader()
        check_pipeline_forward_pass()