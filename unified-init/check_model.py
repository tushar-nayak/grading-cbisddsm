import importlib.util
import sys
from pathlib import Path

import torch


def dynamic_import(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE_DIR = Path(__file__).resolve().parent
script1_path = BASE_DIR / "1_train_saliency_affine.py"
script2_path = BASE_DIR / "2_train_crossattention_grader.py"
script3_path = BASE_DIR / "3_visualize_pipeline.py"

try:
    stn_mod = dynamic_import("script1", script1_path)
    grader_mod = dynamic_import("script2", script2_path)
    viz_mod = dynamic_import("script3", script3_path)
except Exception as e:
    print(f"Load error: {e}")
    sys.exit(1)


def run_checks():
    print("=" * 60)
    print("PRE-FLIGHT VALIDATION")
    print("=" * 60)

    device = torch.device("cpu")

    try:
        stn = stn_mod.RawSpatialTransformer().to(device)
        stn.eval()
        dummy_cc = torch.randn(1, 1, 224, 224)
        dummy_mlo = torch.randn(1, 1, 224, 224)
        aligned, _ = stn(dummy_cc, dummy_mlo)
        print(f"STN initialized. Output shape: {aligned.shape}")

        grader = grader_mod.CNNCrossAttentionGrader().to(device)
        grader.eval()
        dummy_fused = torch.randn(1, 2, 224, 224)
        logits = grader(dummy_fused)
        print(f"Grader initialized. Logits shape: {logits.shape}")

        visualizer = viz_mod.CNN_Visualizer().to(device)
        visualizer.eval()
        attn = visualizer(dummy_fused)
        print(f"Visualizer initialized. Attention map shape: {attn.shape}")

        print("\nValidation passed.")
    except Exception as e:
        print(f"Architecture mismatch found: {e}")


if __name__ == "__main__":
    run_checks()
