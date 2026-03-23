import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_spatial import SpatialMammogramDataset
from models.master_pipeline import CompleteMammogramPipeline
from utils.metrics import calculate_ncc, calculate_jacobian_determinant

def simulated_llm_scribe(predicted_class, confidence):
    """Simulates an LLM API call for report generation."""
    prompt = f"Write a clinical impression for a BI-RADS {predicted_class} finding with {confidence:.1f}% AI confidence."
    return f"IMPRESSION: The multi-modal analysis indicates findings consistent with BI-RADS {predicted_class} (Confidence: {confidence:.1f}%). Standard protocols for this category should be observed."

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SpatialMammogramDataset('../data/spatial_manifest.csv')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = CompleteMammogramPipeline().to(device)
    # model.load_state_dict(torch.load('../models/weights/master_pipeline_v1.pth'))
    model.eval()

    total_ncc = 0
    total_folds = 0
    
    print("Starting Evaluation Pipeline...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 5: break # Just test the first 5 for now
            
            cc_img, cc_mask, cc_bbox, mlo_img, mlo_mask, mlo_bbox, labels = [b.to(device) for b in batch]
            
            grade_logits, warped_mlo_mask, _, ddf = model(cc_img, mlo_img)
            
            # 1. Registration Metrics
            warped_mlo_img = model.stn(torch.cat([cc_img, cc_mask, cc_bbox], dim=1), 
                                       torch.cat([mlo_img, mlo_mask, mlo_bbox], dim=1)) # get DDF
            warped_img_actual = warp_tensor(mlo_img, ddf)
            
            ncc = calculate_ncc(cc_img, warped_img_actual)
            folds = calculate_jacobian_determinant(ddf)
            total_ncc += ncc
            total_folds += folds
            
            # 2. Classification & LLM Output
            probs = F.softmax(grade_logits, dim=1)
            predicted_class = torch.argmax(probs).item() + 1
            confidence = probs[0][predicted_class - 1].item() * 100
            
            report = simulated_llm_scribe(predicted_class, confidence)
            
            print(f"\n--- Patient {i+1} ---")
            print(f"Registration NCC: {ncc:.4f} | Tissue Folds: {folds}")
            print(report)

if __name__ == "__main__":
    evaluate()