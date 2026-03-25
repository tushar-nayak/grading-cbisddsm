import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

# Ensure Python can find the adjacent directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_spatial import SpatialMammogramDataset
from models.master_pipeline import CompleteMammogramPipeline
from utils.losses import MasterPipelineLoss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing pipeline on {device}...")

    dataset = SpatialMammogramDataset('../data/spatial_manifest.csv')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    
    model = CompleteMammogramPipeline().to(device)
    criterion = MasterPipelineLoss(lambda_dice=2.0, lambda_smooth=0.5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_total, running_cls, running_dice = 0.0, 0.0, 0.0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress:
            cc_img, cc_mask, cc_bbox, mlo_img, mlo_mask, mlo_bbox, labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            grade_logits, warped_mlo_mask, generated_cc_mask, ddf = model(cc_img, mlo_img)
            
            total_loss, cls_loss, align_loss = criterion(
                grade_logits, labels, warped_mlo_mask, cc_mask, ddf
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_total += total_loss.item()
            running_cls += cls_loss.item()
            running_dice += align_loss.item()
            
            progress.set_postfix({'Total': f"{total_loss.item():.3f}", 'Cls': f"{cls_loss.item():.3f}", 'Dice': f"{align_loss.item():.3f}"})

        print(f"Epoch Avg -> Total: {running_total/len(dataloader):.4f} | Cls: {running_cls/len(dataloader):.4f} | Dice: {running_dice/len(dataloader):.4f}")

    torch.save(model.state_dict(), '../models/weights/master_pipeline_v1.pth')

if __name__ == "__main__":
    main()
