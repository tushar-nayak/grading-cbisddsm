import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SpatialMammogramDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def create_bbox_heatmap(self, bbox, img_size=(224, 224)):
        heatmap = torch.zeros((1, img_size[0], img_size[1]))
        if pd.notna(bbox):
            x1, y1, x2, y2 = map(int, str(bbox).split(','))
            heatmap[0, y1:y2, x1:x2] = 1.0
        return heatmap

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        cc_img = self.transform(Image.open(row['cc_image_path']).convert('L'))
        mlo_img = self.transform(Image.open(row['mlo_image_path']).convert('L'))
        
        cc_mask = self.transform(Image.open(row['cc_mask_path']).convert('L'))
        mlo_mask = self.transform(Image.open(row['mlo_mask_path']).convert('L'))
        
        cc_bbox = self.create_bbox_heatmap(row.get('cc_bbox', None))
        mlo_bbox = self.create_bbox_heatmap(row.get('mlo_bbox', None))
        
        label = int(row['birads_label']) - 1 

        return cc_img, cc_mask, cc_bbox, mlo_img, mlo_mask, mlo_bbox, torch.tensor(label, dtype=torch.long)