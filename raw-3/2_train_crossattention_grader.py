from torchvision import models

class CNNCrossAttentionGrader(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        
        # 1. The CNN Feature Extractor (ResNet-50)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Modify the first layer to accept 1-channel grayscale (Mammogram)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Strip the classification head and global pooling layer.
        # We want the raw spatial feature maps (Shape will be [Batch, 2048, 7, 7])
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # 2. The Cross-Attention Module
        # ResNet50 outputs 2048 channels.
        embed_dim = 2048
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        
        # 3. The Ordinal Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512), # CC + Attended MLO
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), # Strong dropout to prevent overfitting
            nn.Linear(512, num_classes - 1)
        )

    def forward(self, dual_tensor):
        # dual_tensor shape: [Batch, 2, 224, 224]
        cc_img = dual_tensor[:, 0:1, :, :] 
        mlo_img = dual_tensor[:, 1:2, :, :]
        
        # 1. Extract Spatial Feature Maps using CNN
        # Shape: [Batch, 2048, 7, 7]
        cc_feat = self.feature_extractor(cc_img)   
        mlo_feat = self.feature_extractor(mlo_img) 
        
        B, C, H, W = cc_feat.shape
        
        # 2. Prepare for Attention (Flatten spatial dimensions 7x7 -> 49 sequence length)
        # Shape: [Batch, 49, 2048]
        cc_seq = cc_feat.view(B, C, -1).permute(0, 2, 1) 
        mlo_seq = mlo_feat.view(B, C, -1).permute(0, 2, 1)
        
        # 3. Cross-Attention
        # Query = CC features. Key/Value = MLO features.
        # "Where in the MLO feature map do I see confirmation of the CC feature map?"
        attended_mlo_seq, _ = self.cross_attention(query=cc_seq, key=mlo_seq, value=mlo_seq)
        
        # 4. Reshape back to image dimensions and Pool
        attended_mlo_feat = attended_mlo_seq.permute(0, 2, 1).view(B, C, H, W)
        
        cc_pooled = self.global_pool(cc_feat).view(B, -1)
        mlo_pooled = self.global_pool(attended_mlo_feat).view(B, -1)
        
        # 5. Fuse and Classify
        fused = torch.cat([cc_pooled, mlo_pooled], dim=1)
        return self.classifier(fused)