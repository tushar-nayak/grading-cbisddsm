import torch

def calculate_ncc(img1, img2):
    """Calculates Normalized Cross-Correlation (Similarity)."""
    i1_mean, i2_mean = torch.mean(img1), torch.mean(img2)
    i1_std, i2_std = torch.std(img1), torch.std(img2)
    ncc = torch.mean((img1 - i1_mean) * (img2 - i2_mean)) / (i1_std * i2_std + 1e-8)
    return ncc.item()

def calculate_jacobian_determinant(ddf):
    """
    Calculates the Jacobian determinant of the dense displacement field.
    Values <= 0 indicate biologically impossible tissue folding.
    """
    dy = ddf[:, 1, 1:, :-1] - ddf[:, 1, :-1, :-1]
    dx = ddf[:, 0, :-1, 1:] - ddf[:, 0, :-1, :-1]
    
    # Approximate determinant for 2D field
    jac = (1 + dx) * (1 + dy) 
    folds = torch.sum(jac <= 0).item()
    return folds