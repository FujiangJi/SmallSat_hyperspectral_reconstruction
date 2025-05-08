import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(SAMLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        pred_flat = pred.reshape(B, C, -1)
        target_flat = target.reshape(B, C, -1)

        dot_product = torch.sum(pred_flat * target_flat, dim=1)
        pred_norm = torch.norm(pred_flat, dim=1)
        target_norm = torch.norm(target_flat, dim=1)

        cos_sim = dot_product / (pred_norm * target_norm + self.eps)
        sam_angle = torch.acos(torch.clamp(cos_sim, -1, 1))
        return sam_angle.mean()
    
    
class FusionLoss(nn.Module):
    def __init__(self, lambda_sam=0.2, lambda_mse=0.6, lambda_smooth=0.2):
        
        super(FusionLoss, self).__init__()
        self.sam_loss = SAMLoss()
        self.mse_loss = nn.MSELoss()
        self.smooth_loss = nn.SmoothL1Loss()

        # Weights for balancing losses
        self.lambda_sam = lambda_sam
        self.lambda_mse = lambda_mse
        self.lambda_smooth = lambda_smooth

    def forward(self, pred, target):
        loss_sam = self.sam_loss(pred, target)
        loss_mse = self.mse_loss(pred, target)
        loss_smooth = self.smooth_loss(pred, target)

        # Combined loss
        total_loss = self.lambda_sam * loss_sam + self.lambda_mse * loss_mse + self.lambda_smooth * loss_smooth
        # print(total_loss)
        # print({'SAM': loss_sam, 'MSE': loss_mse, 'SmoothL1': loss_smooth})
        return total_loss

            
      

