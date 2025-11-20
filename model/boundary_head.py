"""
Boundary Detection Head for FIFO
Multi-task learning auxiliary task to improve edge/boundary detection in foggy scenes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryHead(nn.Module):
    """
    Lightweight boundary detection head that takes encoder features
    and outputs binary boundary map.
    
    Uses features from layer0 (low-level) and layer1 (mid-level) for better edge detection.
    """
    
    def __init__(self, in_channels_low=64, in_channels_mid=256, out_channels=1):
        super(BoundaryHead, self).__init__()
        
        # Process low-level features (out1 from conv1) - rich in edge information
        self.low_conv = nn.Sequential(
            nn.Conv2d(in_channels_low, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Process mid-level features (out2 from layer1) - semantic context
        self.mid_conv = nn.Sequential(
            nn.Conv2d(in_channels_mid, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)  # 1x1 conv for final prediction
        )
        
    def forward(self, feat_low, feat_mid):
        """
        Args:
            feat_low: Low-level features from layer0 [B, C_low, H/4, W/4]
            feat_mid: Mid-level features from layer1 [B, C_mid, H/8, W/8]
        Returns:
            boundary_map: Binary boundary prediction [B, 1, H, W]
        """
        # Process features
        low = self.low_conv(feat_low)  # [B, 128, H/4, W/4]
        mid = self.mid_conv(feat_mid)  # [B, 128, H/8, W/8]
        
        # Upsample mid to match low resolution
        mid_up = F.interpolate(mid, size=low.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate
        fused = torch.cat([low, mid_up], dim=1)  # [B, 256, H/4, W/4]
        
        # Final prediction
        boundary = self.fusion(fused)  # [B, 1, H/4, W/4]
        
        return boundary


def generate_boundary_label(seg_label, kernel_size=3):
    """
    Generate boundary labels from segmentation ground truth masks.
    
    Uses morphological operations to find object boundaries.
    This should be called on GT labels, NOT on foggy input images!
    
    Args:
        seg_label: [B, H, W] segmentation ground truth (int labels)
        kernel_size: size of dilation/erosion kernel (default 3)
    Returns:
        boundary_label: [B, 1, H, W] binary boundary map (0=non-boundary, 1=boundary)
    """
    device = seg_label.device
    B, H, W = seg_label.shape
    
    boundary_labels = []
    
    for i in range(B):
        label = seg_label[i]  # [H, W]
        
        # Convert to float for morphological operations
        label_float = label.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Create kernel for dilation/erosion
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
        
        # Dilation
        dilated = F.max_pool2d(label_float, kernel_size, stride=1, padding=kernel_size//2)
        
        # Erosion (approximated by -max_pool(-x))
        eroded = -F.max_pool2d(-label_float, kernel_size, stride=1, padding=kernel_size//2)
        
        # Boundary = dilated - eroded (morphological gradient)
        boundary = (dilated - eroded) > 0
        boundary = boundary.float()
        
        boundary_labels.append(boundary)
    
    boundary_labels = torch.cat(boundary_labels, dim=0)  # [B, 1, H, W]
    
    return boundary_labels
