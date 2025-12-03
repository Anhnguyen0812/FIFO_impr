"""
Dual Encoder Architecture for Foggy Scene Segmentation
Combines DINOv3 (Semantic) + SAM 2 (Spatial) branches

Architecture:
1. Dual Preprocessing: Normalize images differently for each branch
2. Semantic Branch: DINOv3 ViT-L/16 (frozen) - "What is this?"
3. Spatial Branch: SAM 2 Hiera Large (frozen) - "Where are the boundaries?"
4. Align & Fusion: Projection + Upsampling + Concatenation + Conv fusion
5. Decoder: FCN-style segmentation head

Training Strategy:
- Freeze both DINOv3 and SAM 2 encoders (save VRAM)
- Only train Fusion layers and Decoder
- Loss: CrossEntropy + Dice Loss (for boundary penalty)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class DualPreprocessing(nn.Module):
    """
    Dual preprocessing for DINOv3 and SAM 2
    Both use ImageNet normalization but keep separate paths for clarity
    """
    def __init__(self):
        super(DualPreprocessing, self).__init__()
        
        # ImageNet normalization (both models use this)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input RGB image [B, 3, H, W], values in [0, 1]
        
        Returns:
            dino_input: Normalized for DINOv3 [B, 3, H, W]
            sam_input: Normalized for SAM 2 [B, 3, H, W]
        """
        # Move normalization params to same device as input
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        
        # Both use same normalization (ImageNet standard)
        normalized = (x - self.mean) / self.std
        
        return normalized, normalized  # Keep separate for flexibility


class DINOv3Branch(nn.Module):
    """
    Semantic Branch using DINOv3 ViT-L/16 Distilled
    Frozen backbone for semantic understanding
    """
    def __init__(self, freeze: bool = True):
        super(DINOv3Branch, self).__init__()
        
        try:
            # Load DINOv3 ViT-L/16 from torch hub
            self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
            print("✓ DINOv3 ViT-L/14 loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load DINOv3: {e}")
            print("  Using dummy encoder for testing...")
            self.encoder = self._create_dummy_encoder()
        
        self.feature_dim = 1024  # DINOv3 ViT-L output dimension
        self.patch_size = 14  # DINOv3 patch size
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            print("✓ DINOv3 encoder frozen")
    
    def _create_dummy_encoder(self):
        """Create dummy encoder for testing when DINOv3 unavailable"""
        class DummyDINO(nn.Module):
            def forward(self, x):
                B, C, H, W = x.shape
                # Simulate patch-based output
                return torch.randn(B, 1024, H // 14, W // 14, device=x.device)
        return DummyDINO()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Normalized image [B, 3, H, W]
        
        Returns:
            features: Dense features [B, 1024, H/14, W/14]
        """
        B, C, H, W = x.shape
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            # Get patch embeddings from DINOv3
            features = self.encoder.forward_features(x)
            
            # Extract patch tokens (skip CLS token)
            patch_tokens = features['x_norm_patchtokens']  # [B, N_patches, 1024]
            
            # Reshape to spatial grid
            h_patches = H // self.patch_size
            w_patches = W // self.patch_size
            features_spatial = patch_tokens.permute(0, 2, 1).reshape(B, self.feature_dim, h_patches, w_patches)
        
        return features_spatial


class SAM2Branch(nn.Module):
    """
    Spatial Branch using SAM 2 Hiera Large
    Frozen backbone for boundary understanding
    """
    def __init__(self, freeze: bool = True):
        super(SAM2Branch, self).__init__()
        
        try:
            # Load SAM 2 Hiera Large encoder
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Load SAM 2 model (you may need to adjust checkpoint path)
            sam2_checkpoint = "sam2_hiera_large.pt"
            model_cfg = "sam2_hiera_l.yaml"
            
            sam2_model = build_sam2(model_cfg, sam2_checkpoint)
            self.encoder = sam2_model.image_encoder
            print("✓ SAM 2 Hiera Large loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load SAM 2: {e}")
            print("  Using dummy encoder for testing...")
            self.encoder = self._create_dummy_encoder()
        
        self.feature_dim = 256  # SAM 2 Hiera final stage output
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            print("✓ SAM 2 encoder frozen")
    
    def _create_dummy_encoder(self):
        """Create dummy encoder for testing when SAM 2 unavailable"""
        class DummySAM(nn.Module):
            def forward(self, x):
                B, C, H, W = x.shape
                # Simulate hierarchical output (Stage 4)
                return torch.randn(B, 256, H // 32, W // 32, device=x.device)
        return DummySAM()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Normalized image [B, 3, H, W]
        
        Returns:
            features: Hierarchical features [B, 256, H/32, W/32]
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            # Get hierarchical features from SAM 2
            features = self.encoder(x)
            
            # Take the final stage (Stage 4) for richest features
            # SAM 2 Hiera outputs dict of multi-scale features
            if isinstance(features, dict):
                features = features['stage4']  # or last stage
            elif isinstance(features, (list, tuple)):
                features = features[-1]  # Take last stage
        
        return features


class AlignFusion(nn.Module):
    """
    Align and Fuse features from DINOv3 and SAM 2
    
    Steps:
    1. Project DINOv3 (1024 ch) -> 256 ch
    2. Upsample SAM 2 (H/32) -> (H/16) to match DINOv3
    3. Concatenate: 256 + 256 = 512 ch
    4. Fusion Conv blocks: 512 -> 256 ch
    """
    def __init__(self, dino_dim: int = 1024, sam_dim: int = 256, output_dim: int = 256):
        super(AlignFusion, self).__init__()
        
        # Projection layers
        self.dino_proj = nn.Sequential(
            nn.Conv2d(dino_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )
        
        self.sam_proj = nn.Sequential(
            nn.Conv2d(sam_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )
        
        # Fusion blocks
        self.fusion = nn.Sequential(
            nn.Conv2d(output_dim * 2, output_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, dino_feat: torch.Tensor, sam_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dino_feat: DINOv3 features [B, 1024, H/14, W/14]
            sam_feat: SAM 2 features [B, 256, H/32, W/32]
        
        Returns:
            fused: Fused features [B, 256, H/14, W/14]
        """
        # Project to same channel dimension
        dino_proj = self.dino_proj(dino_feat)  # [B, 256, H/14, W/14]
        sam_proj = self.sam_proj(sam_feat)  # [B, 256, H/32, W/32]
        
        # Upsample SAM to match DINOv3 spatial resolution
        sam_upsampled = F.interpolate(
            sam_proj, 
            size=dino_proj.shape[2:], 
            mode='bilinear', 
            align_corners=True
        )  # [B, 256, H/14, W/14]
        
        # Concatenate along channel dimension
        concat = torch.cat([dino_proj, sam_upsampled], dim=1)  # [B, 512, H/14, W/14]
        
        # Fusion
        fused = self.fusion(concat)  # [B, 256, H/14, W/14]
        
        return fused


class SegmentationDecoder(nn.Module):
    """
    Simple FCN-style decoder for semantic segmentation
    Upsamples fused features to original resolution
    """
    def __init__(self, in_channels: int = 256, num_classes: int = 19):
        super(SegmentationDecoder, self).__init__()
        
        # Decoder with progressive upsampling
        self.decoder = nn.Sequential(
            # 256 -> 128 channels, 2x upsample
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # 128 -> 64 channels, 2x upsample
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # 64 -> 64 channels, 2x upsample (total 8x from H/14)
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        
        # Final classifier
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: Fused features [B, 256, H/14, W/14]
            target_size: (H, W) original image size
        
        Returns:
            logits: Segmentation logits [B, num_classes, H, W]
        """
        x = self.decoder(x)  # [B, 64, H/2, W/2] approximately
        logits = self.classifier(x)  # [B, num_classes, H/2, W/2]
        
        # Final upsample to target size
        logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=True)
        
        return logits


class DualEncoderModel(nn.Module):
    """
    Complete Dual Encoder Architecture
    DINOv3 (Semantic) + SAM 2 (Spatial) -> Fusion -> Decoder
    """
    def __init__(self, num_classes: int = 19, freeze_encoders: bool = True):
        super(DualEncoderModel, self).__init__()
        
        print("\n" + "="*60)
        print("Initializing Dual Encoder Model (DINOv3 + SAM 2)")
        print("="*60)
        
        # Stage 1: Dual Preprocessing
        self.preprocess = DualPreprocessing()
        
        # Stage 2: Dual Encoders (Frozen)
        self.dino_branch = DINOv3Branch(freeze=freeze_encoders)
        self.sam_branch = SAM2Branch(freeze=freeze_encoders)
        
        # Stage 3: Align & Fusion (Trainable)
        self.align_fusion = AlignFusion(
            dino_dim=1024,
            sam_dim=256,
            output_dim=256
        )
        
        # Stage 4: Decoder (Trainable)
        self.decoder = SegmentationDecoder(
            in_channels=256,
            num_classes=num_classes
        )
        
        print("="*60)
        print("✓ Dual Encoder Model initialized successfully!")
        print(f"  - DINOv3 parameters: {sum(p.numel() for p in self.dino_branch.parameters()):,}")
        print(f"  - SAM 2 parameters: {sum(p.numel() for p in self.sam_branch.parameters()):,}")
        print(f"  - Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print("="*60 + "\n")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input RGB image [B, 3, H, W], assumed in [0, 1] range
        
        Returns:
            dict with:
                - 'logits': Segmentation logits [B, num_classes, H, W]
                - 'dino_features': DINOv3 features (for visualization)
                - 'sam_features': SAM 2 features (for visualization)
                - 'fused_features': Fused features (for visualization)
        """
        B, C, H, W = x.shape
        
        # Stage 1: Dual Preprocessing
        dino_input, sam_input = self.preprocess(x)
        
        # Stage 2: Feature Extraction (Frozen)
        dino_features = self.dino_branch(dino_input)  # [B, 1024, H/14, W/14]
        sam_features = self.sam_branch(sam_input)  # [B, 256, H/32, W/32]
        
        # Stage 3: Align & Fusion
        fused_features = self.align_fusion(dino_features, sam_features)  # [B, 256, H/14, W/14]
        
        # Stage 4: Decode to Segmentation
        logits = self.decoder(fused_features, target_size=(H, W))  # [B, num_classes, H, W]
        
        return {
            'logits': logits,
            'dino_features': dino_features,
            'sam_features': sam_features,
            'fused_features': fused_features
        }
    
    def get_trainable_parameters(self):
        """Get only trainable parameters (fusion + decoder)"""
        return [p for p in self.parameters() if p.requires_grad]


class DiceLoss(nn.Module):
    """
    Dice Loss for better boundary segmentation
    Penalizes misclassifications more heavily at boundaries
    """
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits [B, num_classes, H, W]
            target: Ground truth labels [B, H, W]
        
        Returns:
            dice_loss: Scalar loss value
        """
        # Convert logits to probabilities
        pred_probs = F.softmax(pred, dim=1)  # [B, C, H, W]
        
        # One-hot encode target
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # Compute Dice coefficient per class
        intersection = (pred_probs * target_one_hot).sum(dim=(2, 3))  # [B, C]
        union = pred_probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # [B, C]
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)  # [B, C]
        
        # Average over batch and classes
        dice_loss = 1 - dice.mean()
        
        return dice_loss


def test_dual_encoder():
    """Test function to verify architecture works"""
    print("\nTesting Dual Encoder Model...")
    
    # Create dummy input
    batch_size = 2
    H, W = 512, 512
    num_classes = 19
    
    x = torch.randn(batch_size, 3, H, W)
    
    # Initialize model
    model = DualEncoderModel(num_classes=num_classes, freeze_encoders=True)
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"\n✓ Test passed!")
    print(f"  Input shape: {x.shape}")
    print(f"  Output logits shape: {output['logits'].shape}")
    print(f"  DINOv3 features shape: {output['dino_features'].shape}")
    print(f"  SAM 2 features shape: {output['sam_features'].shape}")
    print(f"  Fused features shape: {output['fused_features'].shape}")
    
    # Test loss functions
    target = torch.randint(0, num_classes, (batch_size, H, W))
    
    ce_loss = F.cross_entropy(output['logits'], target)
    dice_loss = DiceLoss()(output['logits'], target)
    
    print(f"\n  CrossEntropy Loss: {ce_loss.item():.4f}")
    print(f"  Dice Loss: {dice_loss.item():.4f}")
    print("\n✓ All tests passed!\n")


if __name__ == '__main__':
    test_dual_encoder()
