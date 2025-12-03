"""
SegFormer v3 Backbone for FIFO
Replaces ResNet-101 with SegFormer MIT-B5 (Transformer-based)

SegFormer advantages over ResNet:
1. Hierarchical Transformer encoder (better long-range dependencies)
2. Overlap patch embeddings (better local details)
3. Mix-FFN decoder (efficient multi-scale fusion)
4. Pre-trained on large-scale datasets
5. Better performance on foggy scenes

Architecture:
- Encoder: MIT-B5 (4 stages, hierarchical features, 82M params)
- Decoder: Lightweight All-MLP decoder
- Output: Multi-scale features compatible with FIFO's fog-pass filter

Citation:
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={NeurIPS},
  year={2021}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import warnings

# Try to import transformers, if not available, provide fallback
try:
    from transformers import SegformerForSemanticSegmentation, SegformerConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "transformers library not found. Please install: pip install transformers\n"
        "Using dummy encoder for testing."
    )


class SegFormerBackbone(nn.Module):
    """
    SegFormer MIT-B5 backbone for FIFO
    Extracts multi-scale hierarchical features
    
    Output features compatible with FIFO's architecture:
    - out1: [B, 64, H/4, W/4]   - Stage 1 (like ResNet conv1)
    - out2: [B, 128, H/8, W/8]  - Stage 2 (like ResNet layer1)
    - out3: [B, 320, H/16, W/16] - Stage 3 (like ResNet layer2)
    - out4: [B, 512, H/32, W/32] - Stage 4 (like ResNet layer3)
    - out5: [B, 512, H/32, W/32] - Stage 4 refined (like ResNet layer4)
    
    Note: MIT-B5 has 82M parameters vs B3's 47M, providing better performance
    """
    
    def __init__(
        self, 
        num_classes: int = 19,
        pretrained: bool = True,
        pretrained_model_name: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        freeze_encoder: bool = False
    ):
        super(SegFormerBackbone, self).__init__()
        
        self.num_classes = num_classes
        print("\n" + "="*70)
        print("Initializing SegFormer MIT-B5 Backbone")
        print("="*70)
        
        if TRANSFORMERS_AVAILABLE:
            # Load pre-trained SegFormer
            if pretrained:
                print(f"Loading pre-trained model: {pretrained_model_name}")
                self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                    pretrained_model_name
                )
                print("✓ Pre-trained weights loaded successfully")
            else:
                print("Initializing from scratch (no pre-training)")
                config = SegformerConfig.from_pretrained(pretrained_model_name)
                config.num_labels = num_classes
                self.segformer = SegformerForSemanticSegmentation(config)
            
            # Get encoder (MIT-B5)
            self.encoder = self.segformer.segformer.encoder
            
            # MIT-B5 channel dimensions: [64, 128, 320, 512] (same as B3, but deeper)
            self.encoder_channels = [64, 128, 320, 512]
            
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                print("✓ Encoder frozen (no gradients)")
            else:
                print("✓ Encoder trainable")
                
        else:
            # Dummy encoder for testing when transformers not installed
            print("⚠ WARNING: Using dummy encoder (transformers not installed)")
            self.encoder = self._create_dummy_encoder()
            self.encoder_channels = [64, 128, 320, 512]
        
        # Projection layers to match FIFO's expected dimensions
        # FIFO expects: [64, 256, 512, 1024, 2048] like ResNet-101
        # SegFormer B5 gives: [64, 128, 320, 512] (same dims as B3, but deeper layers)
        
        # Stage 1: 64 -> 64 (identity, matches FIFO conv1)
        self.proj1 = nn.Identity()
        
        # Stage 2: 128 -> 256 (match FIFO layer1)
        self.proj2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3: 320 -> 512 (match FIFO layer2)
        self.proj3 = nn.Sequential(
            nn.Conv2d(320, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Stage 4: 512 -> 1024 (match FIFO layer3)
        self.proj4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Stage 5: 512 -> 2048 (match FIFO layer4)
        # Since SegFormer only has 4 stages, we reuse stage 4 with additional processing
        self.proj5 = nn.Sequential(
            nn.Conv2d(512, 2048, kernel_size=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        
        print("="*70)
        print(f"✓ SegFormer Backbone initialized successfully!")
        print(f"  - Encoder channels: {self.encoder_channels}")
        print(f"  - Output channels: [64, 256, 512, 1024, 2048] (FIFO compatible)")
        print(f"  - Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print("="*70 + "\n")
    
    def _create_dummy_encoder(self):
        """Create dummy encoder for testing when transformers not available"""
        class DummyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Conv2d(3, 64, 1)
            
            def forward(self, pixel_values):
                B, C, H, W = pixel_values.shape
                # Simulate 4 stages of hierarchical features
                features = [
                    torch.randn(B, 64, H//4, W//4, device=pixel_values.device),
                    torch.randn(B, 128, H//8, W//8, device=pixel_values.device),
                    torch.randn(B, 320, H//16, W//16, device=pixel_values.device),
                    torch.randn(B, 512, H//32, W//32, device=pixel_values.device),
                ]
                return features
        
        return DummyEncoder()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: Input image [B, 3, H, W]
        
        Returns:
            Tuple of 5 feature maps (compatible with FIFO):
                - out1: [B, 64, H/4, W/4]
                - out2: [B, 256, H/8, W/8]
                - out3: [B, 512, H/16, W/16]
                - out4: [B, 1024, H/32, W/32]
                - out5: [B, 2048, H/32, W/32]
        """
        B, C, H, W = x.shape
        
        # Extract hierarchical features from SegFormer encoder
        if TRANSFORMERS_AVAILABLE:
            # SegFormer encoder returns list of 4 feature maps
            encoder_outputs = self.encoder(x, output_hidden_states=True, return_dict=True)
            features = encoder_outputs.hidden_states  # List of [B, H*W, C] for each stage
            
            # Reshape to spatial format [B, C, H, W]
            feature_maps = []
            h, w = H, W
            for i, feat in enumerate(features):
                # feat shape: [B, H_i*W_i, C_i]
                h = h // (2 if i > 0 else 4)  # Stage 1: /4, Stage 2-4: /2 each
                w = w // (2 if i > 0 else 4)
                c = self.encoder_channels[i]
                
                # Reshape: [B, H*W, C] -> [B, C, H, W]
                feat_map = feat.transpose(1, 2).reshape(B, c, h, w)
                feature_maps.append(feat_map)
        else:
            # Dummy encoder
            feature_maps = self.encoder(x)
        
        # Extract 4 stages from SegFormer
        feat1 = feature_maps[0]  # [B, 64, H/4, W/4]
        feat2 = feature_maps[1]  # [B, 128, H/8, W/8]
        feat3 = feature_maps[2]  # [B, 320, H/16, W/16]
        feat4 = feature_maps[3]  # [B, 512, H/32, W/32]
        
        # Project to FIFO-compatible dimensions
        out1 = self.proj1(feat1)  # [B, 64, H/4, W/4] - unchanged
        out2 = self.proj2(feat2)  # [B, 256, H/8, W/8]
        out3 = self.proj3(feat3)  # [B, 512, H/16, W/16]
        out4 = self.proj4(feat4)  # [B, 1024, H/32, W/32]
        out5 = self.proj5(feat4)  # [B, 2048, H/32, W/32] - reuse feat4
        
        return out1, out2, out3, out4, out5
    
    def freeze_encoder(self):
        """Freeze encoder weights (for fine-tuning decoder only)"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("✓ Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights (for full fine-tuning)"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("✓ Encoder unfrozen")


class SegFormerFIFO(nn.Module):
    """
    Complete SegFormer-based FIFO model
    Combines SegFormer backbone with FIFO's RefineNet-style decoder
    """
    
    def __init__(
        self,
        num_classes: int = 19,
        pretrained: bool = True,
        pretrained_model_name: str = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
        freeze_encoder: bool = False
    ):
        super(SegFormerFIFO, self).__init__()
        
        # SegFormer backbone
        self.backbone = SegFormerBackbone(
            num_classes=num_classes,
            pretrained=pretrained,
            pretrained_model_name=pretrained_model_name,
            freeze_encoder=freeze_encoder
        )
        
        # RefineNet-style decoder (reuse from original FIFO)
        # This part is the same as original RefineNetLW decoder
        from utils.layer_factory import conv1x1, CRPBlock
        
        self.do = nn.Dropout(p=0.5)
        
        # Decoder layers (same as original FIFO)
        self.p_ims1d2_outl1_dimred = conv1x1(2048, 512, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)
        self.p_ims1d2_outl2_dimred = conv1x1(1024, 256, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred = conv1x1(512, 256, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred = conv1x1(256, 256, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)

        self.clf_conv = nn.Conv2d(
            256, num_classes, kernel_size=3, stride=1, padding=1, bias=True
        )
    
    def _make_crp(self, in_planes, out_planes, stages):
        from utils.layer_factory import CRPBlock
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass matching original FIFO architecture
        """
        # Backbone forward (SegFormer instead of ResNet)
        out1, out2, out3, out4, out5 = self.backbone(x)
        
        # Decoder (same as original FIFO RefineNet decoder)
        l4 = self.do(out5)
        l3 = self.do(out4)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode="bilinear", align_corners=True)(x4)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=out3.size()[2:], mode="bilinear", align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(out3)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=out2.size()[2:], mode="bilinear", align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(out2)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)

        out = self.clf_conv(x1)
        
        return out1, out2, out3, out4, out5, out


def segformer_fifo(
    num_classes: int = 19,
    pretrained: bool = True,
    pretrained_model_name: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    freeze_encoder: bool = False,
    **kwargs
):
    """
    Create SegFormer-based FIFO model
    
    Args:
        num_classes: Number of segmentation classes
        pretrained: Whether to load pre-trained weights
        pretrained_model_name: HuggingFace model name to load
        freeze_encoder: Whether to freeze encoder weights
    
    Returns:
        SegFormerFIFO model
    
    Recommended pretrained models:
        - "nvidia/segformer-b5-finetuned-cityscapes-1024-1024" (default, 82M params)
        - "nvidia/segformer-b5-finetuned-ade-640-640" (ADE20k)
        - "nvidia/segformer-b3-finetuned-cityscapes-1024-1024" (lighter, 47M params)
    """
    model = SegFormerFIFO(
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_model_name=pretrained_model_name,
        freeze_encoder=freeze_encoder
    )
    return model


def test_segformer_backbone():
    """Test function to verify SegFormer backbone works"""
    print("\nTesting SegFormer Backbone...")
    
    # Create model
    model = SegFormerBackbone(num_classes=19, pretrained=False)
    model.eval()
    
    # Test input
    x = torch.randn(2, 3, 512, 512)
    
    # Forward pass
    with torch.no_grad():
        out1, out2, out3, out4, out5 = model(x)
    
    print(f"\n✓ Test passed!")
    print(f"  Input shape: {x.shape}")
    print(f"  out1 (conv1): {out1.shape}")
    print(f"  out2 (layer1): {out2.shape}")
    print(f"  out3 (layer2): {out3.shape}")
    print(f"  out4 (layer3): {out4.shape}")
    print(f"  out5 (layer4): {out5.shape}")
    
    # Verify shapes match FIFO expectations
    assert out1.shape == (2, 64, 128, 128), f"out1 shape mismatch: {out1.shape}"
    assert out2.shape == (2, 256, 64, 64), f"out2 shape mismatch: {out2.shape}"
    assert out3.shape == (2, 512, 32, 32), f"out3 shape mismatch: {out3.shape}"
    assert out4.shape == (2, 1024, 16, 16), f"out4 shape mismatch: {out4.shape}"
    assert out5.shape == (2, 2048, 16, 16), f"out5 shape mismatch: {out5.shape}"
    
    print("\n✓ All shape checks passed! SegFormer backbone is FIFO-compatible.\n")


if __name__ == '__main__':
    test_segformer_backbone()
