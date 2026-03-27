"""
Model Architectures — SegFormer, DeepLabV3+, Swin-UNet.
All models: (B, 3, 512, 512) → (B, num_classes, 512, 512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from src.config import NUM_SEG_CLASSES, NUM_ROOF_TYPES, TrainConfig


# ============================================================
# BACKBONE UTILITIES
# ============================================================

def _check_import(module_name: str):
    """Check if a module is available and give install hint."""
    try:
        __import__(module_name)
    except ImportError:
        raise ImportError(
            f'{module_name} not installed. Run: '
            f'pip install {module_name.replace("_", "-")}'
        )


# ============================================================
# SEGFORMER (Main Model — Best accuracy/speed tradeoff)
# ============================================================

class SegFormerModel(nn.Module):
    """
    SegFormer with MiT-B3 backbone.
    - Hierarchical Transformer encoder
    - Lightweight MLP decoder
    - Outstanding for dense prediction at 512×512
    
    Reference: Xie et al., "SegFormer: Simple and Efficient Design for
    Semantic Segmentation with Transformers", NeurIPS 2021.
    """
    
    def __init__(
        self,
        num_classes: int = NUM_SEG_CLASSES,
        backbone: str = 'mit_b3',
        pretrained: bool = True,
        num_roof_classes: int = NUM_ROOF_TYPES + 1,  # +1 for non-building (0)
    ):
        super().__init__()
        _check_import('transformers')
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
        
        # Map backbone name to HuggingFace model ID
        backbone_map = {
            'mit_b0': 'nvidia/segformer-b0-finetuned-ade-512-512',
            'mit_b1': 'nvidia/segformer-b1-finetuned-ade-512-512',
            'mit_b2': 'nvidia/segformer-b2-finetuned-ade-512-512',
            'mit_b3': 'nvidia/segformer-b3-finetuned-ade-512-512',
            'mit_b4': 'nvidia/segformer-b4-finetuned-ade-512-512',
            'mit_b5': 'nvidia/segformer-b5-finetuned-ade-512-512',
        }
        
        model_id = backbone_map.get(backbone, backbone_map['mit_b3'])
        
        if pretrained:
            # Load pretrained and change head for our classes
            self.backbone = SegformerForSemanticSegmentation.from_pretrained(
                model_id,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig.from_pretrained(model_id)
            config.num_labels = num_classes
            self.backbone = SegformerForSemanticSegmentation(config)
        
        # Roof type classification head (secondary task)
        # Uses last encoder stage features → separate projection head
        encoder_channels = self.backbone.config.hidden_sizes[-1]   # 512 for mit_b3
        hidden_size = self.backbone.config.decoder_hidden_size       # 256
        self.roof_head = nn.Sequential(
            nn.Conv2d(encoder_channels, hidden_size, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, num_roof_classes, 1),
        )
        
        self.num_classes = num_classes
        self.num_roof_classes = num_roof_classes
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input RGB image
        
        Returns:
            Dict with 'seg_logits' (B, C, H, W) and 'roof_logits' (B, R, H, W)
        """
        h, w = x.shape[2:]
        
        # Single encoder pass — get multi-scale hidden states
        encoder_outputs = self.backbone.segformer(
            x, output_hidden_states=True, return_dict=True,
        )
        hidden_states = encoder_outputs.hidden_states  # tuple of 4 scales
        
        # Decode for segmentation (reuse backbone decode_head)
        seg_logits = self.backbone.decode_head(hidden_states)
        seg_logits = F.interpolate(
            seg_logits, size=(h, w), mode='bilinear', align_corners=False,
        )
        
        # Roof type head — project last encoder stage features
        last_feat = hidden_states[-1]  # (B, C_last, H/32, W/32)
        roof_logits = self.roof_head(last_feat)
        roof_logits = F.interpolate(
            roof_logits, size=(h, w), mode='bilinear', align_corners=False,
        )
        
        return {
            'seg_logits': seg_logits,
            'roof_logits': roof_logits,
        }


class SegFormerSimple(nn.Module):
    """
    Simplified SegFormer — just semantic segmentation, no multi-task.
    Cleaner for initial training. Switch to multi-task after convergence.
    """
    
    def __init__(
        self,
        num_classes: int = NUM_SEG_CLASSES,
        backbone: str = 'mit_b3',
        pretrained: bool = True,
    ):
        super().__init__()
        _check_import('transformers')
        from transformers import SegformerForSemanticSegmentation
        
        backbone_map = {
            'mit_b0': 'nvidia/segformer-b0-finetuned-ade-512-512',
            'mit_b1': 'nvidia/segformer-b1-finetuned-ade-512-512',
            'mit_b2': 'nvidia/segformer-b2-finetuned-ade-512-512',
            'mit_b3': 'nvidia/segformer-b3-finetuned-ade-512-512',
            'mit_b4': 'nvidia/segformer-b4-finetuned-ade-512-512',
            'mit_b5': 'nvidia/segformer-b5-finetuned-ade-512-512',
        }
        
        model_id = backbone_map.get(backbone, backbone_map['mit_b3'])
        
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_id,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            from transformers import SegformerConfig
            config = SegformerConfig.from_pretrained(model_id)
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, w = x.shape[2:]
        outputs = self.model(x)
        logits = outputs.logits
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        return {'seg_logits': logits}


# ============================================================
# DEEPLABV3+ (Strong baseline — proven for satellite imagery)
# ============================================================

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with ResNet-101 or EfficientNet backbone.
    Excellent for structured outputs (buildings, roads).
    
    Uses segmentation_models_pytorch (smp) library.
    """
    
    def __init__(
        self,
        num_classes: int = NUM_SEG_CLASSES,
        backbone: str = 'resnet101',
        pretrained: str = 'imagenet',
    ):
        super().__init__()
        _check_import('segmentation_models_pytorch')
        import segmentation_models_pytorch as smp
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights=pretrained,
            in_channels=3,
            classes=num_classes,
        )
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.model(x)
        return {'seg_logits': logits}


# ============================================================
# U-NET (smp variants — versatile baseline)
# ============================================================

class UNetPlusPlus(nn.Module):
    """
    UNet++ with various backbones via segmentation_models_pytorch.
    Good for fine-grained boundary detection.
    """
    
    def __init__(
        self,
        num_classes: int = NUM_SEG_CLASSES,
        backbone: str = 'efficientnet-b4',
        pretrained: str = 'imagenet',
    ):
        super().__init__()
        _check_import('segmentation_models_pytorch')
        import segmentation_models_pytorch as smp
        
        self.model = smp.UnetPlusPlus(
            encoder_name=backbone,
            encoder_weights=pretrained,
            in_channels=3,
            classes=num_classes,
        )
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.model(x)
        return {'seg_logits': logits}


# ============================================================
# MODEL FACTORY
# ============================================================

def build_model(config: TrainConfig = TrainConfig()) -> nn.Module:
    """
    Build a model from config.
    
    Supported models:
      - 'segformer': SegFormer with MiT backbone (default)
      - 'segformer_simple': SegFormer without multi-task heads
      - 'deeplabv3plus': DeepLabV3+ with ResNet/EfficientNet
      - 'unetpp': UNet++ with various backbones
    """
    model_name = config.model_name.lower()
    
    if model_name == 'segformer':
        model = SegFormerModel(
            num_classes=config.num_classes,
            backbone=config.backbone,
            pretrained=config.pretrained,
        )
    elif model_name == 'segformer_simple':
        model = SegFormerSimple(
            num_classes=config.num_classes,
            backbone=config.backbone,
            pretrained=config.pretrained,
        )
    elif model_name == 'deeplabv3plus':
        model = DeepLabV3Plus(
            num_classes=config.num_classes,
            backbone=config.backbone if 'resnet' in config.backbone or 'efficient' in config.backbone else 'resnet101',
            pretrained='imagenet' if config.pretrained else None,
        )
    elif model_name in ('unetpp', 'unet++', 'unet_plus_plus'):
        model = UNetPlusPlus(
            num_classes=config.num_classes,
            backbone=config.backbone if 'efficient' in config.backbone else 'efficientnet-b4',
            pretrained='imagenet' if config.pretrained else None,
        )
    else:
        raise ValueError(f'Unknown model: {model_name}. Choose from: segformer, segformer_simple, deeplabv3plus, unetpp')
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {model_name} | Params: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable')
    
    return model


# ============================================================
# EMA (Exponential Moving Average) — Stabilize training
# ============================================================

class EMAModel:
    """
    Exponential Moving Average of model parameters.
    Improves generalization by maintaining a smoothed version of weights.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """Update shadow weights with current model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self, model: nn.Module):
        """Replace model weights with EMA weights (for eval/inference)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """Restore original weights after EMA evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
