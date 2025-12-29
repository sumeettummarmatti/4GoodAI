"""
Scratch Detection Model Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ScratchDetector(nn.Module):
    """Multi-class defect detection model"""
    
    def __init__(self, model_name='resnet50', pretrained=True, 
                 num_classes=6, dropout=0.3):
        """
        Args:
            model_name: timm model name
            pretrained: Use ImageNet pretrained weights
            num_classes: Number of output classes (2 for binary)
            dropout: Dropout rate
        """
        super(ScratchDetector, self).__init__()
        
        # Load pretrained backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Get number of features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.num_features = features.shape[1]
        
        # Spatial Attention Module
        self.spatial_attention = SpatialAttention()
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        print(f"✓ Model created: {model_name}")
        print(f"  Features: {self.num_features}")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)  # [B, C, H, W]
        
        # Apply spatial attention
        attended_features = self.spatial_attention(features)
        
        # Global pooling
        pooled = self.global_pool(attended_features)  # [B, C, 1, 1]
        pooled = pooled.flatten(1)  # [B, C]
        
        # Classification
        output = self.classifier(pooled)
        
        return output
    
    def get_attention_map(self, x):
        """Get attention map for visualization"""
        with torch.no_grad():
            features = self.backbone(x)
            attention = self.spatial_attention.get_attention(features)
        return attention

class SpatialAttention(nn.Module):
    """Spatial attention to focus on scratch regions"""
    
    def __init__(self):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel-wise statistics
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Concatenate
        combined = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, H, W]
        
        # Generate attention map
        attention = self.conv(combined)  # [B, 1, H, W]
        
        # Apply attention
        return x * attention
    
    def get_attention(self, x):
        """Get attention map for visualization"""
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(combined)
        return attention

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor [0, 1]
            gamma: Focusing parameter (gamma >= 0)
            reduction: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, num_classes] logits
            targets: [B] class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_model(model_name='resnet50', pretrained=True, 
                 num_classes=6, device='cuda'):
    """Factory function to create model"""
    
    model = ScratchDetector(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    
    model = model.to(device)
    
    return model

def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    return total, trainable

if __name__ == "__main__":
    # Test model creation
    print("Testing model architecture...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = create_model(device=device)
    
    # Count parameters
    count_parameters(model)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test attention map
    attention = model.get_attention_map(dummy_input)
    print(f"Attention map shape: {attention.shape}")
    
    print("\n✓ Model working correctly!")