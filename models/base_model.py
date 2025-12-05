import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class BaseModel(nn.Module, ABC):
    """Base model class for all models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.feature_extractor = None
        self.classifier = None
        
    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input"""
        pass
    
    @abstractmethod
    def forward_classifier(self, features: torch.Tensor) -> torch.Tensor:
        """Classify from features"""
        pass
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both features and logits"""
        features = self.forward_features(x)
        logits = self.forward_classifier(features)
        return features, logits
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters"""
        return {name: param.data.clone() for name, param in self.named_parameters()}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters"""
        for name, param in self.named_parameters():
            if name in parameters:
                param.data = parameters[name].clone()
    
    def to_device(self, device: str):
        """Move model to device"""
        return self.to(device)