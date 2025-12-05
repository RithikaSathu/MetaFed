import torch.nn as nn
from .cnn_model import CNNModel
from .rnn_model import RNNModel
from .vit_model import VisionTransformer

class ModelFactory:
    @staticmethod
    def create_model(model_type, config, num_classes=12):
        """
        Factory method to create different model architectures
        """
        if model_type == 'cnn':
            return CNNModel(config['CNN_CONFIG'], num_classes)
        elif model_type == 'rnn':
            return RNNModel(config['RNN_CONFIG'], num_classes)
        elif model_type == 'vit':
            return VisionTransformer(config['VIT_CONFIG'], num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_configs():
        """
        Return default configurations for all models
        """
        from config import Config
        return {
            'cnn': Config.CNN_CONFIG,
            'rnn': Config.RNN_CONFIG,
            'vit': Config.VIT_CONFIG
        }