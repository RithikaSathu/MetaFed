import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from typing import Dict

class MetaFedHeterogeneous(MetaFed):
    """MetaFed with Heterogeneous Models (CNN, RNN, ViT)"""
    
    def __init__(self, model_factory, config):
        super().__init__(model_factory, config)
        self.model_types = ['cnn', 'rnn', 'vit']  # Different models for different federations
    
    def initialize_models(self):
        """Initialize different model architectures for each federation"""
        for fed_id in range(self.config.NUM_FEDERATIONS):
            model_type = self.model_types[fed_id % len(self.model_types)]
            
            self.federation_models[fed_id] = {
                'model': self.model_factory.create_model(model_type, self.config).to(self.config.DEVICE),
                'model_type': model_type,
                'personalized': False
            }
        
        # Initialize common model (use CNN as common architecture)
        self.common_model = self.model_factory.create_model('cnn', self.config).to(self.config.DEVICE)
    
    def heterogeneous_knowledge_distillation(self, teacher_model, student_model, data, lambda_kd):
        """
        Knowledge distillation between heterogeneous models
        """
        # Forward pass through both models
        teacher_output, teacher_features = teacher_model(data)
        student_output, student_features = student_model(data)
        
        # Feature alignment loss (if feature dimensions differ)
        if teacher_features.shape != student_features.shape:
            # Use adaptive pooling or projection if needed
            if teacher_features.dim() == 3 and student_features.dim() == 2:
                # Teacher is RNN (seq, batch, features), Student is CNN/ViT (batch, features)
                teacher_features = teacher_features.mean(dim=0)  # Average over sequence
            elif teacher_features.dim() == 2 and student_features.dim() == 3:
                # Teacher is CNN/ViT, Student is RNN
                student_features = student_features.mean(dim=1)  # Average over sequence
            
            # If still different dimensions, use linear projection
            if teacher_features.shape[-1] != student_features.shape[-1]:
                projection = nn.Linear(student_features.shape[-1], teacher_features.shape[-1]).to(data.device)
                student_features = projection(student_features)
        
        # Feature distillation loss
        feature_loss = F.mse_loss(student_features, teacher_features)
        
        # Output distillation loss (soft targets)
        T = 3.0  # Temperature for softening
        teacher_soft = F.softmax(teacher_output / T, dim=1)
        student_soft = F.log_softmax(student_output / T, dim=1)
        
        output_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
        
        # Combined loss
        kd_loss = 0.7 * feature_loss + 0.3 * output_loss
        
        return kd_loss
    
    def train_federation_cyclic(self, federation_id, train_loader, val_loader, 
                               teacher_model=None, lambda_kd=5.0, stage='common'):
        """Train with heterogeneous model support"""
        model_info = self.federation_models[federation_id]
        model = model_info['model']
        model_type = model_info['model_type']
        model.train()
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        criterion_cls = nn.CrossEntropyLoss()
        
        # Get validation accuracy
        val_accuracy = self.evaluate_single(model, val_loader)
        
        for epoch in range(self.config.LOCAL_EPOCHS):
            total_loss = 0
            total_cls_loss = 0
            total_kd_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                
                optimizer.zero_grad()
                
                # Model-specific data formatting
                formatted_data = self._format_data_for_model(data, model_type)
                
                # Forward pass
                output, features = model(formatted_data)
                cls_loss = criterion_cls(output, target)
                
                # Heterogeneous knowledge distillation
                kd_loss = 0
                if teacher_model is not None and stage == 'common':
                    teacher_formatted_data = self._format_data_for_model(data, self.model_types[(federation_id-1) % 3])
                    kd_loss = self.heterogeneous_knowledge_distillation(
                        teacher_model, model, teacher_formatted_data, lambda_kd
                    )
                
                # Total loss
                if stage == 'common':
                    loss = cls_loss + lambda_kd * kd_loss
                else:
                    loss = cls_loss
                
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_kd_loss += kd_loss.item() if teacher_model is not None else 0
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            epoch_acc = 100. * correct / total
            print(f'Fed {federation_id} ({model_type.upper()}) | Stage: {stage} | '
                  f'Epoch {epoch+1}: Loss: {total_loss/len(train_loader):.4f}, '
                  f'Acc: {epoch_acc:.2f}%')
        
        return model.state_dict()
    
    def _format_data_for_model(self, data, model_type):
        """Format data according to model requirements"""
        if model_type == 'cnn':
            # CNN expects (batch, channels, sequence)
            return data
        elif model_type == 'rnn':
            # RNN expects (batch, sequence, features)
            return data.transpose(1, 2)
        elif model_type == 'vit':
            # ViT expects (batch, channels, sequence)
            # Note: Our ViT implementation converts to 2D internally
            return data
        else:
            return data
    
    def evaluate_single(self, model, dataloader):
        """Evaluate with model-specific data formatting"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                
                # Get model type from model name
                model_type = 'cnn'
                if 'RNN' in model.__class__.__name__:
                    model_type = 'rnn'
                elif 'VisionTransformer' in model.__class__.__name__:
                    model_type = 'vit'
                
                # Format data
                formatted_data = self._format_data_for_model(data, model_type)
                
                output, _ = model(formatted_data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy