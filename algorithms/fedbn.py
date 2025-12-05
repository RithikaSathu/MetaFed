import torch
import torch.nn as nn
import copy
from typing import Dict
from .fedavg import FedAvg

class FedBN(FedAvg):
    """FedBN: Federated Learning with Batch Normalization"""
    
    def __init__(self, model_factory, config):
        super().__init__(model_factory, config)
    
    def client_update(self, client_id, train_loader, epochs=5):
        """Perform local training while keeping BN layers local"""
        if client_id not in self.client_models:
            self.client_models[client_id] = copy.deepcopy(self.global_model)
        
        model = self.client_models[client_id]
        model.train()
        
        # Get parameters excluding BN layers for global updates
        bn_params = []
        non_bn_params = []
        
        for name, param in model.named_parameters():
            if 'bn' in name or 'batch_norm' in name:
                bn_params.append(param)
            else:
                non_bn_params.append(param)
        
        # Only optimize non-BN parameters with SGD
        optimizer = torch.optim.SGD(
            non_bn_params,
            lr=self.config.LEARNING_RATE,
            momentum=self.config.MOMENTUM,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # BN parameters are updated during forward pass automatically
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                
                optimizer.zero_grad()
                output, _ = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            accuracy = 100. * correct / total
            print(f'FedBN Client {client_id} - Epoch {epoch+1}: Loss: {total_loss/len(train_loader):.4f}, Acc: {accuracy:.2f}%')
        
        # Return only non-BN parameters for aggregation
        state_dict = model.state_dict()
        bn_state_dict = {}
        
        # Separate BN parameters
        for key in list(state_dict.keys()):
            if 'bn' in key or 'batch_norm' in key:
                bn_state_dict[key] = state_dict[key]
                del state_dict[key]  # Remove BN params from aggregation
        
        return {
            'global_params': state_dict,
            'bn_params': bn_state_dict
        }
    
    def server_aggregate(self, client_updates: Dict[int, Dict]):
        """Aggregate only non-BN parameters"""
        if not client_updates:
            return self.global_model.state_dict()
        
        # Aggregate global parameters (non-BN)
        avg_global_params = {}
        total_samples = sum([update['num_samples'] for update in client_updates.values()])
        
        for client_id, update in client_updates.items():
            weight = update['num_samples'] / total_samples
            params = update['params']['global_params']
            
            for key in params.keys():
                if key not in avg_global_params:
                    avg_global_params[key] = params[key] * weight
                else:
                    avg_global_params[key] += params[key] * weight
        
        # Update global model with aggregated parameters
        global_state_dict = self.global_model.state_dict()
        global_state_dict.update(avg_global_params)
        self.global_model.load_state_dict(global_state_dict)
        
        return global_state_dict
    
    def update_client_models(self, client_updates):
        """Update client models with aggregated global params + local BN params"""
        for client_id, update in client_updates.items():
            if client_id in self.client_models:
                # Get current model state dict
                model_state = self.client_models[client_id].state_dict()
                
                # Update with global parameters
                model_state.update(self.global_model.state_dict())
                
                # Restore local BN parameters
                if 'bn_params' in update['params']:
                    model_state.update(update['params']['bn_params'])
                
                # Load updated state dict
                self.client_models[client_id].load_state_dict(model_state)