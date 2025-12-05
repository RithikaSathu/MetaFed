import torch
import torch.nn as nn
import copy
from typing import Dict, List
import numpy as np

class FedAvg:
    """Federated Averaging Algorithm"""
    
    def __init__(self, model_factory, config):
        self.config = config
        self.model_factory = model_factory
        self.global_model = None
        self.client_models = {}
        
    def initialize_global_model(self, model_type='cnn'):
        """Initialize global model"""
        self.global_model = self.model_factory.create_model(
            model_type, self.config
        ).to(self.config.DEVICE)
    
    def client_update(self, client_id, train_loader, epochs=5):
        """Perform local training on client"""
        if client_id not in self.client_models:
            self.client_models[client_id] = copy.deepcopy(self.global_model)
        
        model = self.client_models[client_id]
        model.train()
        
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            momentum=self.config.MOMENTUM,
            weight_decay=self.config.WEIGHT_DECAY
        )
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
            print(f'Client {client_id} - Epoch {epoch+1}: Loss: {total_loss/len(train_loader):.4f}, Acc: {accuracy:.2f}%')
        
        return model.state_dict()
    
    def server_aggregate(self, client_updates: Dict[int, Dict]):
        """Aggregate client updates using FedAvg"""
        if not client_updates:
            return self.global_model.state_dict()
        
        # Initialize averaged parameters
        avg_params = {}
        total_samples = sum([update['num_samples'] for update in client_updates.values()])
        
        # Weighted averaging
        for client_id, update in client_updates.items():
            weight = update['num_samples'] / total_samples
            params = update['params']
            
            for key in params.keys():
                if key not in avg_params:
                    avg_params[key] = params[key] * weight
                else:
                    avg_params[key] += params[key] * weight
        
        # Update global model
        self.global_model.load_state_dict(avg_params)
        return avg_params
    
    def train_round(self, federation_dataloaders, round_num):
        """Perform one round of federated training"""
        print(f"\n{'='*50}")
        print(f"FedAvg Round {round_num}")
        print(f"{'='*50}")
        
        client_updates = {}
        
        # Client updates
        for fed_id, dataloaders in federation_dataloaders.items():
            print(f"\nTraining Federation {fed_id}...")
            
            # Get number of samples
            num_samples = len(dataloaders['train'].dataset)
            
            # Local training
            client_params = self.client_update(
                fed_id, 
                dataloaders['train'], 
                epochs=self.config.LOCAL_EPOCHS
            )
            
            client_updates[fed_id] = {
                'params': client_params,
                'num_samples': num_samples
            }
        
        # Server aggregation
        global_params = self.server_aggregate(client_updates)
        
        # Update all client models with new global model
        for fed_id in self.client_models:
            self.client_models[fed_id].load_state_dict(global_params)
        
        return global_params
    
    def evaluate(self, federation_dataloaders):
        """Evaluate models on all federations"""
        results = {}
        
        for fed_id, dataloaders in federation_dataloaders.items():
            model = self.client_models.get(fed_id, self.global_model)
            model.eval()
            
            test_loader = dataloaders['test']
            criterion = nn.CrossEntropyLoss()
            
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                    output, _ = model(data)
                    test_loss += criterion(output, target).item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            accuracy = 100. * correct / total
            avg_loss = test_loss / len(test_loader)
            
            results[fed_id] = {
                'accuracy': accuracy,
                'loss': avg_loss,
                'correct': correct,
                'total': total
            }
            
            print(f'Federation {fed_id} - Test Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%')
        
        return results