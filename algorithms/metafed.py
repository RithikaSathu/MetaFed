import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from typing import Dict, List, Tuple

class MetaFed:
    """MetaFed: Federated Learning among Federations with Cyclic Knowledge Distillation"""
    
    def __init__(self, model_factory, config):
        self.config = config
        self.model_factory = model_factory
        self.federation_models = {}
        self.common_model = None
        self.current_round = 0
        
    def initialize_models(self, model_type='cnn'):
        """Initialize models for all federations"""
        for fed_id in range(self.config.NUM_FEDERATIONS):
            self.federation_models[fed_id] = {
                'model': self.model_factory.create_model(model_type, self.config).to(self.config.DEVICE),
                'personalized': False
            }
        
        # Initialize common model
        self.common_model = self.model_factory.create_model(model_type, self.config).to(self.config.DEVICE)
    
    def knowledge_distillation_loss(self, teacher_model, student_model, data, lambda_kd):
        """Compute knowledge distillation loss"""
        # Get features from teacher and student
        with torch.no_grad():
            _, teacher_features = teacher_model(data)
        
        _, student_features = student_model(data)
        
        # Feature distillation loss (L2 distance)
        kd_loss = F.mse_loss(student_features, teacher_features)
        
        return kd_loss
    
    def train_federation_cyclic(self, federation_id, train_loader, val_loader, 
                               teacher_model=None, lambda_kd=5.0, stage='common'):
        """Train a single federation with knowledge distillation"""
        model_info = self.federation_models[federation_id]
        model = model_info['model']
        model.train()
        
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            momentum=self.config.MOMENTUM,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        criterion_cls = nn.CrossEntropyLoss()
        
        # Validation accuracy for threshold
        val_accuracy = self.evaluate_single(model, val_loader)
        
        # Determine if we need to initialize from teacher
        if teacher_model is not None and val_accuracy < self.config.L_T1 * 100:
            print(f"Low validation accuracy ({val_accuracy:.2f}%), initializing from teacher...")
            model.load_state_dict(teacher_model.state_dict())
        
        for epoch in range(self.config.LOCAL_EPOCHS):
            total_loss = 0
            total_cls_loss = 0
            total_kd_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                
                optimizer.zero_grad()
                
                # Forward pass
                output, features = model(data)
                cls_loss = criterion_cls(output, target)
                
                # Knowledge distillation if teacher is provided
                kd_loss = 0
                if teacher_model is not None and stage == 'common':
                    kd_loss = self.knowledge_distillation_loss(teacher_model, model, data, lambda_kd)
                
                # Total loss
                if stage == 'common':
                    loss = cls_loss + lambda_kd * kd_loss
                else:  # personalization stage
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
            print(f'Fed {federation_id} | Stage: {stage} | Epoch {epoch+1}: '
                  f'Loss: {total_loss/len(train_loader):.4f}, '
                  f'Cls: {total_cls_loss/len(train_loader):.4f}, '
                  f'KD: {total_kd_loss/len(train_loader):.4f}, '
                  f'Acc: {epoch_acc:.2f}%')
        
        return model.state_dict()
    
    def common_knowledge_accumulation(self, federation_dataloaders):
        """Stage I: Common Knowledge Accumulation"""
        print(f"\n{'='*60}")
        print("STAGE I: COMMON KNOWLEDGE ACCUMULATION")
        print(f"{'='*60}")
        
        # Perform cyclic training for multiple rounds
        for round_idx in range(self.config.CYCLIC_ROUNDS):
            print(f"\nCyclic Round {round_idx + 1}/{self.config.CYCLIC_ROUNDS}")
            print("-" * 40)
            
            # Train federations in cyclic order
            for i in range(self.config.NUM_FEDERATIONS):
                current_fed = i
                previous_fed = (i - 1) % self.config.NUM_FEDERATIONS
                
                print(f"\nTraining Federation {current_fed} with teacher from Federation {previous_fed}")
                
                # Get teacher model (previous federation)
                teacher_model = self.federation_models[previous_fed]['model'] if round_idx > 0 or previous_fed != current_fed else None
                
                # Train current federation
                fed_dataloaders = federation_dataloaders[current_fed]
                self.train_federation_cyclic(
                    federation_id=current_fed,
                    train_loader=fed_dataloaders['train'],
                    val_loader=fed_dataloaders['val'],
                    teacher_model=teacher_model,
                    lambda_kd=self.config.LAMBDA_0,
                    stage='common'
                )
            
            # Update common model (average of all federation models)
            self._update_common_model()
    
    def personalization_stage(self, federation_dataloaders):
        """Stage II: Personalization Stage"""
        print(f"\n{'='*60}")
        print("STAGE II: PERSONALIZATION")
        print(f"{'='*60}")
        
        # Initialize all models with common knowledge
        common_state = self.common_model.state_dict()
        for fed_id in self.federation_models:
            self.federation_models[fed_id]['model'].load_state_dict(common_state)
        
        # Personalize each federation
        for fed_id in range(self.config.NUM_FEDERATIONS):
            print(f"\nPersonalizing Federation {fed_id}")
            print("-" * 40)
            
            fed_dataloaders = federation_dataloaders[fed_id]
            
            # Train without knowledge distillation (focus on local data)
            self.train_federation_cyclic(
                federation_id=fed_id,
                train_loader=fed_dataloaders['train'],
                val_loader=fed_dataloaders['val'],
                teacher_model=None,  # No teacher in personalization
                lambda_kd=0.0,
                stage='personalization'
            )
            
            # Mark as personalized
            self.federation_models[fed_id]['personalized'] = True
    
    def _update_common_model(self):
        """Update common model by averaging all federation models"""
        print("\nUpdating common model...")
        
        # Collect all model parameters
        all_params = []
        for fed_id in self.federation_models:
            model_state = self.federation_models[fed_id]['model'].state_dict()
            all_params.append(model_state)
        
        # Average parameters
        avg_params = {}
        for key in all_params[0].keys():
            avg_params[key] = torch.stack([params[key] for params in all_params]).mean(0)
        
        # Update common model
        self.common_model.load_state_dict(avg_params)
    
    def evaluate_single(self, model, dataloader):
        """Evaluate a single model"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                output, _ = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def evaluate(self, federation_dataloaders):
        """Evaluate all federation models"""
        results = {}
        
        for fed_id, dataloaders in federation_dataloaders.items():
            model = self.federation_models[fed_id]['model']
            test_loader = dataloaders['test']
            
            accuracy = self.evaluate_single(model, test_loader)
            
            results[fed_id] = {
                'accuracy': accuracy,
                'personalized': self.federation_models[fed_id]['personalized']
            }
            
            print(f'MetaFed Federation {fed_id} - Test Acc: {accuracy:.2f}% '
                  f'({"Personalized" if self.federation_models[fed_id]["personalized"] else "Common"})')
        
        # Also evaluate common model
        if self.common_model is not None:
            common_accuracies = []
            for fed_id, dataloaders in federation_dataloaders.items():
                acc = self.evaluate_single(self.common_model, dataloaders['test'])
                common_accuracies.append(acc)
            
            avg_common_acc = np.mean(common_accuracies)
            print(f'MetaFed Common Model - Avg Test Acc: {avg_common_acc:.2f}%')
            results['common'] = {'accuracy': avg_common_acc}
        
        return results
    
    def train(self, federation_dataloaders):
        """Complete MetaFed training"""
        # Stage I: Common Knowledge Accumulation
        self.common_knowledge_accumulation(federation_dataloaders)
        
        # Stage II: Personalization
        self.personalization_stage(federation_dataloaders)
        
        return self.federation_models