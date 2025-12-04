import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any
import copy
import numpy as np
from .fedavg import FedAvg
from ..utils.metrics import calculate_metrics

class FedProx(FedAvg):
    """FedProx: Federated Learning with Proximal Term"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mu = config.get("mu", 0.01)  # Proximal term weight
        
    def train_client(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        global_model: nn.Module = None,
        epochs: int = 3,
        lr: float = 0.001
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Train model on client data with proximal term
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data
            global_model: Reference global model for proximal term
            epochs: Number of local epochs
            lr: Learning rate
            
        Returns:
            Tuple of (model_state_dict, training_metrics)
        """
        model = model.to(self.device)
        model.train()
        
        if global_model is not None:
            global_model = global_model.to(self.device)
            global_params = {n: p.detach() for n, p in global_model.named_parameters()}
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        metrics = {
            "train_loss": [],
            "train_acc": []
        }
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                _, output = model(data)
                loss = criterion(output, target)
                
                # Add proximal term
                if global_model is not None:
                    proximal_term = 0.0
                    for n, p in model.named_parameters():
                        if n in global_params:
                            proximal_term += torch.norm(p - global_params[n]) ** 2
                    loss += self.mu / 2 * proximal_term
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()
                total += target.size(0)
                
                epoch_loss += loss.item()
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100. * correct / total if total > 0 else 0
            
            metrics["train_loss"].append(avg_loss)
            metrics["train_acc"].append(accuracy)
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
        
        # Get model parameters
        model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        # Track communication cost
        param_size = sum(p.numel() for p in model.parameters())
        self.communication_tracker.add_upload(param_size * 4)
        
        return model_state, metrics
    
    def run_federation(
        self,
        federation_id: int,
        model: nn.Module,
        client_loaders: List[torch.utils.data.DataLoader],
        test_loader: torch.utils.data.DataLoader,
        num_rounds: int = 10,
        local_epochs: int = 3,
        lr: float = 0.001
    ) -> Dict[str, Any]:
        """
        Run FedProx for a federation
        
        Args:
            federation_id: ID of the federation
            model: Global model
            client_loaders: List of DataLoaders for each client
            test_loader: DataLoader for test data
            num_rounds: Number of federation rounds
            local_epochs: Number of local epochs
            lr: Learning rate
            
        Returns:
            Dictionary with results
        """
        print(f"\n{'='*50}")
        print(f"Running FedProx for Federation {federation_id}")
        print(f"{'='*50}")
        
        # Results storage
        results = {
            "round_losses": [],
            "round_accuracies": [],
            "client_metrics": [],
            "test_metrics": [],
            "communication_costs": []
        }
        
        global_model = copy.deepcopy(model)
        
        for round_idx in range(num_rounds):
            print(f"\nRound {round_idx + 1}/{num_rounds}")
            
            # Client training
            client_models = []
            client_weights = []
            round_client_metrics = []
            
            for client_idx, train_loader in enumerate(client_loaders):
                print(f"\n  Training Client {client_idx + 1}/{len(client_loaders)}")
                
                # Local training with proximal term
                client_model = copy.deepcopy(global_model)
                client_state, client_metrics = self.train_client(
                    client_model, train_loader, global_model=global_model,
                    epochs=local_epochs, lr=lr
                )
                
                client_models.append(client_state)
                client_weights.append(len(train_loader.dataset))
                round_client_metrics.append(client_metrics)
            
            # Model aggregation
            print("\n  Aggregating client models...")
            aggregated_state = self.aggregate(client_models, client_weights)
            global_model.load_state_dict(aggregated_state)
            
            # Evaluate on test data
            test_metrics = self.evaluate(global_model, test_loader)
            
            # Store results
            avg_client_loss = np.mean([m["train_loss"][-1] for m in round_client_metrics])
            avg_client_acc = np.mean([m["train_acc"][-1] for m in round_client_metrics])
            
            results["round_losses"].append(avg_client_loss)
            results["round_accuracies"].append(avg_client_acc)
            results["client_metrics"].append(round_client_metrics)
            results["test_metrics"].append(test_metrics)
            
            # Track communication
            comm_cost = self.communication_tracker.get_total_cost()
            results["communication_costs"].append(comm_cost)
            
            print(f"  Round {round_idx + 1} Results:")
            print(f"    Client Loss: {avg_client_loss:.4f}")
            print(f"    Client Acc: {avg_client_acc:.2f}%")
            print(f"    Test Acc: {test_metrics.get('accuracy', 0):.2f}%")
            print(f"    Communication Cost: {comm_cost / 1e6:.2f} MB")
        
        return results