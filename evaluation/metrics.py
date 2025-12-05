import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsCalculator:
    """Calculate various evaluation metrics"""
    
    def __init__(self, num_classes=12):
        self.num_classes = num_classes
    
    def calculate_all_metrics(self, model, dataloader, device):
        """Calculate all metrics for a model"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                _, preds = output.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision_macro': precision_score(all_targets, all_preds, average='macro', zero_division=0),
            'recall_macro': recall_score(all_targets, all_preds, average='macro', zero_division=0),
            'f1_macro': f1_score(all_targets, all_preds, average='macro', zero_division=0),
            'precision_weighted': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
            'recall_weighted': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
            'f1_weighted': f1_score(all_targets, all_preds, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(all_targets, all_preds).tolist()
        }
        
        return metrics
    
    def compare_algorithms(self, algorithm_results):
        """Compare results from different algorithms"""
        comparison = {}
        
        for algo_name, fed_results in algorithm_results.items():
            fed_accuracies = []
            fed_metrics = []
            
            for fed_id, metrics in fed_results.items():
                if fed_id != 'common':  # Exclude common model if present
                    fed_accuracies.append(metrics.get('accuracy', 0))
                    fed_metrics.append({
                        'federation': fed_id,
                        'accuracy': metrics.get('accuracy', 0),
                        'f1_score': metrics.get('f1_macro', 0)
                    })
            
            comparison[algo_name] = {
                'average_accuracy': np.mean(fed_accuracies) if fed_accuracies else 0,
                'std_accuracy': np.std(fed_accuracies) if fed_accuracies else 0,
                'max_accuracy': np.max(fed_accuracies) if fed_accuracies else 0,
                'min_accuracy': np.min(fed_accuracies) if fed_accuracies else 0,
                'federation_metrics': fed_metrics
            }
        
        return comparison
    
    def save_metrics(self, metrics, filepath):
        """Save metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def load_metrics(self, filepath):
        """Load metrics from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def plot_confusion_matrix(self, cm, class_names=None, title='Confusion Matrix'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt