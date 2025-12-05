import json
import os
from datetime import datetime
from typing import Dict, Any

class TrainingLogger:
    def __init__(self, log_dir='./logs/training_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_log_{timestamp}.json')
        
        # Initialize log structure
        self.log_data = {
            'start_time': timestamp,
            'experiments': [],
            'metrics': {}
        }
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data
        }
        
        self.log_data['experiments'].append(event)
        self._save_log()
    
    def log_round(self, algorithm: str, round_num: int, metrics: Dict):
        """Log a training round"""
        round_log = {
            'algorithm': algorithm,
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        if algorithm not in self.log_data['metrics']:
            self.log_data['metrics'][algorithm] = []
        
        self.log_data['metrics'][algorithm].append(round_log)
        self._save_log()
    
    def log_model(self, algorithm: str, model_info: Dict):
        """Log model information"""
        model_log = {
            'algorithm': algorithm,
            'timestamp': datetime.now().isoformat(),
            'model_info': model_info
        }
        
        self.log_data['experiments'].append(model_log)
        self._save_log()
    
    def _save_log(self):
        """Save log to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=4)
    
    def get_summary(self):
        """Get experiment summary"""
        summary = {
            'total_experiments': len(self.log_data['experiments']),
            'algorithms_trained': list(self.log_data['metrics'].keys()),
            'log_file': self.log_file
        }
        
        # Add metrics summary for each algorithm
        for algo, rounds in self.log_data['metrics'].items():
            if rounds:
                last_round = rounds[-1]
                summary[f'{algo}_last_round'] = last_round
        
        return summary