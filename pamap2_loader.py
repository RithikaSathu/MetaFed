import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from scipy import stats

class PAMAP2Dataset(Dataset):
    """PAMAP2 Dataset Loader"""
    
    def __init__(self, data_path, subjects=None, activities=None, transform=None):
        self.data_path = data_path
        self.transform = transform
        
        # Activity labels mapping (we'll use 12 activities)
        self.activity_map = {
            1: 'lying',
            2: 'sitting',
            3: 'standing',
            4: 'walking',
            5: 'running',
            6: 'cycling',
            7: 'Nordic walking',
            9: 'watching TV',
            10: 'computer work',
            11: 'car driving',
            12: 'ascending stairs',
            13: 'descending stairs',
            16: 'vacuum cleaning',
            17: 'ironing',
            18: 'folding laundry',
            19: 'house cleaning',
            20: 'playing soccer',
            24: 'rope jumping'
        }
        
        # Keep only the 12 main activities
        self.selected_activities = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
        self.activity_to_idx = {act: idx for idx, act in enumerate(self.selected_activities)}
        
        self.data, self.labels = self._load_data(subjects)
    
    def _load_data(self, subjects):
        all_data = []
        all_labels = []
        
        if subjects is None:
            subjects = range(1, 10)  # Subjects 1-9
        
        for subject in subjects:
            file_path = os.path.join(self.data_path, f'subject10{subject}.dat')
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found")
                continue
            
            # Read data
            df = pd.read_csv(file_path, sep=' ', header=None)
            
            # Select relevant columns: activity label + IMU data
            # Columns: 1=activity, 4-20=hand IMU, 21-37=chest IMU, 38-54=ankle IMU
            activity_col = 1
            imu_columns = list(range(4, 55))  # All IMU data columns
            
            # Extract data
            subject_data = df.iloc[:, imu_columns].values
            subject_labels = df.iloc[:, activity_col].values
            
            # Filter selected activities
            mask = np.isin(subject_labels, self.selected_activities)
            subject_data = subject_data[mask]
            subject_labels = subject_labels[mask]
            
            # Map labels to indices
            subject_labels = np.array([self.activity_to_idx[label] for label in subject_labels])
            
            # Normalize data
            subject_data = self._normalize_data(subject_data)
            
            # Segment data using sliding window
            window_size = 128
            step_size = 64
            
            for i in range(0, len(subject_data) - window_size + 1, step_size):
                window_data = subject_data[i:i+window_size]
                window_label = stats.mode(subject_labels[i:i+window_size])[0][0]
                
                # Only include windows where most labels are the same
                if len(np.unique(subject_labels[i:i+window_size])) == 1:
                    all_data.append(window_data.T)  # Transpose to (channels, seq_len)
                    all_labels.append(window_label)
        
        return np.array(all_data), np.array(all_labels)
    
    def _normalize_data(self, data):
        """Normalize sensor data"""
        # Handle NaN values
        data = np.nan_to_num(data, nan=0.0)
        
        # Normalize each feature independently
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        
        normalized_data = (data - mean) / std
        return normalized_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        
        if self.transform:
            data = self.transform(data)
        
        return data, label

def split_data_by_subject(data_path, num_federations=3):
    """
    Split PAMAP2 data by subject to simulate different federations
    """
    # Group subjects into federations
    subject_groups = {
        0: [1, 2, 3],  # Federation 0
        1: [4, 5, 6],  # Federation 1
        2: [7, 8, 9]   # Federation 2
    }
    
    federation_datasets = {}
    
    for fed_id, subjects in subject_groups.items():
        dataset = PAMAP2Dataset(data_path, subjects=subjects)
        
        # Split into train, validation, test
        total_size = len(dataset)
        train_size = int(0.6 * total_size)
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        federation_datasets[fed_id] = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    
    return federation_datasets

def create_dataloaders(federation_datasets, batch_size=64):
    """
    Create dataloaders for each federation
    """
    dataloaders = {}
    
    for fed_id, datasets in federation_datasets.items():
        fed_dataloaders = {
            'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True),
            'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=False),
            'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False)
        }
        dataloaders[fed_id] = fed_dataloaders
    
    return dataloaders