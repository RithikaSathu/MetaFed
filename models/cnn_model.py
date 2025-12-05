import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, config, num_classes=12):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv1d(config['channels'][0], config['channels'][1], 
                              kernel_size=config['kernel_sizes'][0], padding=1)
        self.bn1 = nn.BatchNorm1d(config['channels'][1])
        self.conv2 = nn.Conv1d(config['channels'][1], config['channels'][2], 
                              kernel_size=config['kernel_sizes'][1], padding=1)
        self.bn2 = nn.BatchNorm1d(config['channels'][2])
        
        self.pool = nn.MaxPool1d(config['pool_sizes'][0])
        
        # Calculate the size after convolutions and pooling
        self._to_linear = None
        self._get_conv_output((1, config['channels'][0], 128))
        
        self.fc1 = nn.Linear(self._to_linear, config['fc_sizes'][0])
        self.fc2 = nn.Linear(config['fc_sizes'][0], config['fc_sizes'][1])
        self.fc3 = nn.Linear(config['fc_sizes'][1], num_classes)
        
        self.dropout = nn.Dropout(0.3)
    
    def _get_conv_output(self, shape):
        batch_size = 1
        input_tensor = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output = self._forward_features(input_tensor)
        self._to_linear = output.data.view(batch_size, -1).size(1)
        return output
    
    def _forward_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        return x
    
    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        features = self._forward_features(x)
        features = features.view(features.size(0), -1)
        
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x, features
    
    def get_features(self, x):
        with torch.no_grad():
            features = self._forward_features(x)
            features = features.view(features.size(0), -1)
        return features