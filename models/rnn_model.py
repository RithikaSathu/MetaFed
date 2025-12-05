import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, config, num_classes=12):
        super(RNNModel, self).__init__()
        
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=52,  # PAMAP2 features
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=config['dropout'] if self.num_layers > 1 else 0
        )
        
        # Calculate LSTM output size
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(config['dropout'])
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        # lstm_out shape: (batch_size, seq_length, hidden_size * num_directions)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # Apply attention weights
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(context))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x, context  # Return logits and context vector