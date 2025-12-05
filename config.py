import torch

class Config:
    # GPU Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Experiment Configuration
    NUM_FEDERATIONS = 3
    NUM_CLIENTS_PER_FED = 3
    NUM_ROUNDS = 50
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    
    # MetaFed Specific
    LAMBDA_0 = 5.0  # Initial knowledge distillation weight
    L_T1 = 0.4  # Threshold for validation accuracy
    CYCLIC_ROUNDS = 3  # Number of cyclic rounds
    
    # Dataset Configuration
    PAMAP2_DATA_PATH = '../data/PAMAP2_Dataset/Protocol'
    NUM_CLASSES = 12  # For PAMAP2
    INPUT_CHANNELS = 52  # PAMAP2 features
    SEQUENCE_LENGTH = 128
    
    # Model Configurations
    CNN_CONFIG = {
        'channels': [52, 64, 128],
        'kernel_sizes': [3, 3],
        'pool_sizes': [2, 2],
        'fc_sizes': [256, 128]
    }
    
    RNN_CONFIG = {
        'hidden_size': 128,
        'num_layers': 2,
        'bidirectional': True,
        'dropout': 0.3
    }
    
    VIT_CONFIG = {
        'image_size': 128,
        'patch_size': 16,
        'num_classes': 12,
        'dim': 128,
        'depth': 6,
        'heads': 8,
        'mlp_dim': 256,
        'channels': 52
    }
    
    # Training
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.2
    
    # Evaluation
    METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix']
    
    # Paths
    LOG_DIR = './logs/'
    MODEL_SAVE_DIR = './logs/models/'
    
    # Communication
    COMMUNICATION_ROUNDS = 10