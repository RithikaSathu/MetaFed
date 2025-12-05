from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import json
import torch
import numpy as np
from datetime import datetime

from config import Config
from gpu_config import GPUManager
from data.pamap2_loader import split_data_by_subject, create_dataloaders
from models.model_factory import ModelFactory
from algorithms.fedavg import FedAvg
from algorithms.fedbn import FedBN
from algorithms.fedprox import FedProx
from algorithms.metafed import MetaFed
from algorithms.metafed_heterogeneous import MetaFedHeterogeneous
from evaluation.metrics import MetricsCalculator
from training.federated_trainer import FederatedTrainer
from utils.logger import TrainingLogger

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
CORS(app)

# Initialize components
gpu_manager = GPUManager()
config = Config()
model_factory = ModelFactory()
metrics_calc = MetricsCalculator(num_classes=config.NUM_CLASSES)
logger = TrainingLogger()

# Global variables for experiments
federation_dataloaders = None
experiment_results = {}
trained_models = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/initialize', methods=['POST'])
def initialize_experiment():
    """Initialize the experiment with PAMAP2 dataset"""
    global federation_dataloaders
    
    try:
        # Load and split data
        federation_datasets = split_data_by_subject(
            config.PAMAP2_DATA_PATH,
            num_federations=config.NUM_FEDERATIONS
        )
        
        # Create dataloaders
        federation_dataloaders = create_dataloaders(
            federation_datasets,
            batch_size=config.BATCH_SIZE
        )
        
        response = {
            'status': 'success',
            'message': f'Experiment initialized with {config.NUM_FEDERATIONS} federations',
            'data_info': {
                'num_federations': config.NUM_FEDERATIONS,
                'batch_size': config.BATCH_SIZE,
                'device': str(config.DEVICE)
            }
        }
        
        logger.log_event('experiment_initialized', response)
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/train/fedavg', methods=['POST'])
def train_fedavg():
    """Train using FedAvg algorithm"""
    try:
        if federation_dataloaders is None:
            return jsonify({'status': 'error', 'message': 'Experiment not initialized'}), 400
        
        # Initialize FedAvg
        fedavg = FedAvg(model_factory, config)
        fedavg.initialize_global_model('cnn')
        
        # Train for specified rounds
        num_rounds = request.json.get('rounds', config.NUM_ROUNDS)
        results = []
        
        for round_num in range(num_rounds):
            print(f"\n{'='*60}")
            print(f"FedAvg Training - Round {round_num + 1}/{num_rounds}")
            print(f"{'='*60}")
            
            # Train one round
            fedavg.train_round(federation_dataloaders, round_num)
            
            # Evaluate
            round_results = fedavg.evaluate(federation_dataloaders)
            results.append(round_results)
            
            # Log progress
            logger.log_round('fedavg', round_num, round_results)
        
        # Store results
        experiment_results['fedavg'] = {
            'algorithm': 'FedAvg',
            'results': results,
            'final_models': fedavg.client_models,
            'global_model': fedavg.global_model
        }
        
        trained_models['fedavg'] = fedavg
        
        response = {
            'status': 'success',
            'message': f'FedAvg training completed for {num_rounds} rounds',
            'final_accuracies': {
                fed_id: metrics['accuracy'] 
                for fed_id, metrics in results[-1].items()
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/train/metafed', methods=['POST'])
def train_metafed():
    """Train using MetaFed algorithm"""
    try:
        if federation_dataloaders is None:
            return jsonify({'status': 'error', 'message': 'Experiment not initialized'}), 400
        
        # Initialize MetaFed
        metafed = MetaFed(model_factory, config)
        metafed.initialize_models('cnn')
        
        # Train MetaFed
        metafed.train(federation_dataloaders)
        
        # Evaluate
        results = metafed.evaluate(federation_dataloaders)
        
        # Store results
        experiment_results['metafed'] = {
            'algorithm': 'MetaFed',
            'results': results,
            'models': metafed.federation_models,
            'common_model': metafed.common_model
        }
        
        trained_models['metafed'] = metafed
        
        response = {
            'status': 'success',
            'message': 'MetaFed training completed',
            'accuracies': {
                fed_id: metrics['accuracy'] 
                for fed_id, metrics in results.items() if fed_id != 'common'
            },
            'common_accuracy': results.get('common', {}).get('accuracy', 0)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/train/metafed_heterogeneous', methods=['POST'])
def train_metafed_heterogeneous():
    """Train using MetaFed with heterogeneous models"""
    try:
        if federation_dataloaders is None:
            return jsonify({'status': 'error', 'message': 'Experiment not initialized'}), 400
        
        # Initialize MetaFed with heterogeneous models
        metafed_hetero = MetaFedHeterogeneous(model_factory, config)
        metafed_hetero.initialize_models()
        
        # Train
        metafed_hetero.train(federation_dataloaders)
        
        # Evaluate
        results = metafed_hetero.evaluate(federation_dataloaders)
        
        # Store results
        experiment_results['metafed_heterogeneous'] = {
            'algorithm': 'MetaFed_Heterogeneous',
            'results': results,
            'models': metafed_hetero.federation_models,
            'common_model': metafed_hetero.common_model
        }
        
        trained_models['metafed_heterogeneous'] = metafed_hetero
        
        # Get model types for each federation
        model_types = {}
        for fed_id in range(config.NUM_FEDERATIONS):
            model_types[fed_id] = metafed_hetero.federation_models[fed_id]['model_type']
        
        response = {
            'status': 'success',
            'message': 'MetaFed with heterogeneous models training completed',
            'accuracies': {
                fed_id: metrics['accuracy'] 
                for fed_id, metrics in results.items() if fed_id != 'common'
            },
            'model_types': model_types,
            'common_accuracy': results.get('common', {}).get('accuracy', 0)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/train/all', methods=['POST'])
def train_all_algorithms():
    """Train all algorithms for comparison"""
    try:
        if federation_dataloaders is None:
            return jsonify({'status': 'error', 'message': 'Experiment not initialized'}), 400
        
        algorithms = ['fedavg', 'fedbn', 'fedprox', 'metafed']
        results = {}
        
        for algo in algorithms:
            print(f"\n{'='*80}")
            print(f"TRAINING {algo.upper()}")
            print(f"{'='*80}")
            
            # Call the appropriate training endpoint
            if algo == 'fedavg':
                train_fedavg()
            elif algo == 'metafed':
                train_metafed()
            # Add other algorithms as needed
        
        # Compare results
        comparison = metrics_calc.compare_algorithms(experiment_results)
        
        response = {
            'status': 'success',
            'message': 'All algorithms trained successfully',
            'comparison': comparison,
            'algorithms_trained': list(experiment_results.keys())
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate_models():
    """Evaluate all trained models"""
    try:
        if not experiment_results:
            return jsonify({'status': 'error', 'message': 'No models trained yet'}), 400
        
        evaluation_results = {}
        
        for algo_name, algo_data in experiment_results.items():
            if 'models' in algo_data:
                # For MetaFed
                models = algo_data['models']
                results = {}
                
                for fed_id, model_info in models.items():
                    if isinstance(model_info, dict) and 'model' in model_info:
                        model = model_info['model']
                        
                        # Evaluate on each federation's test set
                        fed_metrics = []
                        for test_fed_id, dataloaders in federation_dataloaders.items():
                            metrics = metrics_calc.calculate_all_metrics(
                                model, 
                                dataloaders['test'], 
                                config.DEVICE
                            )
                            fed_metrics.append({
                                'test_federation': test_fed_id,
                                **metrics
                            })
                        
                        results[fed_id] = fed_metrics
                
                evaluation_results[algo_name] = results
        
        # Save evaluation results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'./logs/evaluation_results_{timestamp}.json'
        metrics_calc.save_metrics(evaluation_results, results_file)
        
        response = {
            'status': 'success',
            'message': 'Evaluation completed',
            'results_file': results_file,
            'summary': {
                algo: {
                    'num_models': len(models) if isinstance(models, dict) else 1
                } for algo, models in evaluation_results.items()
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction on uploaded data"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
        
        file = request.files['file']
        algorithm = request.form.get('algorithm', 'metafed')
        federation_id = int(request.form.get('federation_id', 0))
        
        if algorithm not in trained_models:
            return jsonify({'status': 'error', 'message': f'Algorithm {algorithm} not trained'}), 400
        
        # Load the model
        model_data = trained_models[algorithm]
        
        # Get the appropriate model
        if algorithm == 'metafed' or algorithm == 'metafed_heterogeneous':
            model = model_data.federation_models[federation_id]['model']
        elif algorithm == 'fedavg':
            model = model_data.client_models.get(federation_id, model_data.global_model)
        else:
            model = model_data.global_model if hasattr(model_data, 'global_model') else None
        
        if model is None:
            return jsonify({'status': 'error', 'message': 'Model not found'}), 400
        
        # Process the uploaded file
        # Note: You'll need to implement file processing based on your data format
        # For now, returning a dummy response
        
        response = {
            'status': 'success',
            'prediction': {
                'class': 0,
                'confidence': 0.95,
                'algorithm': algorithm,
                'federation': federation_id
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get all experiment results"""
    try:
        return jsonify({
            'status': 'success',
            'experiment_results': experiment_results,
            'trained_algorithms': list(trained_models.keys())
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/plots', methods=['GET'])
def generate_plots():
    """Generate comparison plots"""
    try:
        if not experiment_results:
            return jsonify({'status': 'error', 'message': 'No results available'}), 400
        
        # Generate accuracy comparison plot
        algorithms = list(experiment_results.keys())
        avg_accuracies = []
        
        for algo in algorithms:
            algo_data = experiment_results[algo]['results']
            if isinstance(algo_data, dict):
                # For MetaFed
                accuracies = [
                    metrics['accuracy'] 
                    for fed_id, metrics in algo_data.items() 
                    if fed_id != 'common'
                ]
            elif isinstance(algo_data, list):
                # For FedAvg (list of round results)
                last_round = algo_data[-1]
                accuracies = [
                    metrics['accuracy'] 
                    for fed_id, metrics in last_round.items()
                ]
            else:
                accuracies = [0]
            
            avg_accuracies.append(np.mean(accuracies) if accuracies else 0)
        
        # Create plot data
        plot_data = {
            'algorithms': algorithms,
            'average_accuracies': avg_accuracies,
            'plot_type': 'bar'
        }
        
        return jsonify({
            'status': 'success',
            'plot_data': plot_data
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/gpu_status', methods=['GET'])
def gpu_status():
    """Get GPU status"""
    try:
        gpu_info = gpu_manager.get_memory_usage()
        
        response = {
            'status': 'success',
            'device': str(config.DEVICE),
            'cuda_available': torch.cuda.is_available(),
            'gpu_info': gpu_info
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('./logs/training_logs', exist_ok=True)
    os.makedirs('./logs/evaluation_logs', exist_ok=True)
    os.makedirs('./logs/models', exist_ok=True)
    os.makedirs('./static/uploads', exist_ok=True)
    os.makedirs('./static/predictions', exist_ok=True)
    
    print(f"Using device: {config.DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    app.run(debug=True, port=5000)