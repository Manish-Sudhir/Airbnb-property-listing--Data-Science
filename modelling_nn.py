import torch
from torch.utils.data import Dataset, DataLoader
from tabular_data import load_airbnb
from sklearn.model_selection import train_test_split
import torch.nn as nn
from sklearn.impute import SimpleImputer
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import json
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import time
from datetime import datetime
import itertools


# Define your AirbnbNightlyPriceRegressionDataset class
class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __getitem__(self, index):
        return (self.features[index], self.labels[index])
    
    def __len__(self):
        return len(self.features)
    
# Load data using load_airbnb function
features, labels = load_airbnb(label='Price_Night')

# Split dataset into train, validation, and test sets using train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=42)
train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.15, random_state=42)

# Impute missing values in features
imputer = SimpleImputer(strategy='mean')
train_features_imputed = imputer.fit_transform(train_features)
val_features_imputed = imputer.transform(val_features)
test_features_imputed = imputer.transform(test_features)

# Convert imputed features back to DataFrames
train_features_imputed_df = pd.DataFrame(data=train_features_imputed, columns=train_features.columns)
val_features_imputed_df = pd.DataFrame(data=val_features_imputed, columns=val_features.columns)
test_features_imputed_df = pd.DataFrame(data=test_features_imputed, columns=test_features.columns)

# Create instances of AirbnbNightlyPriceRegressionDataset
train_dataset = AirbnbNightlyPriceRegressionDataset(train_features_imputed_df, train_labels)
val_dataset = AirbnbNightlyPriceRegressionDataset(val_features_imputed_df, val_labels)
test_dataset = AirbnbNightlyPriceRegressionDataset(test_features_imputed_df, test_labels)

# Create DataLoader instances
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create a SummaryWriter for TensorBoard
writer = SummaryWriter(log_dir='logs')

class NNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, config):
        super().__init__()
        self.config = config

        # Define layers
        self.layers = torch.nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self,X):
        return self.layers(X)
    
# Calculate performance metrics
def get_time(model):
    start_time = time.time()
    training_duration = time.time() - start_time
        
    # Calculate performance latency
    inference_latencies = []
    for batch_features, batch_labels in test_dataloader:
        start_inference_time = time.time()
        model(batch_features)
        inference_latencies.append(time.time() - start_inference_time)
    inference_latency = sum(inference_latencies) / len(test_dataloader)
    return training_duration, inference_latencies
        
def train(model, train_dataloader, val_dataloader, test_dataloader, config, num_epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss_sum = 0.0  # Aggregate loss across batches
        train_predictions = []
        train_labels = []

        for batch_features, batch_labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels.view(-1, 1).float())
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_predictions.extend(outputs.detach().numpy())
            train_labels.extend(batch_labels.numpy())
            # Write training loss to TensorBoard
            writer.add_scalar('Training Loss', loss.item(), global_step=epoch)
        
        train_rmse, train_r2 = evaluate_model(model, train_dataloader)

        # Calculate and write average train rmse and r2 for the epoch
        writer.add_scalar('Train RMSE', train_rmse, global_step=epoch)
        writer.add_scalar('Train R2', train_r2, global_step=epoch)
        
        model.eval()  # Set the model to evaluation mode
        val_loss_sum = 0.0
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for batch_features, batch_labels in val_dataloader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels.view(-1, 1).float())
        
                val_loss_sum += loss.item()
                val_predictions.extend(outputs.detach().numpy())
                val_labels.extend(batch_labels.numpy())

        val_loss, val_r2 = evaluate_model(model, val_dataloader)

        # Calculate and write average train rmse and r2 for the epoch
        writer.add_scalar('Validation RMSE', val_loss, global_step=epoch)
        writer.add_scalar('Validation R2', val_r2, global_step=epoch)
        
        # Calculate metrics for test set using similar approach
        test_loss_sum = 0.0
        test_predictions = []
        test_labels = []

        with torch.no_grad():
            for batch_features, batch_labels in test_dataloader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels.view(-1, 1).float())
        
                test_loss_sum += loss.item()
                test_predictions.extend(outputs.detach().numpy())
                test_labels.extend(batch_labels.numpy())

        test_rmse, test_r2 = evaluate_model(model, test_dataloader)

        # Calculate and write average train rmse and r2 for the epoch
        writer.add_scalar('Test RMSE', test_rmse, global_step=epoch)
        writer.add_scalar('Test R2', test_r2, global_step=epoch)

        # Calculate performance metrics
        # train_rmse = mean_squared_error(batch_labels, outputs)**0.5
        # train_r2 = r2_score(batch_labels, outputs)
        training_duration, inference_latency = get_time(model)
        
        metrics = {
            'RMSE_loss_train': train_rmse,
            'RMSE_loss_val': val_loss,
            'RMSE_loss_test': test_rmse,
            'R_squared_train': train_r2,
            'R_squared_val': val_r2,
            'R_squared_test': test_r2,
            'training_duration': training_duration,
            'inference_latency': inference_latency
        }
        
        # Save the model and metrics
        model_folder = os.path.join('models/neural_networks/regression', datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
        save_model(model, config, metrics, model_folder)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss:.4f}")

# Close the TensorBoard writer
writer.close()

# Read the YAML config file
def get_nn_config():
    with open('nn_config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def evaluate_model(model, dataloader):
    model.eval()
    test_labels = []
    test_predictions = []

    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            outputs = model(batch_features)
            test_labels.extend(batch_labels.numpy())
            test_predictions.extend(outputs.numpy())

    test_rmse = mean_squared_error(test_labels, test_predictions)**0.5
    test_r2 = r2_score(test_labels, test_predictions)
    

    return test_rmse, test_r2

def generate_nn_configs():
    # Define the range of values for each hyperparameter
    learning_rates = [0.001, 0.01, 0.1]
    hidden_layer_widths = [32, 64, 128]
    depths = [1, 2, 3]
    
    # Generate all combinations of hyperparameters
    config_combinations = itertools.product(learning_rates, hidden_layer_widths, depths)
    
    # Create a list of config dictionaries
    configs = []
    for lr, width, depth in config_combinations:
        config = {
            'optimiser': 'Adam',
            'learning_rate': lr,
            'hidden_layer_width': width,
            'depth': depth
        }
        configs.append(config)
    
    return configs


def find_best_nn(train_dataloader, val_dataloader, test_dataloader, num_epochs=10):
    best_metrics = None
    best_model = None
    best_config = None
    
    configs = generate_nn_configs()
    
    for idx, config in enumerate(configs):
        print(f"Training model with config {idx + 1}/{len(configs)}")
        model = NNModel(input_size, hidden_size, output_size, config=config)
        train(model, train_dataloader, val_dataloader, test_dataloader, config, num_epochs)
        val_loss, val_r2 = evaluate_model(model, val_dataloader)

        # Calculate metrics for train and test sets
        train_rmse, train_r2 = evaluate_model(model, train_dataloader)
        test_rmse, test_r2 = evaluate_model(model, test_dataloader)

        # Calculate performance metrics
        training_duration, inference_latency = get_time(model)

        # Update metrics dictionary
        metrics = {
            'RMSE_loss_train': train_rmse,
            'RMSE_loss_val': val_loss,
            'RMSE_loss_test': test_rmse,
            'R_squared_train': train_r2,
            'R_squared_val': val_r2,
            'R_squared_test': test_r2,
            'training_duration': training_duration,
            'inference_latency': inference_latency,
        }

        # # Save the configuration used in hyperparameters.json
        # config['optimiser'] = 'Adam'  # You can modify this if necessary
        # hyperparameters_folder = os.path.join('models/neural_networks/hyperparameters', datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
        # os.makedirs(hyperparameters_folder, exist_ok=True)
        # hyperparameters_file = os.path.join(hyperparameters_folder, 'hyperparameters.json')
        # with open(hyperparameters_file, 'w') as f:
        #     json.dump(config, f, indent=4)

        if best_metrics is None or metrics['RMSE_loss_val'] < best_metrics['RMSE_loss_val']:
            best_metrics = metrics
            best_model = model
            best_config = config
            
    # Save the best model in a folder
    best_model_folder = os.path.join('models/neural_networks/best_model', datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    print('HEY PRINT')
    print(best_config)
    print(best_metrics)
    print(best_model)
    save_model(best_model, best_config, best_metrics, best_model_folder)
    
    return best_model, best_metrics, best_config


def save_model(model, hyperparameters, metrics, folder):
    os.makedirs(folder, exist_ok=True)
    model_file = os.path.join(folder, 'model.pt')

    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), model_file)
    else:
        joblib.dump(model, model_file)

     # Save the hyperparameters to a JSON file
    hyperparameters_file = os.path.join(folder, 'hyperparameters.json')
    with open(hyperparameters_file, 'w') as f:
        json.dump(hyperparameters, f, indent=4)
    
    train_rmse, val_rmse, test_rmse = metrics['RMSE_loss_train'], metrics['RMSE_loss_val'], metrics['RMSE_loss_test']
    train_r2, val_r2, test_r2 = metrics['R_squared_train'], metrics['R_squared_val'], metrics['R_squared_test']
    training_duration = metrics['training_duration']
    inference_latency = metrics['inference_latency']
    
    # THIS FAKEEEEEE
    metrics_file = os.path.join(folder, 'metrics.json')
    metrics.update({
        'RMSE_loss_train': train_rmse,
        'RMSE_loss_val': val_rmse,
        'RMSE_loss_test': test_rmse,
        'R_squared_train': train_r2,
        'R_squared_val': val_r2,
        'R_squared_test': test_r2,
        'training_duration': training_duration,
        'inference_latency': inference_latency
    })
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    # Create an instance of your model
    config = get_nn_config()
    input_size = len(train_features.columns)
    hidden_size = config['hidden_layer_width']
    output_size = 1  # Regression task, predicting a single value
#     model = NNModel(input_size, hidden_size, output_size, config=config)
    
#  # Train the model
#     model_folder = os.path.join('models/neural_networks/regression', datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
#     train(model, train_dataloader, val_dataloader, test_dataloader, config, num_epochs=10)
    best_model, best_metrics, best_config = find_best_nn(train_dataloader, val_dataloader, test_dataloader, num_epochs=10)
