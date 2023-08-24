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
    

# def train(model, train_dataloader, val_dataloader, test_dataloader, config, num_epochs=10):
#     optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
#     criterion = nn.MSELoss()
#     for epoch in range(num_epochs):
#         model.train()  # Set the model to training mode
#         for batch_features, batch_labels in train_dataloader:
#             optimizer.zero_grad()
#             outputs = model(batch_features)
#             loss = criterion(outputs, batch_labels.view(-1,1).float())
#             loss.backward()
#             optimizer.step()
#             # Write training loss to TensorBoard
#             writer.add_scalar('Training Loss', loss.item(), global_step=epoch)
        
#         model.eval()  # Set the model to evaluation mode
#         val_loss = 0.0
#         val_rmse = 0.0
#         val_r2 = 0.0
#         start_inference_time = time.time()

#         with torch.no_grad(): 
#             for batch_features, batch_labels in val_dataloader:
#                 outputs = model(batch_features)
#                 val_loss += criterion(outputs, batch_labels.unsqueeze(1)).item()

#                 # Calculate RMSE and R^2
#                 val_rmse += mean_squared_error(batch_labels, outputs)**0.5
#                 val_r2 += r2_score(batch_labels, outputs)
            
#             # Calculate and write average validation loss, latency, rmse and r2 for the epoch
#             inference_latency = (time.time() -start_inference_time) /len(val_dataloader)
#             val_loss /= len(val_dataloader)
#             val_rmse /= len(val_dataloader)
#             val_r2 /= len(val_dataloader)

#             # Save metrics and model after each epoch
#             metrics = {
#                 'RMSE_loss_val': val_rmse,
#                 'R_squared_val': val_r2,
#                 'inference_latency': inference_latency
#             }
#             save_folder = os.path.join('models/neural_networks/regression', datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
#             save_model(model, config, metrics, save_folder)

#             writer.add_scalar('Validation Loss', val_loss, global_step=epoch)
        
#             print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss:.4f}")

def train(model, train_dataloader, val_dataloader, test_dataloader, config, num_epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for batch_features, batch_labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            # Write training loss to TensorBoard
            writer.add_scalar('Training Loss', loss.item(), global_step=epoch)
        
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_r2 = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in val_dataloader:
                outputs = model(batch_features)
                val_loss += criterion(outputs, batch_labels.unsqueeze(1)).item()

                # Calculate RMSE and R^2
                val_r2 += r2_score(batch_labels, outputs)
            
            # Calculate and write average validation rmse loss, latency, and r2 for the epoch
            val_loss /= len(val_dataloader)
            writer.add_scalar('Validation Loss', val_loss, global_step=epoch)
            val_r2 /= len(val_dataloader)
            writer.add_scalar('Validation R2', val_r2, global_step=epoch)
        
        # Calculate performance metrics
        train_rmse = mean_squared_error(batch_labels, outputs)**0.5
        train_r2 = r2_score(batch_labels, outputs)
        start_time = time.time()
        test_rmse, test_r2 = evaluate_model(model, test_dataloader)
        training_duration = time.time() - start_time
        
        # Calculate performance latency
        inference_latencies = []
        for batch_features, batch_labels in test_dataloader:
            start_inference_time = time.time()
            model(batch_features)
            inference_latencies.append(time.time() - start_inference_time)
        inference_latency = sum(inference_latencies) / len(test_dataloader)
        
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



# # New save_model function
# def save_model(model, hyperparameters, metrics, folder):
#     os.makedirs(folder, exist_ok=True)
#     model_file = os.path.join(folder, 'model.pt')

#     if isinstance(model, nn.Module):
#         torch.save(model.state_dict(), model_file)
#     else:
#         joblib.dump(model, model_file)

#     hyperparameters_file = os.path.join(folder, 'hyperparameters.json')
#     with open(hyperparameters_file, 'w') as f:
#         json.dump(hyperparameters, f, indent=4)
    
#     metrics_file = os.path.join(folder, 'metrics.json')
#     with open(metrics_file, 'w') as f:
#         json.dump(metrics, f, indent=4)

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
        json.dump(config, f, indent=4)
    
    train_rmse, val_rmse, test_rmse = metrics['RMSE_loss_train'], metrics['RMSE_loss_val'], metrics['RMSE_loss_test']
    train_r2, val_r2, test_r2 = metrics['R_squared_train'], metrics['R_squared_val'], metrics['R_squared_test']
    training_duration = metrics['training_duration']
    inference_latency = metrics['inference_latency']
    
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
    model = NNModel(input_size, hidden_size, output_size, config=config)
    # # Train the model
    # train(model, train_dataloader, val_dataloader, test_dataloader, config, num_epochs=10)

    # # Create a timestamped folder for saving the model and metrics
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # save_folder = os.path.join("models", "neural_networks", "regression", timestamp)
 # Train the model
    model_folder = os.path.join('models/neural_networks/regression', datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    train(model, train_dataloader, val_dataloader, test_dataloader, config, num_epochs=10)
    
    # Save the trained model, hyperparameters, and metrics
    save_model(model, config, model_folder)