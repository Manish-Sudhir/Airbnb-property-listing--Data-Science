import torch
from torch.utils.data import Dataset, DataLoader
from tabular_data import load_airbnb
from sklearn.model_selection import train_test_split
import torch.nn as nn
from sklearn.impute import SimpleImputer
import pandas as pd

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

class NNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define layers
        self.layers = torch.nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self,X):
        return self.layers(X)
    

def train(model, train_dataloader, val_dataloader, test_dataloader, num_epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for batch_features, batch_labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels.view(-1,1).float())
            loss.backward()
            optimizer.step()
        
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad(): 
            for batch_features, batch_labels in val_dataloader:
                outputs = model(batch_features)
                val_loss += criterion(outputs, batch_labels.unsqueeze(1)).item()
            
            val_loss /= len(val_dataloader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss:.4f}")

# Create an instance of your model
input_size = len(train_features.columns)  # Adjust this based on your actual number of input features
hidden_size = 64  # Adjust as needed
output_size = 1  # Regression task, predicting a single value
model = NNModel(input_size, hidden_size, output_size)

# Train the model
train(model, train_dataloader, val_dataloader, test_dataloader, num_epochs=10)