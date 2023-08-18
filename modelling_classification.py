import pandas as pd
import os
import joblib
import json
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load Airbnb data for classification
def load_airbnb_classification(label='Category'):
    df = pd.read_csv('listing.csv')
    df_cleaned = df.dropna(subset=[label])
    df_cleaned = df_cleaned[~df['guests'].apply(lambda x: isinstance(x, str) and 'Somerford Keynes England Unit' in x)]
    labels = df_cleaned[label]
    features = df_cleaned.drop(columns=[label])
    return features, labels

def preprocess_data(features):
    numerical_columns = ['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating',
                         'Location_rating', 'Check-in_rating', 'Value_rating',
                         'guests', 'beds', 'bathrooms']
    non_numeric_columns = [col for col in features.columns if col not in numerical_columns]

    imputer = SimpleImputer(strategy='mean')
    features[numerical_columns] = imputer.fit_transform(features[numerical_columns])

    features = features.drop(columns=non_numeric_columns)
    return features

# Train a logistic regression model
def train_classification_model(X_train, y_train):
    logreg_model = LogisticRegression(max_iter=10000, random_state=42)
    logreg_model.fit(X_train, y_train)
    return logreg_model

# Hyperparamter tuning for classifcation
def tune_classification_model_hyperparameters(model_class, param_grid, X_train, y_train, X_val, y_val):
    # Create a GridSearchCV object with accuracy scoring
    grid_search = GridSearchCV(model_class, param_grid, scoring='accuracy', cv=3)
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    
    # Get the best model and best hyperparameters
    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_
    
    # Evaluate the best model on the validation set
    y_val_pred = best_model.predict(X_val)
    validation_accuracy = accuracy_score(y_val, y_val_pred)
    
    # Create a dictionary of performance metrics
    best_performance = {
        "validation_accuracy": validation_accuracy,
    }
    
    return best_model, best_hyperparameters, best_performance

def save_model(model, hyperparameters, metrics, folder):
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Save the trained model to a file
    model_file = os.path.join(folder, 'model.joblib')
    joblib.dump(model, model_file)

    # Save the hyperparameters to a JSON file
    hyperparameters_file = os.path.join(folder, 'hyperparameters.json')
    with open(hyperparameters_file, 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    # Save the performance metrics to a JSON file
    metrics_file = os.path.join(folder, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)


def main():
    # Load the dataset with 'Category' as the label
    features, labels = load_airbnb_classification(label='Category')
    
    # Preprocess data
    features_preprocessed = preprocess_data(features)   
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_preprocessed, labels, test_size=0.3, random_state=42)
    
    
    # Train a classification model
    classification_model = train_classification_model(X_train, y_train)
    
    # Make predictions on training and test sets
    y_train_pred = classification_model.predict(X_train)
    y_test_pred = classification_model.predict(X_test)
    
    # Compute performance measures for training set
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred, average='weighted')
    recall_train = recall_score(y_train, y_train_pred, average='weighted')
    f1_train = f1_score(y_train, y_train_pred, average='weighted')
    
    # Compute performance measures for test set
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred, average='weighted')
    recall_test = recall_score(y_test, y_test_pred, average='weighted')
    f1_test = f1_score(y_test, y_test_pred, average='weighted')
    
    # Print performance measures
    print("Training Accuracy:", accuracy_train)
    print("Training Precision:", precision_train)
    print("Training Recall:", recall_train)
    print("Training F1 Score:", f1_train)
    print("Test Accuracy:", accuracy_test)
    print("Test Precision:", precision_test)
    print("Test Recall:", recall_test)
    print("Test F1 Score:", f1_test)

    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2', None]
    }

    best_logreg_model, best_hyperparameters, best_performance = tune_classification_model_hyperparameters(
        classification_model, param_grid, X_train, y_train, X_test, y_test
    )

    print("Best Model:", best_logreg_model)
    print("Best Hyperparameters:", best_hyperparameters)
    print("Best Performance:", best_performance)

    # Save the best logistic regression model, hyperparameters, and metrics
    save_model(best_logreg_model, best_hyperparameters, best_performance, folder='models/classification/logistic_regression')

if __name__ == "__main__":
    main()


