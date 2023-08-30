import pandas as pd
import os
import joblib
import json
from sklearn.impute import SimpleImputer
from tabular_data import clean_tabular_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# Load Airbnb data for classification
def load_airbnb_classification(label='Category'):
    df = pd.read_csv('listing.csv')
    df_cleaned = clean_tabular_data(df)
    df_cleaned = df_cleaned.dropna(subset=[label])
    df_cleaned = df_cleaned[~df['guests'].apply(lambda x: isinstance(x, str) and 'Somerford Keynes England Unit' in x)]
    labels = df_cleaned[label]
    features = df_cleaned.drop(columns=[label])
    return features, labels

def preprocess_data(features):
    numerical_columns = ['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating',
                         'Location_rating', 'Check-in_rating', 'Value_rating',
                         'guests', 'beds', 'bathrooms']
    non_numeric_columns = [col for col in features.columns if col not in numerical_columns]

    # imputer = SimpleImputer(strategy='mean')
    # features[numerical_columns] = imputer.fit_transform(features[numerical_columns])

    features = features.drop(columns=non_numeric_columns)
    return features

# Train model
def train_classification_model(model_class, hyperparameters, X_train, y_train):
    # Initialize and train the model with given hyperparameters
    model = model_class(**hyperparameters)
    model.fit(X_train, y_train)
    return model

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

def evaluate_classification_model(model_class, X_test, y_test):
    y_pred = model_class.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return accuracy, precision, recall, f1

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

def evaluate_all_models(features, labels, task_folder):
    # Preprocess data
    features_preprocessed = preprocess_data(features)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_preprocessed, labels, test_size=0.3, random_state=42)
    
    # Define hyperparameter grids for each model
    decision_tree_param_grid = {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    random_forest_param_grid = {
        'n_estimators': [50, 100, 200, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    gradient_boosting_param_grid = {
        'n_estimators': [50, 100, 200, 150],
        'learning_rate': [0.01, 0.1, 0.05],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    logistic_regression_param_grid = {
        'C': [0.1, 1, 10],
        # 'penalty': ['l2', None],
        'penalty': ['l1', 'l2', 'elasticnet', None]
    }
    
    # Models to evaluate
    models = [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        LogisticRegression(max_iter=1000, random_state=42)
    ]
    
    for model in models:
        if isinstance(model, DecisionTreeClassifier):
            param_grid = decision_tree_param_grid
            model_name = "decision_tree"
        elif isinstance(model, RandomForestClassifier):
            param_grid = random_forest_param_grid
            model_name = "random_forest"
        elif isinstance(model, GradientBoostingClassifier):
            param_grid = gradient_boosting_param_grid
            model_name = "gradient_boosting"
        elif isinstance(model, LogisticRegression):
            param_grid = logistic_regression_param_grid
            model_name = "logistic_regression"
        else:
            continue  # Skip unknown models
        
        # Tune and train the model using GridSearchCV
        best_model, best_hyperparameters, best_performance = tune_classification_model_hyperparameters(
            model, param_grid, X_train, y_train, X_test, y_test
        )
        
        # Train the best model using the full training data
        final_model = train_classification_model(best_model.__class__, best_hyperparameters, X_train, y_train)
        

        # Evaluate the final model on the test set
        test_accuracy, test_precision, test_recall, test_f1 = evaluate_classification_model(final_model, X_test, y_test)
        
        # Make predictions on test set
        y_test_pred = final_model.predict(X_test)
        
        # Calculate performance metrics for the test set
        accuracy_test = accuracy_score(y_test, y_test_pred)
        precision_test = precision_score(y_test, y_test_pred, average='weighted')
        recall_test = recall_score(y_test, y_test_pred, average='weighted')
        f1_test = f1_score(y_test, y_test_pred, average='weighted')
        
        # Save the model, hyperparameters, and metrics
        folder_path = f"models/{task_folder}/{model_name}"
        save_model(final_model, best_hyperparameters, {
            "test_accuracy": test_accuracy, 
            "test_precision": test_precision, 
            "test_recall": test_recall, 
            "test_f1": test_f1,
            **best_performance
            }, folder_path)
    
        
        print(f"{model_name.capitalize()} - Best Hyperparameters:", best_hyperparameters)
        print("Test Accuracy:", accuracy_test)
        print("Test Precision:", precision_test)
        print("Test Recall:", recall_test)
        print("Test F1 Score:", f1_test)
        print("Best Performance:", best_performance)

def find_best_model(models_folder, task_folder):
    model_folders = [f for f in os.listdir(os.path.join(models_folder, task_folder)) if os.path.isdir(os.path.join(models_folder, task_folder, f))]

    best_model = None
    best_metrics = None
    best_hyperparameters = None
    best_accuracy = 0.0

    for model_name in model_folders:
        model_folder = os.path.join(models_folder, task_folder, model_name)
        metrics_file = os.path.join(model_folder, 'metrics.json')

        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                accuracy = metrics['test_accuracy']

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_metrics = metrics

                    # Load hyperparameters
                    hyperparameters_file = os.path.join(model_folder, 'hyperparameters.json')
                    with open(hyperparameters_file, 'r') as hf:
                        best_hyperparameters = json.load(hf)

                    # Load model
                    model_file = os.path.join(model_folder, 'model.joblib')
                    best_model = joblib.load(model_file)

    return best_model, best_hyperparameters, best_metrics


def main():
    # Load the dataset with 'Category' as the label
    features, labels = load_airbnb_classification(label='Category')
    
    # Call the evaluate_all_models function for classification task
    evaluate_all_models(features, labels, task_folder='classification')

    # Find the best model
    best_model, best_hyperparameters, best_metrics = find_best_model("models", "classification")

    print("Best Model:", best_model)
    print("Best Hyperparameters:", best_hyperparameters)
    print("Best Metrics:", best_metrics)

if __name__ == "__main__":
    main()

# def main():
#     # Load the dataset with 'Category' as the label
#     features, labels = load_airbnb_classification(label='Category')
    
#     # Preprocess data
#     features_preprocessed = preprocess_data(features)   
    
#     # Split data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(features_preprocessed, labels, test_size=0.3, random_state=42)
    
    
#     # Train a classification model
#     classification_model = train_classification_model(X_train, y_train)
    
#     # Make predictions on training and test sets
#     y_train_pred = classification_model.predict(X_train)
#     y_test_pred = classification_model.predict(X_test)
    
#     # Compute performance measures for training set
#     accuracy_train = accuracy_score(y_train, y_train_pred)
#     precision_train = precision_score(y_train, y_train_pred, average='weighted')
#     recall_train = recall_score(y_train, y_train_pred, average='weighted')
#     f1_train = f1_score(y_train, y_train_pred, average='weighted')
    
#     # Compute performance measures for test set
#     accuracy_test = accuracy_score(y_test, y_test_pred)
#     precision_test = precision_score(y_test, y_test_pred, average='weighted')
#     recall_test = recall_score(y_test, y_test_pred, average='weighted')
#     f1_test = f1_score(y_test, y_test_pred, average='weighted')
    
#     # Print performance measures
#     print("Training Accuracy:", accuracy_train)
#     print("Training Precision:", precision_train)
#     print("Training Recall:", recall_train)
#     print("Training F1 Score:", f1_train)
#     print("Test Accuracy:", accuracy_test)
#     print("Test Precision:", precision_test)
#     print("Test Recall:", recall_test)
#     print("Test F1 Score:", f1_test)

#     param_grid = {
#         'C': [0.1, 1, 10],
#         'penalty': ['l2', None]
#     }

#     best_logreg_model, best_hyperparameters, best_performance = tune_classification_model_hyperparameters(
#         classification_model, param_grid, X_train, y_train, X_test, y_test
#     )

#     print("Best Model:", best_logreg_model)
#     print("Best Hyperparameters:", best_hyperparameters)
#     print("Best Performance:", best_performance)

#     # Save the best logistic regression model, hyperparameters, and metrics
#     save_model(best_logreg_model, best_hyperparameters, best_performance, folder='models/classification/logistic_regression')