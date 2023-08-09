import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from tabular_data import load_airbnb

def preprocess_data(features, labels):
    # Preprocess your data here (imputation, scaling, etc.)

    # Split the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Impute missing values in features
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    return X_train_imputed, y_train, X_val_imputed, y_val, X_test_imputed, y_test

def train_model(model_class, hyperparameters, X_train, y_train):
    # Initialize and train the model with given hyperparameters
    model = model_class(**hyperparameters)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y):
    # Evaluate the model and return performance metrics
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    return rmse, r2, mse

def tune_regression_model_hyperparameters(model, param_grid, X_train, y_train, X_val, y_val):
    # Create a GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=3)
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    
    # Get the best model and best hyperparameters
    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_
    
    return best_model, best_hyperparameters

def main():
    # Load the dataset with 'price_night' as the label
    features, labels = load_airbnb()

    # Preprocess data and create validation set
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(features, labels)

    # Define hyperparameter grid for SGDRegressor
    param_grid = {
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [1000, 2000]
        # Add more hyperparameters and values as needed
    }

    # Create and tune the model using GridSearchCV
    sgd_model = SGDRegressor()
    best_model, best_hyperparameters = tune_regression_model_hyperparameters(
        sgd_model, param_grid, X_train, y_train, X_val, y_val
    )
    
    # Train the best model using the full training data
    final_model = train_model(best_model.__class__, best_hyperparameters, X_train, y_train)
    
    # Evaluate the final model on the test set
    test_rmse, test_r2, mse = evaluate_model(final_model, X_test, y_test)
    
    print("Best Hyperparameters:", best_hyperparameters)
    print("Test RMSE:", test_rmse)
    print("Test R2:", test_r2)
    print("Mean Squared Error:", mse)

if __name__ == "__main__":
    main()
