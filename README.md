# Machine Learning and Neural Network Models for Airbnb Dataset

This repository contains Python scripts for training and evaluating machine learning models, as well as neural network models, for both regression and classification tasks using the Airbnb dataset.

## Introduction

This project focuses on training and evaluating machine learning models, including neural networks, for regression and classification tasks using the Airbnb dataset. The dataset provides valuable insights into property listings, making it a suitable candidate for predicting prices, categories, and more.

## Installation

1. Clone the repository:

gh repo clone Manish-Sudhir/Airbnb-property-listing--Data-Science


2. Navigate to the repository folder:

cd Airbnb-property-listing--Data-Science

3. Install the required packages:

pip install -r requirements.txt


## Usage

### Regression Models

1. Ensure you have the necessary dataset (`listing.csv`) in the repository.
2. Run the `modelling.py` script:


This script preprocesses the data, tunes different regression models, evaluates them, and saves the best model along with its hyperparameters and metrics.

### Classification Models

1. Ensure you have the necessary dataset (`listing.csv`) in the repository.
2. Run the `modelling_classification.py` script:


This script preprocesses the data, tunes different classification models, evaluates them, and saves the best model along with its hyperparameters and metrics.

### Neural Network Models

1. Ensure you have the necessary dataset (`listing.csv`) in the repository.
2. Run the `modelling_neural_network.py` script:


This script preprocesses the data, designs and trains neural network models, evaluates them, and saves the best model along with its performance metrics.


## Models

The repository includes various models and neural network architectures for different tasks:

### Regression:
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Stochastic Gradient Descent Regressor

### Classification:
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression Classifier

### Neural Networks:
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)

## Results

After running the scripts, the best models, hyperparameters, and metrics are saved in the `models` directory for each task. This comprehensive approach helps in choosing the best model architecture for different predictive tasks related to Airbnb data.

## Usefulness

The models and neural networks developed here can be utilized in various ways for the Airbnb dataset:

- **Price Prediction**: Regression models can predict property prices based on various features, assisting property owners in setting competitive prices.
- **Category Classification**: Classification models help categorize listings, aiding users in finding accommodations that match their preferences.
- **Market Insights**: By analyzing feature importance, these models provide insights into what factors influence property prices and customer preferences.
- **Decision Support**: Predictive models can assist hosts and guests in making informed decisions related to accommodations and bookings.

## Contributing

Contributions are welcome! If you find any issues or want to enhance the functionality, feel free to submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).




