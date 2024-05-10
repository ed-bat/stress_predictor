''' 
vm_predictor1_mlpregressor_2D.py

This script trains a Scikit-Learn MLPregressor to predict the von Mises stress distribution on a specified cross_section of the geometry.

'''

STRESS_FILE_DIRECTORY = "C:/Users/ed_ba/OneDrive - The University of Manchester/Year 3/Project/finite_element_data/"

#----------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scipy.stats import loguniform
from sklearn.model_selection import cross_val_score
import time
import joblib

def extract_model_info(model):
    # Get all parameters of the model
    params = model.get_params()
    
    # Extracting parameters
    return {
        'Model': 'MLPRegressor',
        'Hidden Layer Sizes': params.get('mlpregressor__hidden_layer_sizes', 'Not Set'),
        'Activation Function': params.get('mlpregressor__activation', 'Not Set'),
        'Solver': params.get('mlpregressor__solver', 'Not Set'),
        'Alpha (Regularization)': params.get('mlpregressor__alpha', 'Not Set'),
        'Initial Learning Rate': params.get('mlpregressor__learning_rate_init', 'Not Set'),
        'Momentum': params.get('mlpregressor__momentum', 'Not Set')
    }



def plot_predictions(actual, predicted, model_info):
    actual_mpa = actual / 1e6  # Convert from Pa to MPa
    predicted_mpa = predicted / 1e6  # Convert from Pa to MPa

    plt.figure(figsize=(10, 6))
    plt.scatter(actual_mpa, predicted_mpa, alpha=0.5, label='Predicted vs Actual')
    plt.title('2D Model Prediction Accuracy')
    plt.xlabel('Actual Values (MPa)')
    plt.ylabel('Predicted Values (MPa)')
    
    # Plot y=x line
    ideal_min = min(actual_mpa.min(), predicted_mpa.min())
    ideal_max = max(actual_mpa.max(), predicted_mpa.max())
    plt.plot([ideal_min, ideal_max], [ideal_min, ideal_max], 'r--', label='Ideal Prediction (y=x)')
    
    # First legend
    plt.legend(loc='upper left')
    
    # Additional text box for model information
    textstr = '\n'.join(f"{key}: {val}" for key, val in model_info.items())
    props = dict(boxstyle='round', facecolor='lightgrey', alpha=1.0, edgecolor='black', linewidth=1)
    plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='bottom', horizontalalignment='right', bbox=props, color='black', fontweight='bold')
    
    plt.grid(True)
    plt.show()


def evaluate_model_performance(model, x_train, y_train, x_valid, y_valid, y_scaler):
    # Predict on the training and validation set
    y_train_pred_scaled = model.predict(x_train).flatten()
    y_valid_pred_scaled = model.predict(x_valid).flatten()

    # Inverse transform the predictions to original scale
    y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    y_valid_pred = y_scaler.inverse_transform(y_valid_pred_scaled.reshape(-1, 1)).flatten()

    # Calculate the RMSE in the original scale
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))

    errors = abs(y_valid - y_valid_pred.ravel())
    max_error = errors.max()

    formatted_train_rmse = format(train_rmse / 1e6, '.4f')  # RMSE in MPa, rounded to 4 decimal places
    formatted_valid_rmse = format(valid_rmse / 1e6, '.4f')  # RMSE in MPa, rounded to 4 decimal places
    valid_rmse_percentage = format(valid_rmse / y_valid.mean() * 100, '.4f')  # RMSE as a percentage
    formatted_max_error = format(max_error/1e6, '.4f')

    # Print performance metrics
    print(f"\nTraining RMSE: {formatted_train_rmse} MPa")
    print(f"Validation RMSE: {formatted_valid_rmse} MPa")
    print(f"Validation RMSE as Percentage of Mean = {valid_rmse_percentage} %")
    print(f"Maximum Error = {formatted_max_error} MPa")

    plot_predictions(y_valid, y_valid_pred, model_info)


def final_model_performance(model, x_test, y_test, y_scaler, start_time, training_start_time, training_end_time):
    prediction_start_time = time.perf_counter()

    # Predict on the test set
    y_test_pred_scaled = model.predict(x_test).flatten()

    # Inverse transform the predictions to original scale
    y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

    prediction_end_time = time.perf_counter()

    # Calculate the RMSE in the original scale
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    errors = abs(y_test - y_test_pred.ravel())
    max_error = errors.max()

    formatted_test_rmse = format(test_rmse / 1e6, '.4f')  # RMSE in MPa, rounded to 4 decimal places
    test_rmse_percentage = format(test_rmse / y_test.mean() * 100, '.4f')  # RMSE as a percentage
    formatted_max_error = format(max_error/1e6, '.4f')

    elapsed_preprocessing =  training_start_time - start_time
    elapsed_training = training_end_time - training_start_time
    elapsed_pred = prediction_end_time - prediction_start_time
    total_predict_time = format(elapsed_preprocessing + elapsed_pred, '.2f')
    total_train_time = format(elapsed_preprocessing + elapsed_training, '.2f')

    # Print performance metrics
    print(f"Final RMSE on test set: {formatted_test_rmse} MPa")
    print(f"Final RMSE as Percentage of Mean = {test_rmse_percentage} %")
    print(f"Maximum Error = {formatted_max_error} MPa\n")
  
    print(f'Preprocessing time: {elapsed_preprocessing}s')
    print(f'Training time: {elapsed_training}s')
    print(f'Prediction time: {elapsed_pred}s\n')

    print(f'Total training time: {total_train_time}s')
    print(f'Total prediction time: {total_predict_time}s')

    plot_predictions(y_test, y_test_pred, model_info)


def perform_randomized_search(model, param_distributions, x_train, y_train, n_iter=10, cv=5, random_state=42):
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        verbose=1,
        random_state=random_state,
        scoring='neg_root_mean_squared_error'
    )
    random_search.fit(x_train, y_train)

    # Extract the best estimator and the cv results
    best_estimator = random_search.best_estimator_
    cv_results = pd.DataFrame(random_search.cv_results_)

  # Filter the DataFrame to keep only parameter columns and the mean test score
    columns_to_keep = [col for col in cv_results.columns if col.startswith('param_') or col == 'mean_test_score']
    filtered_cv_results = cv_results[columns_to_keep]
    filtered_cv_results['mean_test_score'] = -filtered_cv_results['mean_test_score']  # Convert negative MSE to positive for readability

    # Clean up column names by removing 'param_' prefix and model name prefix
    filtered_cv_results.rename(columns=lambda x: x.replace('param_', '').split('__')[-1], inplace=True)
    filtered_cv_results.rename(columns={'mean_test_score': 'Mean RMSE'}, inplace=True)
    filtered_cv_results.sort_values(by='Mean RMSE', ascending=True, inplace=True)

    print("\nBest Model:", best_estimator)
    print(f"\n{filtered_cv_results.head()}")

    return best_estimator, filtered_cv_results


def prepare_data(y_scaler, test_size=0.2, random_state=42):
    # Load data
    data = pd.read_csv(f"{STRESS_FILE_DIRECTORY}2D_stress_field.csv")

    # Define inputs and outputs
    x = data.drop(['von_mises', 'S11', 'S22', 'S33', 'S12', 'S13', 'S23'], axis=1)
    y = data['von_mises']

    # Split the data into training, testing, and validation sets
    x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full, test_size=test_size, random_state=random_state)

    # Scale the target variable
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_train_full_scaled = y_scaler.fit_transform(y_train_full.values.reshape(-1, 1)).ravel()
    y_valid_scaled = y_scaler.transform(y_valid.values.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    return x_train_full, y_train_full, y_train_full_scaled, x_train, y_train, y_train_scaled, x_valid, y_valid, y_valid_scaled, x_test, y_test, y_test_scaled


def perform_cross_validation(model, x_train_full, y_train_full, cv=10):
    crossval_rmses = -cross_val_score(model, x_train_full, y_train_full, scoring='neg_root_mean_squared_error', cv=cv)
    print(pd.Series(crossval_rmses).describe())

#-------------------------------------------------------------------------------------------------------------------------------------
start_time = time.perf_counter()
x_scaler = StandardScaler()
y_scaler = StandardScaler()


model_type =  MLPRegressor(alpha=3.14e-3,
                            hidden_layer_sizes=(200, 200, 200),
                            learning_rate_init=0.017357,
                            max_iter=1000, momentum=0.918778,
                            random_state=42, solver='lbfgs',
                            verbose=True
                            )

param_dist = {
    'mlpregressor__hidden_layer_sizes': [(size, size, size) for size in range(50, 210, 10)],
    'mlpregressor__activation': ['relu', 'tanh', 'logistic'],
    'mlpregressor__solver': ['adam', 'sgd', 'lbfgs'],
    'mlpregressor__alpha': loguniform(1e-5, 1e-1),
    'mlpregressor__learning_rate_init': loguniform(1e-4, 1e-1),
    'mlpregressor__momentum': loguniform(0.8, 0.99)
}


x_train_full, y_train_full, y_train_full_scaled, x_train, y_train, y_train_scaled, x_valid, y_valid, y_valid_scaled, x_test, y_test, y_test_scaled = prepare_data(
    y_scaler, test_size=0.2, random_state=42
)

model = make_pipeline(x_scaler, model_type)
training_start_time = time.perf_counter()
model.fit(x_train, y_train_scaled)
training_end_time = time.perf_counter()
# joblib.dump(model, '2D_mlpregressor')

model_info = extract_model_info(model)

# model, search_results = perform_randomized_search(model, param_dist, x_train, y_train_scaled, n_iter=200, cv=3)
evaluate_model_performance(model, x_train, y_train, x_valid, y_valid, y_scaler)
# perform_cross_validation(model, x_train_full, y_train_full_scaled, cv=10)
# y_pred = model_predict(x_valid, y_scaler)

# final_model_performance(model, x_test, y_test, y_scaler, start_time, training_start_time, training_end_time)