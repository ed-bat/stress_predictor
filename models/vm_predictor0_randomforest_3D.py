''' 
vm_predictor0_randomforest_3D.py

This script trains a Random Forest regressor to predict the 3D von Mises stress distribution

'''

STRESS_FILE_DIRECTORY = "C:/Users/ed_ba/OneDrive - The University of Manchester/Year 3/Project/finite_element_data/"

#----------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import time
import joblib

def extract_model_info(model):
    """Extracts information from the model for visualization purposes."""
    # Get all parameters of the model
    params = model.get_params()
    
    # Extracting parameters
    return {
        'Model': 'RandomForestRegressor',
        'n_estimators': params['randomforestregressor__n_estimators'],
        'max_samples': params['randomforestregressor__max_samples'],
        'max_depth': params.get('randomforestregressor__max_depth', 'Not Set') 
    }

def plot_predictions(actual, predicted, model_info):
    actual_mpa = actual / 1e6  # Convert from Pa to MPa
    predicted_mpa = predicted / 1e6  # Convert from Pa to MPa

    plt.figure(figsize=(10, 6))
    plt.scatter(actual_mpa, predicted_mpa, alpha=0.5, label='Predicted vs Actual')
    plt.title('3D Model Prediction Accuracy')
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
    min_error = errors.min()

    formatted_train_rmse = format(train_rmse / 1e6, '.4f')  # RMSE in MPa, rounded to 4 decimal places
    formatted_valid_rmse = format(valid_rmse / 1e6, '.4f')  # RMSE in MPa, rounded to 4 decimal places
    valid_rmse_percentage = format(valid_rmse / y_valid.mean() * 100, '.4f')  # RMSE as a percentage
    formatted_max_error = format(max_error/1e6, '.4f')
    formatted_min_error = format(min_error/1e6, '.4f')

    # Print performance metrics
    print(f"\nTraining RMSE: {formatted_train_rmse} MPa")
    print(f"Validation RMSE: {formatted_valid_rmse} MPa")
    print(f"Validation RMSE as Percentage of Mean = {valid_rmse_percentage} %")
    print(f"Maximum Error = {formatted_max_error} MPa")
    print(f"Minimum Error = {formatted_min_error} MPa\n")

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
    data = pd.read_csv(f"{STRESS_FILE_DIRECTORY}3D_stress_field.csv")

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

#----------------------------------------------------------------------------------------------------------------------------------------------
start_time = time.perf_counter()
x_scaler = StandardScaler()
y_scaler = StandardScaler()

model_type = RandomForestRegressor( 
    n_estimators=1205,
    max_depth=29, 
    max_samples=27619, 
    n_jobs=-1, 
    random_state=42
    )

# For Hyperparameter tuning
param_dist = {
    'randomforestregressor__n_estimators': randint(low=250, high=1500),
    'randomforestregressor__max_depth': randint(low=10, high=50),
    'randomforestregressor__max_samples': randint(low=5000, high=28000)
}

x_train_full, y_train_full, y_train_full_scaled, x_train, y_train, y_train_scaled, x_valid, y_valid, y_valid_scaled, x_test, y_test, y_test_scaled = prepare_data(
    y_scaler, test_size=0.2, random_state=42
)

model = make_pipeline(x_scaler, model_type)
# model = joblib.load('3D_randomforest')
training_start_time = time.perf_counter()
model.fit(x_train, y_train_scaled)
training_end_time = time.perf_counter()
# joblib.dump(model, '3D_randomforest')

model_info = extract_model_info(model)

# model, search_results = perform_randomized_search(model, param_dist, x_train, y_train_scaled, n_iter=50, cv=2)
evaluate_model_performance(model, x_train, y_train, x_valid, y_valid, y_scaler)
# perform_cross_validation(model, x_train_full, y_train_full, cv=10)


# final_model_performance(model, x_test, y_test, y_scaler, start_time, training_start_time, training_end_time)