''' 
vm_predictor2_kerassequential_3D.py

This script builds and trains an ANN using the Keras sequential API to predict the full von Mises stress distribution.

'''

STRESS_FILE_DIRECTORY = "C:/Users/ed_ba/OneDrive - The University of Manchester/Year 3/Project/finite_element_data/split_files_3D/"

#----------------------------------------------------------------------------------------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scipy.stats import loguniform
from keras.callbacks import EarlyStopping
import keras_tuner as kt
from keras.models import load_model
from keras_tuner import HyperParameters
import datetime
from tensorflow import keras
import time

def extract_model_info(hyperparameters):
    formatted_info = {
        'Model Type': 'Keras Sequential',
        'Number of Hidden Layers': hyperparameters['n_hidden'],
        'Neurons per Hidden Layer': hyperparameters['n_neurons'],
        'Learning Rate': hyperparameters['learning_rate'],
        'Activation Function': hyperparameters['activation'],
        'Optimizer': hyperparameters['optimizer_type'],
        'Momentum': hyperparameters.get('momentum', 'N/A')
    }

    return formatted_info


def plot_predictions_keras(actual, predicted, hyperparameters):
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, alpha=0.5, label='Predicted vs Actual')
    plt.title('3D Model Prediction Accuracy')
    plt.xlabel('Actual Values (MPa)')
    plt.ylabel('Predicted Values (MPa)')
    
    # Plot y=x line
    ideal_min = min(np.min(actual), np.min(predicted))
    ideal_max = max(np.max(actual), np.max(predicted))
    plt.plot([ideal_min, ideal_max], [ideal_min, ideal_max], 'r--', label='Ideal Prediction (y=x)')
    
    plt.legend(loc='upper left')

    model_info = extract_model_info(hyperparameters)
    textstr = '\n'.join(f"{key}: {val}" for key, val in model_info.items())
    props = dict(boxstyle='round', facecolor='lightgrey', alpha=1.0, edgecolor='black', linewidth=1)
    plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='bottom', horizontalalignment='right', bbox=props, color='black', fontweight='bold')
    
    plt.grid(True)
    plt.show()


def process_set(data_set, model):
    actual, predicted = [], []
    for x_batch, y_batch in data_set:
        y_pred = model.predict(x_batch)
        predicted.extend(y_pred.flatten())
        actual.extend(y_batch.numpy().flatten())

    return np.array(actual), np.array(predicted)


def calculate_rmse(actual, predicted, y_std):
    actual_unstd = (actual * y_std + y_mean) / 1e6
    predicted_unstd = (predicted * y_std + y_mean) / 1e6
    rmse = np.sqrt(mean_squared_error(actual_unstd, predicted_unstd))

    return rmse, actual_unstd, predicted_unstd


def evaluate_model_performance(model, train_set, valid_set, y_mean, y_std, hyperparameters):
    # Process both the training and validation datasets
    actual_train, predicted_train = process_set(train_set, model)
    actual_valid, predicted_valid = process_set(valid_set, model)

    # Calculate RMSE for training and validation sets
    train_rmse, _, _ = calculate_rmse(actual_train, predicted_train, y_std)
    valid_rmse, actual_unstd, predicted_unstd = calculate_rmse(actual_valid, predicted_valid, y_std)

    # Calculate the maximum error for the validation set
    errors = abs(actual_unstd - predicted_unstd)
    max_error = errors.max()

    # RMSE percentage for the validation set
    valid_rmse_percentage = format(valid_rmse / y_mean * 1e8, '.4f') 

    # Print results
    print(f'Training RMSE: {train_rmse:.4f} MPa')
    print(f'Validation RMSE: {valid_rmse:.4f} MPa')
    print(f"Validation RMSE as Percentage of Mean = {valid_rmse_percentage} %")
    print(f'Maximum Error on Validation Set: {max_error:.4f} MPa')
    
    plot_predictions_keras(actual_unstd, predicted_unstd, hyperparameters)


def final_model_performance(model, test_set, y_mean, y_std, hyperparameters, start_time, training_start_time, training_end_time):
    prediction_start_time = time.perf_counter()

    actual_test, predicted_test = process_set(test_set, model)

    prediction_end_time = time.perf_counter()

    rmse, actual_unstd, predicted_unstd = calculate_rmse(actual_test, predicted_test, y_std)
    errors = abs(actual_unstd - predicted_unstd)
    max_error = errors.max()
    rmse_percentage = format(rmse / y_mean * 1e8, '.4f')  # RMSE as a percentage

    elapsed_preprocessing =  training_start_time - start_time
    elapsed_training = training_end_time - training_start_time
    elapsed_pred = prediction_end_time - prediction_start_time
    total_predict_time = format(elapsed_preprocessing + elapsed_pred, '.2f')
    total_train_time = format(elapsed_preprocessing + elapsed_training, '.2f')

    # Print results
    print(f'Final RMSE on test set: {rmse:.4f} MPa')
    print(f"Final RMSE as Percentage of Mean = {rmse_percentage} %")
    print(f'Maximum Error: {max_error:.4f} MPa\n')

    print(f'Preprocessing time: {elapsed_preprocessing}s')
    print(f'Training time: {elapsed_training}s')
    print(f'Prediction time: {elapsed_pred}s\n')

    print(f'Total training time: {total_train_time}s')
    print(f'Total prediction time: {total_predict_time}s')
    
    plot_predictions_keras(actual_unstd, predicted_unstd, hyperparameters)


def parse_csv_line(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)

    return tf.stack(fields[:-1]), tf.stack(fields[-1:])

def preprocess(line):
    x, y = parse_csv_line(line)

    return (x-x_mean) / x_std, (y-y_mean) / y_std

def load_and_preprocess_data(filepaths, n_readers=5, shuffle_buffer_size=10_000, seed=42, batch_size=2048):
    dataset = tf.data.Dataset.list_files(filepaths, seed=seed)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=3, max_value=8)
    n_neurons = hp.Int("n_neurons", min_value=50, max_value=250)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    activation = hp.Choice('activation', ['relu', 'leaky_relu', 'elu', 'selu', 'gelu', 'swish', 'mish'])
    optimizer_type = hp.Choice('optimizer_type', ['sgd', 'sgd_momentum', 'nesterov_sgd', 'rmsprop', 'adam', 'adamax', 'nadam'])

    # Sequential model
    model = tf.keras.Sequential([])

    # Addition of Dense layers based on selected activation
    for _ in range(n_hidden):
        if activation == 'relu':
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu', kernel_initializer="he_normal"))
        elif activation == 'leaky_relu':
            model.add(tf.keras.layers.Dense(n_neurons, activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer="he_normal"))
        elif activation == 'elu':
            model.add(tf.keras.layers.Dense(n_neurons, activation='elu', kernel_initializer="he_normal"))
        elif activation == 'selu':
            model.add(tf.keras.layers.Dense(n_neurons, activation='selu', kernel_initializer="lecun_normal"))
        elif activation == 'gelu':
            gelu = lambda x: 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
            model.add(tf.keras.layers.Dense(n_neurons, activation=gelu, kernel_initializer="he_normal"))
        elif activation == 'swish':
            model.add(tf.keras.layers.Dense(n_neurons, activation='swish', kernel_initializer="he_normal"))
        elif activation == 'mish':
            mish = lambda x: x * tf.math.tanh(tf.math.softplus(x))
            model.add(tf.keras.layers.Dense(n_neurons, activation=mish, kernel_initializer="he_normal"))

    # Add the output layer
    model.add(tf.keras.layers.Dense(1))

    # Select optimizer based on choice
    if optimizer_type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_type == 'sgd_momentum':
        momentum = hp.Float("momentum", min_value=0.7, max_value=0.9)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer_type == 'nesterov_sgd':
        momentum = hp.Float("momentum", min_value=0.7, max_value=0.9)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    elif optimizer_type == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type == 'adamax':
        optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer_type == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

    model.compile(loss='mse', optimizer=optimizer, metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    return model


def load_hyperparameters(hyperparameters):
    params = hyperparameters
    hp = HyperParameters()
    
    for key, value in params.items():
        hp.Fixed(key, value)

    return hp


def save_top_models(tuner, project_name, top_models):
    best_trials = tuner.oracle.get_best_trials(num_trials=top_models)
    for i, trial in enumerate(best_trials):
        # Rebuild the model with the best hyperparameters
        model = tuner.hypermodel.build(trial.hyperparameters)
        
        # Retrain the model
        model.fit(train_set, epochs=n_epochs, validation_data=valid_set)
        
        # Save the model
        model_path = f'keras_models/{project_name}/best_model_{i+1}.h5'
        model.save(model_path)
        print(f'Model saved to {model_path}')


def save_search_results(random_search_tuner, project_name, top_models=10):
    save_top_models(random_search_tuner, project_name, top_models)

    # Get all trials, sorted by performance
    all_trials = sorted(random_search_tuner.oracle.trials.values(), key=lambda x: x.score if x.score is not None else float('inf'))

    # Define the file path for the results
    file_path = f'keras_models/{project_name}/tuning_results.txt'

    # Write the results to a text file
    with open(file_path, 'w') as f:
        f.write("All Trials Hyperparameters and Performance (Best to Worst):\n")
        for trial in all_trials:
            f.write(f"\nTrial ID: {trial.trial_id}\n")
            f.write("Hyperparameters:\n")
            for param, value in trial.hyperparameters.values.items():
                f.write(f"  {param}: {value}\n")
            if trial.score is not None:
                f.write(f"Validation RMSE: {trial.score:.4f}\n")
            else:
                f.write("Validation RMSE: Trial did not complete successfully\n")

# -------------------------------------------------------------------------------------------------------------------------------------
start_time = time.perf_counter()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

log_dir = "C:/Users/ed_ba/Documents/Project/keras_models/logs/" + datetime.datetime.now().strftime("run_%Y_%m_%d-%H_%M_%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

n_inputs = 6 
stats_df = pd.read_csv(STRESS_FILE_DIRECTORY + "training_stats.csv", index_col=0)
x_mean = stats_df.loc['mean'].iloc[:n_inputs].values
x_std = stats_df.loc['std'].iloc[:n_inputs].values
y_mean = stats_df.loc['mean']['von_mises']
y_std = stats_df.loc['std']['von_mises']

train_set = load_and_preprocess_data(tf.io.gfile.glob(STRESS_FILE_DIRECTORY + "train/*.csv"))
valid_set = load_and_preprocess_data(tf.io.gfile.glob(STRESS_FILE_DIRECTORY + "validation/*.csv"))
test_set = load_and_preprocess_data(tf.io.gfile.glob(STRESS_FILE_DIRECTORY + "test/*.csv"))

n_epochs = 1000
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

hyperparameters = {
    'n_hidden': 8,
    'n_neurons': 230,
    'learning_rate': 0.0014,
    'activation': 'relu',
    'optimizer_type': 'nadam',
    'momentum': 0.73
    }
hp = load_hyperparameters(hyperparameters)
model = build_model(hp)
# model = tf.keras.models.load_model('3D_kerassequential.h5')

project_name = "random_search_3D"
random_search_tuner = kt.RandomSearch(
    build_model, objective=kt.Objective("val_rmse", direction="min"), max_trials=50, overwrite=True,
    directory="C:/Users/ed_ba/Documents/Project/code/Models/keras_models", project_name=project_name, seed=42
)

training_start_time = time.perf_counter()
with tf.device('/GPU:0'):
    history = model.fit(train_set, 
                        epochs=n_epochs,  
                        validation_data=valid_set, 
                        callbacks=[early_stopping, tensorboard]
                        )

    # random_search_tuner.search(train_set, epochs=1000, validation_data=valid_set, callbacks=[early_stopping])
    # save_search_results(random_search_tuner, project_name, top_models=5)

training_end_time = time.perf_counter()
# model.save(f'3D_kerassequential.h5')

evaluate_model_performance(model, train_set, valid_set, y_mean, y_std, hyperparameters)
# final_model_performance(model, test_set, y_mean, y_std, hyperparameters, start_time, training_start_time, training_end_time)