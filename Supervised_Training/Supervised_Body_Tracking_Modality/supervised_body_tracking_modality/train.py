
from conf import CUSTOM_SETTINGS, MODALITY_FOLDER, COMPONENT_OUTPUT_FOLDER

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import shutil
import torch
from decouple import config
import pathlib
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

def example_run():
    # Construct the full paths for each CSV file
    train_data_path = os.path.join(MODALITY_FOLDER, 'train.csv')
    test_data_path = os.path.join(MODALITY_FOLDER, 'test.csv')
    val_data_path = os.path.join(MODALITY_FOLDER, 'val.csv')
    
    # Load data from CSV files using the paths
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    val_data = pd.read_csv(val_data_path)
    
    # Separate features and target labels
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    X_val = val_data.iloc[:, :-1].values
    y_val = val_data.iloc[:, -1].values

    # Reshape data for Conv1D: (num_samples, num_features, 1)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_val_encoded = label_encoder.transform(y_val)

    # Get the number of unique classes for the output layer
    num_classes = len(np.unique(y_train_encoded))

    # Build the 1D CNN Model
    activation_fcn = CUSTOM_SETTINGS["sup_config"]["activation"]
    model = Sequential([
        Conv1D(filters=CUSTOM_SETTINGS["sup_config"]["filters"], kernel_size=CUSTOM_SETTINGS["sup_config"]["kernel_size"], activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(CUSTOM_SETTINGS["sup_config"]["dense_neurons"], activation=CUSTOM_SETTINGS["sup_config"]['activation']),
        Dropout(CUSTOM_SETTINGS["sup_config"]["dropout"]),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=CUSTOM_SETTINGS["sup_config"]['optimizer_name'], loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    epoch = CUSTOM_SETTINGS["sup_config"]["epochs"]
    batch_size = CUSTOM_SETTINGS["sup_config"]["batch_size"]

    # Train the Model
    history = model.fit(X_train, y_train_encoded, epochs=epoch, batch_size=batch_size, validation_data=(X_val, y_val_encoded), verbose=2)

    # Evaluate the Model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)

<<<<<<< HEAD
=======
    # # Predict classes on the test set
    # predictions = model.predict(X_test)
    # predicted_classes = np.argmax(predictions, axis=1)
    # class_names = ['BORED', 'ENGAGED', 'FRUSTRATED', 'UNLABELED']
    # # Print reports
    # print("Classification Report:")
    # print(classification_report(y_test_encoded, predicted_classes, target_names=class_names))
    # print("Confusion Matrix:")
    # cm = confusion_matrix(y_test_encoded, predicted_classes)
    # print(pd.DataFrame(cm, index=class_names, columns=class_names))

>>>>>>> 5971aec (lopo2)
    # Predict classes on the test set
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    # Automatically determine class names from y_test_encoded
    unique_classes = np.unique(y_test_encoded)
    class_names = [f"Class {label}" for label in unique_classes]
    # Print reports
    print("Classification Report:")
    print(classification_report(y_test_encoded, predicted_classes, target_names=class_names))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test_encoded, predicted_classes)
    print(pd.DataFrame(cm, index=class_names, columns=class_names))

    # Training and validation losses and accuracies
    print(f"Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
    print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    def clear_directory(directory):
        """Clears the specified directory, removing and recreating it."""
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

    def save_model_weights_and_checkpoint(model):
        """Saves the weights of the provided TensorFlow/Keras model as a .h5 file and full model as .ckpt."""
        clear_directory(COMPONENT_OUTPUT_FOLDER)

        # Path to save the .h5 file for weights only
        weights_path = os.path.join(COMPONENT_OUTPUT_FOLDER, 'model_weights.h5')
        # Path to save the .ckpt file for full model checkpoint
        checkpoint_path = os.path.join(COMPONENT_OUTPUT_FOLDER, 'model_checkpoint.ckpt')

        # Save the model weights
        model.save_weights(weights_path)
        print(f"Weights have been saved to {weights_path}")

        # Save the full model checkpoint
        model.save(checkpoint_path)
        print(f"Full model checkpoint has been saved to {checkpoint_path}")

    save_model_weights_and_checkpoint(model)    

if __name__ == '__main__':
    example_run()
