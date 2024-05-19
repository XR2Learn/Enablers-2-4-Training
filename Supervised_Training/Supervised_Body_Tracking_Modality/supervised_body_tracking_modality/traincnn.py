# Python code here

from conf import CUSTOM_SETTINGS, MODALITY_FOLDER, COMPONENT_OUTPUT_FOLDER

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
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
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    # Train the Model
    history = model.fit(X_train, y_train_encoded, epochs=5, batch_size=10, validation_data=(X_val, y_val_encoded), verbose=2)

    # Evaluate the Model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)

    from sklearn.metrics import classification_report, confusion_matrix

    # Predict classes on the test set
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    class_names = ['BORED', 'ENGAGED', 'FRUSTRATED', 'UNLABELED']

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



if __name__ == '__main__':
    example_run()
