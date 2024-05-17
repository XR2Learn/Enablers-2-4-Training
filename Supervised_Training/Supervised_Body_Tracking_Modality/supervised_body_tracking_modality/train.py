# Python code here

from conf import CUSTOM_SETTINGS, MODALITY_FOLDER, COMPONENT_OUTPUT_FOLDER

import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
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
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    X_val = val_data.iloc[:, :-1]
    y_val = val_data.iloc[:, -1]

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_val_encoded = label_encoder.transform(y_val)

    # XGBClassifier
    xgb_model = XGBClassifier(
        eval_metric='mlogloss',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
    )
    xgb_model.fit(X_train, y_train_encoded)

    # Predict the labels on the test and validation set
    y_pred_test = xgb_model.predict(X_test)
    y_pred_val = xgb_model.predict(X_val)
    y_pred_train = xgb_model.predict(X_train)

    # Calculate and print the confusion matrix for the validation set
    print("Confusion Matrix - Validation Set:")
    conf_matrix_val = confusion_matrix(y_val_encoded, y_pred_val)
    print(conf_matrix_val)

    # Calculate and print the classification report for the validation set
    print("\nClassification Report - Validation Set:")
    print(classification_report(y_val_encoded, y_pred_val, target_names=label_encoder.classes_))

    # Calculate and print the confusion matrix for the test set
    print("Confusion Matrix - Test Set:")
    conf_matrix_test = confusion_matrix(y_test_encoded, y_pred_test)
    print(conf_matrix_test)

    # Calculate and print the classification report for the test set
    print("\nClassification Report - Test Set:")
    print(classification_report(y_test_encoded, y_pred_test, target_names=label_encoder.classes_))

    # Calculate accuracies
    train_accuracy = accuracy_score(y_train_encoded, y_pred_train)
    test_accuracy = accuracy_score(y_test_encoded, y_pred_test)
    val_accuracy = accuracy_score(y_val_encoded, y_pred_val)

    # Print accuracies
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")






if __name__ == '__main__':
    example_run()
