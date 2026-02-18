import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import sys
from features.feature_extractor import extract_features

def train_model(model_type='logistic_regression'):
    """
    Trains a machine learning model for phishing URL detection with advanced features and scaling.

    Args:
        model_type (str): The type of model to train ('logistic_regression' or 'random_forest').
                          Defaults to 'logistic_regression'.
    """
    print("Loading dataset...")
    try:
        df = pd.read_csv('dataset/phishing_dataset.csv')
    except FileNotFoundError:
        print("Error: dataset/phishing_dataset.csv not found. Please ensure the dataset is in place.")
        return

    print("Extracting features (this may take a while for large datasets)...")
    features_list = []
    labels = []
    # Use a dummy URL to get feature names for DataFrame creation
    dummy_features = extract_features("http://example.com")
    feature_names = list(dummy_features.keys())

    for index, row in df.iterrows():
        url = row['url']
        label = row['label']
        features = extract_features(url)
        # Ensure all features are present, fill missing with 0 if feature_extractor adds new ones
        # This is important if dummy_features is different from actual extracted features due to URL content
        # For our current setup, dummy_features should match the structure.
        features_list.append([features.get(name, 0) for name in feature_names])
        labels.append(1 if label == 'phishing' else 0) # 1 for phishing, 0 for legitimate

    X = pd.DataFrame(features_list, columns=feature_names)
    y = pd.Series(labels)

    # Scale numerical features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y) # stratify for balanced classes

    # Initialize and train the model
    model = None
    if model_type == 'logistic_regression':
        print("Training Logistic Regression model...")
        model = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear') # 'liblinear' often good for smaller datasets and L1/L2
    elif model_type == 'random_forest':
        print("Training Random Forest Classifier model...")
        model = RandomForestClassifier(random_state=42, n_estimators=100) # Increased estimators
    else:
        print(f"Error: Unknown model type '{model_type}'. Please choose 'logistic_regression' or 'random_forest'.")
        return

    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['legitimate', 'phishing']))

    # Save the model and scaler
    model_filename = f'model/{model_type}_model.joblib'
    scaler_filename = 'model/scaler.joblib'
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Model saved to {model_filename}")
    print(f"Scaler saved to {scaler_filename}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_choice = sys.argv[1].lower()
        if model_choice not in ['logistic_regression', 'random_forest']:
            print("Invalid model choice. Please use 'logistic_regression' or 'random_forest'.")
            sys.exit(1)
        train_model(model_choice)
    else:
        train_model('logistic_regression') # Default model
