import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# === Function to load and clean data ===
def load_and_prepare_data():
    """Load data from Excel and prepare for modeling"""
    try:
        # Load the dataset from an Excel file
        df = pd.read_excel('diabetes_data.xlsx')

        # Replace 0s in certain columns with the median (as 0 is not realistic for these features)
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            df[col] = df[col].replace(0, df[col].median())  # Replace 0 with median

        return df

    except FileNotFoundError:
        # If the file is not found, print an error message
        print("Error: 'diabetes_data.xlsx' not found")
        print("Please run 'download_diabetes_data.py' first")
        return None

# === Function to train the machine learning model ===
def train_model(X_train, y_train):
    """Train and return a Random Forest model"""
    # Create and train a Random Forest Classifier with 100 trees
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)  # Fit the model with training data
    return model

# === Function to evaluate the trained model ===
def evaluate_model(model, X_test, y_test):
    """Evaluate model and print metrics"""
    y_pred = model.predict(X_test)  # Predict on test data

    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Display results
    print(f"\nModel Accuracy: {accuracy:.2%}")
    print("\nConfusion Matrix:")
    print(cm)

    return accuracy, cm

# === Function to visualize feature importance ===
def plot_feature_importance(model, features):
    """Plot feature importance"""
    importance = model.feature_importances_  # Get feature importance scores
    plt.figure(figsize=(10, 6))
    plt.barh(features, importance)  # Horizontal bar plot
    plt.title('Feature Importance for Diabetes Prediction')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()

# === Function to predict for a new patient ===
def make_prediction(model, scaler, patient_data):
    """Make prediction for a single patient"""
    # Convert input into DataFrame
    patient_df = pd.DataFrame([patient_data])

    # Apply scaling (same as training data)
    patient_scaled = scaler.transform(patient_df)

    # Predict the class and probability of diabetes
    prediction = model.predict(patient_scaled)
    probability = model.predict_proba(patient_scaled)[0][1]

    # Display patient prediction result
    print("\nPatient Details:")
    print(patient_df)
    print(f"\nPrediction: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")
    print(f"Probability: {probability:.2%}")

# === Main execution function ===
def main():
    print("=== Diabetes Prediction Model ===")

    # Step 1: Load and clean the dataset
    df = load_and_prepare_data()
    if df is None:
        return  # Stop if dataset loading failed

    # Step 2: Separate features (X) and target label (y)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Step 3: Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Step 4: Normalize features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
    X_test_scaled = scaler.transform(X_test)        # Transform test data only

    # Step 5: Train the model
    print("\nTraining model...")
    model = train_model(X_train_scaled, y_train)

    # Step 6: Evaluate the model on test set
    print("\nEvaluating model...")
    accuracy, cm = evaluate_model(model, X_test_scaled, y_test)

    # Step 7: Display feature importance graph
    plot_feature_importance(model, X.columns)

    # Step 8: Make a prediction for a new patient
    example_patient = {
        'Pregnancies': 2,
        'Glucose': 120,
        'BloodPressure': 70,
        'SkinThickness': 30,
        'Insulin': 80,
        'BMI': 25,
        'DiabetesPedigreeFunction': 0.3,
        'Age': 35
    }
    make_prediction(model, scaler, example_patient)

# === Entry point for script ===
if __name__ == "__main__":
    main()