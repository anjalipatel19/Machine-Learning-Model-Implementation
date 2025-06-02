# Machine-Learning-Model-Implementation

A machine learning model that classifies patient health data to predict diabetes using feature scaling and a Random Forest algorithm.

**COMPANY:** CODETECH IT SOLUTIONS

**Name:** Anjali Dilip Patel

**INTERN ID:** CT06DL927

**Domain:** Python Programming

**BATCH Duration:** 6 Weeks

**Mentor:** Neela Santhosh Kumar

**PROJECT:** MACHINE LEARNING MODEL IMPLEMENTATION

# Diabetes Prediction Model

## Overview

This project implements a machine learning model to classify whether a patient is diabetic based on medical features. It demonstrates the end-to-end process of data preprocessing, model training, and performance evaluation using scikit-learn.

## Key Features

* Data preprocessing including replacing invalid zero values
* Feature scaling using StandardScaler
* Random Forest classification model
* Confusion matrix and accuracy score for evaluation
* Prediction on new patient data

## Technical Implementation

### Data Processing

* Input data format: Excel file with 9 columns
* Columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
* Zero values in some columns are replaced with the median to improve model performance

### Feature Engineering

* StandardScaler is used to normalize feature values
* Ensures that all input features are on a similar scale
* Prevents bias towards higher numerical features

### Model Training

* Algorithm: Random Forest Classifier with 100 estimators
* Trained on 80% of the dataset using `train_test_split`
* Produces a robust model capable of predicting diabetes based on medical features

### Evaluation Metrics

* Accuracy score calculated on the 20% test set
* Confusion matrix plotted using matplotlib and seaborn
* Example prediction shows whether a sample patient is diabetic or not

## How to Use

### Requirements

* Python 3.x
* Required packages: pandas, scikit-learn, matplotlib, seaborn, openpyxl

Install dependencies with:

```
pip install pandas scikit-learn matplotlib seaborn openpyxl
```

### Running the Project

1. Ensure the `diabetes_data.xlsx` file is in the same directory
2. Run the script:

```
python diabetes_prediction.py
```

## Output:

* Accuracy score printed in terminal
* Confusion matrix saved or displayed as an image
* Final prediction output for a sample patient with probability

![image](https://github.com/user-attachments/assets/6f4d05bf-6e72-45a0-bac1-6c5678b3452a)


## License

* *This project*: [MIT License](LICENSE) – Free to use, modify, and distribute
