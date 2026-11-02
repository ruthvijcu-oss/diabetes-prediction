# ============================================================
# PROJECT: DIABETES PREDICTION USING MACHINE LEARNING
# Algorithms: Logistic Regression and Random Forest
# Dataset: Pima Indians Diabetes Dataset (Kaggle)
# Input Features: Glucose, BMI, Insulin, Age
# Metrics: Accuracy, Recall
# ============================================================


# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")


# 2. LOAD DATASET

data = pd.read_csv(r"C:\Users\HP\Downloads\diabetes.csv")

print("\n========== DATASET ==========")
print(data.head())
print("\nDataset Shape:", data.shape)
print("\nDataset Information:")
data.info()
print("\nStatistical Description:")
print(data.describe())


# 3. CHECK MISSING VALUES AND DUPLICATES

print("\n========== DATA QUALITY CHECK ==========")
print("\nMissing Values:")
print(data.isnull().sum())
print("\nDuplicate Records:", data.duplicated().sum())


# 4. SELECT REQUIRED FEATURES (as per assignment: Section-2A)

features = ['Glucose', 'BMI', 'Insulin', 'Age']

selected = data[features + ['Outcome']].copy()


# 5. EXPLORATORY DATA ANALYSIS

# Outcome distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=selected, x='Outcome', hue='Outcome',
              palette='Set2', legend=False)
plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = Non-Diabetic, 1 = Diabetic)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

# Feature distributions by outcome
for col in features:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=selected, x=col, hue='Outcome', kde=True)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Correlation heatmap (selected features + Outcome)
plt.figure(figsize=(8, 6))
sns.heatmap(selected.corr(), annot=True, cmap='coolwarm',
            fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# 6. SEPARATE INPUT AND OUTPUT

X = selected[features]
y = selected['Outcome']

print("\nInput Features:", X.columns.tolist())
print("\nTarget Variable:")
print(y.value_counts())


# 7. DATA PREPROCESSING
# Zero is not a valid measurement for Glucose, BMI or Insulin,
# so these zeros are treated as missing values (NaN).
# Age has no invalid zeros.

zero_columns = ['Glucose', 'BMI', 'Insulin']

X = X.copy()
X[zero_columns] = X[zero_columns].replace(0, np.nan)

print("\nMissing values after zero replacement:")
print(X.isnull().sum())


# 8. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# 9. LOGISTIC REGRESSION MODEL

lr_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000,
                                      class_weight='balanced'))
])

lr_model.fit(X_train, y_train)
lr_prediction = lr_model.predict(X_test)
print("\nLogistic Regression Model Trained Successfully!")


# 10. RANDOM FOREST MODEL

rf_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    ))
])

rf_model.fit(X_train, y_train)
rf_prediction = rf_model.predict(X_test)
print("Random Forest Model Trained Successfully!")


# 11. EVALUATION FUNCTION

def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n========== {name.upper()} RESULTS ==========")
    print("Accuracy :", round(acc * 100, 2), "%")
    print("Recall   :", round(rec * 100, 2), "%")
    print("Precision:", round(prec * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    return acc, prec, rec, f1


lr_accuracy, lr_precision, lr_recall, lr_f1 = evaluate(
    "Logistic Regression", y_test, lr_prediction)
rf_accuracy, rf_precision, rf_recall, rf_f1 = evaluate(
    "Random Forest", y_test, rf_prediction)


# 12. CONFUSION MATRICES

ConfusionMatrixDisplay.from_predictions(
    y_test, lr_prediction,
    display_labels=['Non-Diabetic', 'Diabetic'], cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.show()

ConfusionMatrixDisplay.from_predictions(
    y_test, rf_prediction,
    display_labels=['Non-Diabetic', 'Diabetic'], cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()


# 13. COMPARE BOTH MODELS

results = pd.DataFrame({
    'Algorithm': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [lr_accuracy, rf_accuracy],
    'Recall': [lr_recall, rf_recall],
    'Precision': [lr_precision, rf_precision],
    'F1 Score': [lr_f1, rf_f1]
})

print("\n========== MODEL COMPARISON ==========")
print(results.round(4))

results.set_index('Algorithm').plot(kind='bar', figsize=(10, 6), width=0.7)
plt.title("Logistic Regression vs Random Forest")
plt.xlabel("Machine Learning Algorithm")
plt.ylabel("Performance Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# 14. CROSS-VALIDATION (more reliable than a single small test set)

print("\n========== 5-FOLD CROSS-VALIDATION ==========")

for name, model in [('Logistic Regression', lr_model),
                    ('Random Forest', rf_model)]:
    cv_acc = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_rec = cross_val_score(model, X, y, cv=5, scoring='recall')
    print(f"{name}: Accuracy = {cv_acc.mean():.3f} (+/- {cv_acc.std():.3f}), "
          f"Recall = {cv_rec.mean():.3f} (+/- {cv_rec.std():.3f})")


# 15. RANDOM FOREST FEATURE IMPORTANCE

rf_classifier = rf_model.named_steps['classifier']

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_classifier.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)

plt.figure(figsize=(8, 5))
sns.barplot(data=feature_importance, x='Importance', y='Feature',
            hue='Feature', palette='viridis', legend=False)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


# 16. SAVE TRAINED MODELS

joblib.dump(lr_model, 'logistic_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
print("\nBoth models saved successfully!")


# 17. NEW PATIENT PREDICTION

def predict_new_patient():
    loaded_lr_model = joblib.load('logistic_model.pkl')
    loaded_rf_model = joblib.load('random_forest_model.pkl')

    print("\n======================================")
    print("       DIABETES PREDICTION SYSTEM")
    print("======================================")
    print("\nEnter Patient Medical Information:")

    glucose = float(input("Glucose Level: "))
    bmi = float(input("BMI: "))
    insulin = float(input("Insulin Level: "))
    age = float(input("Age: "))

    # Column order must match the training features
    patient = pd.DataFrame([[glucose, bmi, insulin, age]], columns=features)

    # Treat entered zeros as missing, same as in training
    patient[zero_columns] = patient[zero_columns].replace(0, np.nan)

    print("\n========== FINAL PREDICTION ==========")

    for name, model in [('LOGISTIC REGRESSION', loaded_lr_model),
                        ('RANDOM FOREST', loaded_rf_model)]:
        result = model.predict(patient)[0]
        probability = model.predict_proba(patient)[0][1]
        print(f"\n{name}:")
        print("Prediction:", "Diabetic" if result == 1 else "Non-Diabetic")
        print("Estimated probability of diabetes:",
              round(probability * 100, 2), "%")

    print("\nNote: These are educational model predictions,")
    print("not confirmed medical diagnoses.")


if __name__ == "__main__":
    predict_new_patient()

    print("\n======================================")
    print("       PROJECT EXECUTION COMPLETE")
    print("======================================")