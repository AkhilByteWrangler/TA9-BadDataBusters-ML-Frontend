import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

# Load dataset
data = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

# Separate features and labels
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Feature engineering: Adding interaction features
X['BMI_Age'] = X['BMI'] * X['Age']  # Interaction between BMI and Age
X['Glucose_BloodPressure'] = X['Glucose'] * X['BloodPressure']  # Interaction between Glucose and Blood Pressure

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing and modeling pipeline
# This pipeline handles missing values, scales features, and applies the RandomForest classifier

# Column transformer to handle numeric preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
            ('scaler', StandardScaler())  # Standardize numerical features
        ]), X.columns)
    ]
)

# Define the pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))  # Use RandomForest as the classifier
])

# Define parameter grid for hyperparameter tuning
# This grid will be used to find the optimal hyperparameters for the RandomForest
param_grid = {
    'classifier__n_estimators': [50, 100, 200],  # Number of trees
    'classifier__max_depth': [5, 10, 15],  # Depth of each tree
    'classifier__min_samples_split': [2, 5, 10]  # Minimum samples to split a node
}

# Use GridSearchCV for hyperparameter tuning with cross-validation
# The model will search through the param_grid and find the best parameters based on ROC AUC score
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Select the best model from the grid search results
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
f1 = f1_score(y_test, y_pred)  # Calculate F1 score
roc_auc = roc_auc_score(y_test, y_pred)  # Calculate ROC AUC score

# Print evaluation metrics
print(f"Model accuracy: {accuracy:.2f}")
print(f"Model F1 Score: {f1:.2f}")
print(f"Model ROC AUC: {roc_auc:.2f}")

# Save the best model to a file using joblib
joblib.dump(best_model, 'diabetes_model_advanced.pkl')
