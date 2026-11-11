
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Sample customer dataset
data = {
    "tenure": [2, 5, 12, 24, 36, 48, 3, 8, 18, 30],
    "monthly_charges": [80, 90, 60, 50, 45, 40, 95, 85, 65, 55],
    "support_calls": [5, 4, 2, 1, 0, 0, 6, 4, 2, 1],
    "churn": [1, 1, 0, 0, 0, 0, 1, 1, 0, 0]
}

df = pd.DataFrame(data)

# Separate features and target
X = df.drop("churn", axis=1)
y = df["churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Build ML pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Predict a new customer
new_customer = pd.DataFrame([{
    "tenure": 4,
    "monthly_charges": 90,
    "support_calls": 5
}])

churn_prediction = model.predict(new_customer)

print("Churn Prediction:", churn_prediction[0])
