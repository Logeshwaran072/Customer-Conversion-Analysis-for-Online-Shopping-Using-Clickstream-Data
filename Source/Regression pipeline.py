import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/regression_data.csv")
X = data.drop(columns=["price"])  # Replace 'target' with actual column name
y = data["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Define pipeline
regression_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
)

# Train model
regression_pipeline.fit(X_train, y_train)

# Save pipeline
joblib.dump(regression_pipeline, "models/regression_pipeline.pkl")

print("Regression pipeline saved successfully!")
