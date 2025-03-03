import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/classification_data.csv")
X = data.drop(columns=["price_2"])
y = data["price_2"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Define pipeline
classification_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)

# Train model
classification_pipeline.fit(X_train, y_train)

# Save pipeline
joblib.dump(classification_pipeline, "models/classification_pipeline.pkl")

print("Classification pipeline saved successfully!")
