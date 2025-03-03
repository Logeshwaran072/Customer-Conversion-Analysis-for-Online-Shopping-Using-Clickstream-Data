import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the final evaluation dataset
eval_data = pd.read_csv("path_to_final_evaluation_dataset.csv")

# Preprocess the data (apply same transformations as training)
# Ensure feature selection, encoding, and scaling match the training pipeline
scaler = joblib.load("models/scaler.pkl")
eval_data_scaled = scaler.transform(
    eval_data.drop(columns=["target_column"], errors="ignore")
)

### Classification Model Evaluation ###
clf_model = joblib.load(
    "D:/Project/lickstream_Project/Clickstream/Model/Classification_model_xgboost.pkl"
)
y_pred_clf = clf_model.predict(eval_data_scaled)

y_true_clf = eval_data["target_column"]  # Ensure target column exists in dataset
print("Classification Model Evaluation:")
print("Accuracy:", accuracy_score(y_true_clf, y_pred_clf))
print("Classification Report:\n", classification_report(y_true_clf, y_pred_clf))

### Regression Model Evaluation ###
reg_model = joblib.load(
    "D:/Project/Clickstream_Project/Clickstream/Model/Clickstream_regression_model_GBR.pkl"
)
y_pred_reg = reg_model.predict(eval_data_scaled)

y_true_reg = eval_data["target_column"]
print("\nRegression Model Evaluation:")
print("MSE:", mean_squared_error(y_true_reg, y_pred_reg))
print("R2 Score:", r2_score(y_true_reg, y_pred_reg))

### Clustering Model Evaluation ###
cluster_model = joblib.load(
    "D:/Project/Clickstream_Project/Clickstream/Model/Kmeans_model.pkl"
)
cluster_labels = cluster_model.predict(eval_data_scaled)

eval_data["Cluster"] = cluster_labels  # Assign clusters to new data
print("\nClustering Model Evaluation:")
print("Cluster Distribution:\n", eval_data["Cluster"].value_counts())

# Save clustered results
eval_data.to_csv("results/final_evaluation_with_clusters.csv", index=False)

print("Evaluation completed. Results saved.")
