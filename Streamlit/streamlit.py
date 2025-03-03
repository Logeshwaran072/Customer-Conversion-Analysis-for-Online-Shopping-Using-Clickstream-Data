import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from preprocessing_streamlit import preprocess_data

# Import preprocessing function
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# Load trained models
classification_model = joblib.load("model/classification_model_xgboost.pkl")
regression_model = joblib.load("model/Clickstream_regression_model_GBR.pkl")
clustering_model = joblib.load("model/Kmeans_model.pkl")

# Configure page layout
st.set_page_config(page_title="Customer Analytic Dashboard", layout="wide")

# ✅ **Fix: Initialize session state variables**
if "model_type" not in st.session_state:
    st.session_state["model_type"] = "Classification"  # Default value


# Function to Reset Everything
def reset_all():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# Sidebar Navigation
if "navigate_to" in st.session_state:
    selected_page = st.session_state["navigate_to"]
    del st.session_state["navigate_to"]
else:
    selected_page = "Main Page"

page = st.sidebar.radio(
    "Go to",
    ["Main Page", "Classification", "Regression", "Clustering"],
    index=["Main Page", "Classification", "Regression", "Clustering"].index(
        selected_page
    ),
)

# 🏠 **Main Page**
if page == "Main Page":
    st.title("📊 Customer Analytic Dashboard")

    if "uploaded" in st.session_state or "uploaded_preprocessed" in st.session_state:
        if st.button("🔄 Reset & Upload New File"):
            reset_all()

    st.sidebar.write("### 📂 Upload Data")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Raw CSV (for preprocessing)", type=["csv"]
    )
    uploaded_preprocessed = st.sidebar.file_uploader(
        "Upload Preprocessed CSV (for direct prediction)", type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["df"] = df
            st.session_state["uploaded"] = True
        except Exception as e:
            st.sidebar.error(f"🚨 Error reading file: {e}")

    if uploaded_preprocessed is not None:
        try:
            df_processed = pd.read_csv(uploaded_preprocessed)
            st.session_state["df_processed"] = df_processed
            st.session_state["uploaded_preprocessed"] = True
        except Exception as e:
            st.sidebar.error(f"🚨 Error reading preprocessed file: {e}")

    if "df" in st.session_state and st.session_state["uploaded"]:
        df = st.session_state["df"]
        st.write("### 📌 Raw Data Preview:")
        st.dataframe(df.head())

        st.sidebar.write("### 🔍 Select Analysis Type & Preprocess")

        model_type = st.sidebar.radio(
            "What type of analysis do you want to perform?",
            ["Classification", "Regression", "Clustering"],
            index=["Classification", "Regression", "Clustering"].index(
                st.session_state.get("model_type", "Classification")
            ),
        )

        if st.sidebar.button("▶ Preprocess Data"):
            st.sidebar.write("🔄 Processing Data...")
            df_processed = preprocess_data(
                df, problem_type=model_type, train_mode=False
            )

            if df_processed is None or df_processed.empty:
                st.sidebar.error(
                    "🚨 Error: Processed data is empty. Please check your input file."
                )
            else:
                st.session_state["df_processed"] = df_processed
                st.session_state["model_type"] = model_type
                st.session_state["preprocessing_done"] = True

    if "df_processed" in st.session_state:
        st.write("### ✅ Preprocessed Data")
        st.dataframe(st.session_state["df_processed"].head())

        # ✅ **Download Preprocessed File**
        csv = st.session_state["df_processed"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Preprocessed Data",
            data=csv,
            file_name="preprocessed_data.csv",
            mime="text/csv",
        )

        if (
            st.session_state.get("preprocessing_done", False)
            or "uploaded_preprocessed" in st.session_state
        ):
            st.success(
                f"✅ Preprocessing is done! Click below to run prediction and move to the **{st.session_state.get('model_type', 'Classification')}** page."
            )

            if st.button("▶ Run Prediction"):
                st.session_state["navigate_to"] = st.session_state["model_type"]
                st.session_state["run_prediction"] = True
                st.rerun()


## 🚀 **Classification Model Page**
if page == "Classification":
    st.title("🏷 Classification Model")

    # ✅ Check if preprocessed data is available
    if "df_processed" in st.session_state and "run_prediction" in st.session_state:
        try:
            df = st.session_state["df_processed"]

            # ✅ Define X (features) and y (target)
            X = df.drop(columns=["price_2"])
            y = df["price_2"]

            # ✅ Convert labels {1,2} → {0,1} if needed
            y = np.where(y == 2, 1, 0)

            # ✅ Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # ✅ Align column order with the model
            X_test = X_test[classification_model.feature_names_in_]

            # ✅ Predict on test set
            y_pred = classification_model.predict(X_test)
            y_proba = classification_model.predict_proba(X_test)[:, 1]

            # ✅ Store results
            st.session_state["classification_result"] = y_pred
            st.session_state["classification_proba"] = y_proba
            st.session_state["y_test"] = y_test

            del st.session_state["run_prediction"]

        except Exception as e:
            st.error(f"⚠️ Error in model prediction: {e}")

    # ✅ Display results if available
    if "classification_result" in st.session_state:
        y_pred = st.session_state["classification_result"]
        y_test = st.session_state["y_test"]
        y_proba = st.session_state["classification_proba"]

        # ✅ **Classification Report**
        st.subheader("📄 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # ✅ **Confusion Matrix**
        st.subheader("🟦 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[0, 1],
            yticklabels=[0, 1],
        )
        st.pyplot(fig)

        # ✅ **AUC-ROC Curve**
        st.subheader("📈 AUC-ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

        # ✅ **Histogram of Predictions**
        st.subheader("📊 Prediction Distribution (Histogram)")
        fig, ax = plt.subplots()
        ax.hist(y_pred, bins=2, edgecolor="black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Class 0", "Class 1"])
        st.pyplot(fig)

        # ✅ **Boxplot of Probability Scores**
        st.subheader("📦 Probability Score Distribution (Boxplot)")
        fig, ax = plt.subplots()
        sns.boxplot(y=y_proba)
        st.pyplot(fig)

        # ✅ **Pie Chart of Class Distribution**
        st.subheader("🥧 Class Distribution (Pie Chart)")
        class_counts = np.bincount(y_pred)
        fig, ax = plt.subplots()
        ax.pie(
            class_counts,
            labels=["Class 0", "Class 1"],
            autopct="%1.1f%%",
            colors=["lightblue", "lightcoral"],
        )
        st.pyplot(fig)


# 🚀 **Regression Model Page**
if page == "Regression":
    st.title("📈 Regression Model")

    # ✅ Check if preprocessed data is available
    if "df_processed" in st.session_state and "run_prediction" in st.session_state:
        try:
            df = st.session_state["df_processed"]

            # ✅ Define X (features) and y (target)
            X = df.drop(columns=["price"])
            y = df["price"]

            # ✅ Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # ✅ Align column order with the model
            X_train = X_train[regression_model.feature_names_in_]
            X_test = X_test[regression_model.feature_names_in_]

            # ✅ Predict on train and test sets
            y_train_pred = regression_model.predict(X_train)
            y_test_pred = regression_model.predict(X_test)

            # ✅ Store results
            st.session_state["regression_train_result"] = y_train_pred
            st.session_state["regression_test_result"] = y_test_pred
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test

            del st.session_state["run_prediction"]

        except Exception as e:
            st.error(f"⚠️ Error in model prediction: {e}")

    # ✅ Display results if available
    if "regression_train_result" in st.session_state:
        y_train = st.session_state["y_train"]
        y_test = st.session_state["y_test"]
        y_train_pred = st.session_state["regression_train_result"]
        y_test_pred = st.session_state["regression_test_result"]

        # ✅ **Regression Metrics**
        st.subheader("📊 Regression Metrics")

        metrics_train = {
            "Train R² Score": r2_score(y_train, y_train_pred),
            "Train MSE": mean_squared_error(y_train, y_train_pred),
            "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "Train MAE": mean_absolute_error(y_train, y_train_pred),
        }
        metrics_test = {
            "Test R² Score": r2_score(y_test, y_test_pred),
            "Test MSE": mean_squared_error(y_test, y_test_pred),
            "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "Test MAE": mean_absolute_error(y_test, y_test_pred),
        }

        st.write("📌 **Train Metrics:**", metrics_train)
        st.write("📌 **Test Metrics:**", metrics_test)

        # ✅ **Histogram of Predictions**
        st.subheader("📊 Prediction Distribution (Histogram)")
        fig, ax = plt.subplots()
        ax.hist(
            y_test_pred, bins=20, edgecolor="black", alpha=0.7, label="Test Predictions"
        )
        ax.hist(
            y_train_pred,
            bins=20,
            edgecolor="black",
            alpha=0.7,
            label="Train Predictions",
        )
        ax.legend()
        st.pyplot(fig)

        # ✅ **Boxplot of Predictions**
        st.subheader("📦 Prediction Distribution (Boxplot)")
        fig, ax = plt.subplots()
        sns.boxplot(data=[y_train_pred, y_test_pred], showfliers=False)
        ax.set_xticklabels(["Train Predictions", "Test Predictions"])
        st.pyplot(fig)

        # ✅ **Scatter Plot: Actual vs Predicted**
        st.subheader("📍 Actual vs Predicted (Scatter Plot)")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_test_pred, alpha=0.5, label="Test")
        ax.scatter(y_train, y_train_pred, alpha=0.5, label="Train")
        ax.plot(
            [y.min(), y.max()], [y.min(), y.max()], "r--"
        )  # Line of perfect predictions
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.legend()
        st.pyplot(fig)

        # ✅ **Residual Plot**
        st.subheader("📉 Residuals Distribution (Residual Plot)")
        fig, ax = plt.subplots()
        residuals_train = y_train - y_train_pred
        residuals_test = y_test - y_test_pred
        ax.scatter(y_train_pred, residuals_train, alpha=0.5, label="Train Residuals")
        ax.scatter(y_test_pred, residuals_test, alpha=0.5, label="Test Residuals")
        ax.axhline(y=0, color="r", linestyle="--")
        ax.set_xlabel("Predicted Price")
        ax.set_ylabel("Residuals")
        ax.legend()
        st.pyplot(fig)

        # ✅ **Pie Chart of Error Distribution**
        st.subheader("🥧 Error Distribution (Pie Chart)")
        errors = [np.abs(residuals_train).mean(), np.abs(residuals_test).mean()]
        labels = ["Train MAE", "Test MAE"]
        fig, ax = plt.subplots()
        ax.pie(
            errors, labels=labels, autopct="%1.1f%%", colors=["lightblue", "lightcoral"]
        )
        st.pyplot(fig)


# 🚀 **Clustering Model Page**
if page == "Clustering":
    st.title("🔗 KMeans Clustering Model")

    # ✅ Check if preprocessed data is available
    if "df_processed" in st.session_state and "run_prediction" in st.session_state:
        try:
            df = st.session_state["df_processed"]

            # ✅ Define X (features) by removing the target column (if present)
            X = df.drop(columns=["price"], errors="ignore")

            # ✅ Automatically Select Best k using Elbow Method
            distortions = []
            K_range = range(2, 11)  # Testing k values from 2 to 10
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
                kmeans.fit(X)
                distortions.append(
                    kmeans.inertia_
                )  # Inertia measures clustering quality

            # ✅ Plot Elbow Method
            st.subheader("📈 Elbow Method to Select k")
            fig, ax = plt.subplots()
            ax.plot(K_range, distortions, marker="o", linestyle="-")
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("Inertia")
            ax.set_title("Elbow Method for Optimal k")
            st.pyplot(fig)

            # ✅ Choosing best k (Change as per need)
            optimal_k = 4  # Manually select based on the elbow point
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
            cluster_labels = kmeans.fit_predict(X)

            # ✅ Calculate Silhouette Score (for clustering quality)
            silhouette_avg = silhouette_score(X, cluster_labels)

            # ✅ Store results
            st.session_state["cluster_labels"] = cluster_labels
            st.session_state["X_cluster"] = X

            st.success(
                f"✅ Clustering Done! Optimal k: {optimal_k}, Silhouette Score: {silhouette_avg:.2f}"
            )

            del st.session_state["run_prediction"]

        except Exception as e:
            st.error(f"⚠️ Error in clustering: {e}")

    # ✅ Display results if available
    if "cluster_labels" in st.session_state:
        cluster_labels = st.session_state["cluster_labels"]
        X = st.session_state["X_cluster"]

        st.subheader("🔢 Cluster Labels")
        st.dataframe(pd.DataFrame({"Cluster": cluster_labels}))

        # ✅ Histogram of Clusters
        st.subheader("📊 Cluster Count (Histogram)")
        fig, ax = plt.subplots()
        ax.hist(cluster_labels, bins=np.unique(cluster_labels).size, edgecolor="black")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # ✅ Pie Chart of Cluster Distribution
        st.subheader("🥧 Cluster Distribution")
        cluster_counts = pd.Series(cluster_labels).value_counts()
        fig, ax = plt.subplots()
        ax.pie(
            cluster_counts,
            labels=cluster_counts.index,
            autopct="%1.1f%%",
            colors=sns.color_palette("Set2"),
        )
        st.pyplot(fig)

        # ✅ Boxplot of Clusters
        st.subheader("📦 Boxplot of Clusters")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=cluster_labels, y=X.iloc[:, 0], ax=ax)  # First Feature Boxplot
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Feature Value")
        st.pyplot(fig)

        # ✅ Scatter Plot (First 2 Principal Components)
        st.subheader("📍 Cluster Scatter Plot (PCA)")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette="Set2"
        )
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("Cluster Visualization")
        st.pyplot(fig)

        # ✅ Count Plot
        st.subheader("📊 Cluster Count Plot")
        fig, ax = plt.subplots()
        sns.countplot(x=cluster_labels, palette="Set2")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Count")
        st.pyplot(fig)
