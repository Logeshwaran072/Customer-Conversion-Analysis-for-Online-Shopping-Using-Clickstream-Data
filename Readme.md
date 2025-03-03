# Customer Conversion Analysis Using Clickstream Data

## 📌 Project Overview

This project focuses on analyzing customer behavior using **clickstream data** from an e-commerce platform. It applies **classification, regression, and clustering models** to improve customer conversion rates, predict potential revenue, and segment users for personalized marketing.

🚀 **Key Features:**

- **Classification:** Predict if a customer will complete a purchase.
- **Regression:** Estimate revenue based on user behavior.
- **Clustering:** Group users for targeted marketing strategies.
- **Streamlit Dashboard:** Interactive web app for predictions and insights.

---

## 📂 Folder Structure

```
Clickstream_Project/
│-- data/                  # Contains datasets
│   ├── train.csv          # Training dataset
│   ├── test.csv           # Testing dataset
│
│-- models/                # Trained machine learning models
│   ├── classification_model_xgboost.pkl
│   ├── regression_model_GBR.pkl
│   ├── clustering_model.pkl
│
│-- notebooks/             # Jupyter notebooks for model development
│   ├── classification.ipynb
│   ├── regression.ipynb
│   ├── clustering.ipynb
│   ├── preprocessing.ipynb
│
│-- source/                # Machine learning pipelines
│   ├── classification_pipeline.py
│   ├── regression_pipeline.py
│   ├── evaluate_pipeline.py
│
│-- streamlit/             # Streamlit app files
│   ├── app.py             # Main Streamlit dashboard
│   ├── preprocessing_streamlit.py
│
│-- README.md              # Project documentation
```

---

## 🛠️ Installation

1️⃣ **Clone the Repository**

```bash
git clone https://github.com/your-username/clickstream-project.git
cd clickstream-project
```

2️⃣ **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### **1️⃣ Preprocess Data**

Run the preprocessing script to clean and prepare the dataset:

```bash
python source/preprocessing_pipeline.py
```

### **2️⃣ Train Machine Learning Models**

Execute notebooks in `notebooks/` to train and evaluate models.

### **3️⃣ Run Streamlit App**

Launch the interactive dashboard:

```bash
streamlit run streamlit/app.py
```

---

## 🌐 Streamlit Webpage Usage Instructions

1️⃣ **Upload Data:**

- You can upload either a **raw dataset** or a **preprocessed dataset** in their respective fields.
- Navigate to the **Main Page** in the Streamlit app.
- Use the **Upload CSV** option to upload your dataset.

2️⃣ **Preprocess Data:**

- Select the **problem type** (Classification, Regression, or Clustering).
- Click the **Preprocess Data** button to clean and transform the dataset.
- Once preprocessing is completed, you will have the option to **download the preprocessed dataset**.

3️⃣ **Run Predictions:**

- Click **Run Prediction** after preprocessing is complete.
- Navigate to the respective model page (**Classification, Regression, or Clustering**).

4️⃣ **Reset Dataset:**

- If you want to upload a new dataset, use the **Reset** button to clear the current session and start fresh.

5️⃣ **View Results & Download Data:**

- View predictions, metrics, and visualizations.
- Download the **preprocessed dataset** if needed.

---

## 🛠️ Machine Learning Models

| **Model Type**     | **Algorithms Used**                                                          |
| ------------------ | ---------------------------------------------------------------------------- |
| **Classification** | Logistic Regression, Decision Trees, Random Forest, XGBoost, Neural Networks |
| **Regression**     | Linear Regression, Ridge, Lasso, Gradient Boosting Regressors                |
| **Clustering**     | K-Means, DBSCAN, Hierarchical Clustering                                     |

---

## 📊 Evaluation Metrics

- **Classification:** Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- **Regression:** RMSE, MAE, R² Score.
- **Clustering:** Silhouette Score, Davies-Bouldin Index.

---

## 📌 Business Use Cases

✔ **Customer Conversion Prediction** → Identify potential buyers.\
✔ **Revenue Forecasting** → Predict customer spending behavior.\
✔ **Customer Segmentation** → Enable personalized marketing.\
✔ **Churn Reduction** → Detect users likely to abandon carts.\
✔ **Product Recommendations** → Suggest relevant items based on browsing patterns.

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

## 🤝 Contributing

- Feel free to submit **issues** or **pull requests**.
- Follow **PEP8** coding guidelines.
- Use GitHub for **version control**.

---


