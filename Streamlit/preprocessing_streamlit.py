import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Define Paths
RAW_DATA_PATH = "D:/Project/Clickstream_Project/Clickstream/Data/train_data.csv"
PREPROCESSING_PIPELINE_PATH = (
    "D:/Project/Clickstrenew am_Project/Clickstream/model/preprocessing_pipeline.pkl"
)


# Load Data
def load_data():
    """Loads training data from CSV."""
    return pd.read_csv(RAW_DATA_PATH)


# Convert Category columns
def convert_col_to_category(df):
    categorical_cols = [
        "country",
        "page1_main_category",
        "page2_clothing_model",
        "colour",
        "location",
        "model_photography",
        "price_2",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


# Identify Column Types
def get_feature_types(df):
    """Returns numerical and categorical feature lists."""
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    return numerical_features, categorical_features


# Feature Engineering
def feature_engineering(df):
    """Creates features from existing columns."""
    df["date"] = pd.to_datetime(
        df[["year", "month", "day"]].astype(str).agg("-".join, axis=1), errors="coerce"
    )

    df["sessions_per_day"] = df.groupby("date")["session_id"].transform("nunique")
    df["page_views_per_session_id"] = df.groupby("session_id")["page"].transform(
        "count"
    )
    df["is_bounce"] = df["page_views_per_session_id"] == 1
    df["is_revisit"] = df.groupby("session_id").cumcount() > 0
    df["is_exit"] = (
        df.groupby("session_id").cumcount()
        == df.groupby("session_id")["session_id"].transform("count") - 1
    )
    df["exit_rate"] = round(
        df.groupby("session_id")["is_exit"].sum()
        / df.groupby("session_id")["session_id"].transform("count"),
        2,
    )
    df["exit_rate"] = df["exit_rate"].fillna(0)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["time_spent_per_category"] = df.groupby(["session_id", "page1_main_category"])[
        "page"
    ].transform("count")
    df["total_clicks_per_session_id"] = df.groupby("session_id")["order"].transform(
        "max"
    )

    df = df.sort_values(by=["session_id", "order"])
    df["browsing_path"] = df.groupby("session_id")["page2_clothing_model"].transform(
        lambda x: "->".join(x.astype(str))
    )

    return df


# Convert boolean to int
def convert_bool_to_int(df):
    """Converts boolean columns to integer (0/1)."""
    for col in df.columns:
        if df[col].dtypes == "bool":
            df[col] = df[col].astype("int64")
    return df


# Encoding
def encoding(df):
    """Performs categorical encoding."""
    df["page2_grouped"] = df["page2_clothing_model"].str[0]
    df["page2_grouped"] = (
        df["page2_grouped"]
        .map({"A": "Category 1", "B": "Category 2", "C": "Category 3"})
        .fillna("Other")
    )
    df = pd.get_dummies(df, columns=["page2_grouped"])

    # Compute thresholds as integers
    country_counts = df["country"].value_counts()
    colour_counts = df["colour"].value_counts()

    # Step 2: Define a threshold for rare categories
    country_threshold = 200
    colour_threshold = 7100

    # Step 3: Identify rare categories and map them to "Other"
    df["country"] = df["country"].apply(
        lambda x: x if country_counts[x] >= country_threshold else "Other"
    )
    df["colour"] = df["colour"].apply(
        lambda x: x if colour_counts[x] >= colour_threshold else "Other"
    )

    # Step 4: One-Hot Encoding for country and colour columns
    df = pd.get_dummies(df, columns=["country", "colour"], drop_first=True)

    # One-Hot Encoding
    df = pd.get_dummies(
        df,
        columns=[
            "location",
            "model_photography",
            "page1_main_category",
        ],
        drop_first=True,
    )

    df["path_length"] = df["browsing_path"].str.count("->") + 1

    # first Page

    df["first_page"] = df["browsing_path"].str.split("->").str[0]

    # last page

    df["last_page"] = df["browsing_path"].str.split("->").str[-1]

    for col in ["first_page", "last_page"]:
        freq_encoding = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq_encoding)

    return df


# Encoding
def drop_columns(df):
    """Drops unnecessary columns."""
    columns_to_drop = [
        "session_id",
        "page2_clothing_model",
        "date",
        "browsing_path",
        "first_page",
        "last_page",
        "year",
    ]
    df = df.drop(columns=columns_to_drop, errors="ignore")
    return df


def scaling(df, problem_type):
    class_col_to_scale = [
        "price",
        "sessions_per_day",
        "page_views_per_session_id",
        "path_length",
        "first_page_freq",
        "last_page_freq",
        "exit_rate",
        "time_spent_per_category",
        "total_clicks_per_session_id",
    ]

    reg_col_to_scale = [
        "order",
        "sessions_per_day",
        "page_views_per_session_id",
        "path_length",
        "first_page_freq",
        "last_page_freq",
        "exit_rate",
        "time_spent_per_category",
        "total_clicks_per_session_id",
    ]

    scaler = StandardScaler()

    # Modify the scaling list based on problem type
    if problem_type == "Classification" or problem_type == "Clustering":
        exclude_col = "price_2"  # Replace with actual column name
        class_col_to_scale = [
            col for col in class_col_to_scale if col != exclude_col
        ]  # Remove from scaling
        df[class_col_to_scale] = scaler.fit_transform(df[class_col_to_scale])

    elif problem_type == "Regression":
        exclude_col = "price"  # Replace with actual column nameS
        reg_col_to_scale = [col for col in reg_col_to_scale if col != exclude_col]
        df[reg_col_to_scale] = scaler.fit_transform(df[reg_col_to_scale])

    return df  # Return the modified dataframe


def preprocess_data(df, problem_type=None, train_mode=True):
    df = convert_col_to_category(df)
    df = feature_engineering(df)
    df = convert_bool_to_int(df)
    df = encoding(df)
    df = convert_bool_to_int(df)
    df = drop_columns(df)
    df = scaling(df, problem_type)

    # Define hardcoded feature lists for each model
    classification_features = [
        "order",
        "page",
        "is_bounce",
        "exit_rate",
        "page2_grouped_Category 1",
        "page2_grouped_Category 3",
        "page2_grouped_Other",
        "country_21",
        "country_24",
        "country_29",
        "country_41",
        "country_44",
        "colour_3",
        "colour_4",
        "colour_6",
        "colour_9",
        "colour_14",
        "colour_Other",
        "location_2",
        "location_3",
        "location_4",
        "location_5",
        "location_6",
        "price_2",
    ]

    regression_features = [
        "order",
        "page",
        "page2_grouped_Category 1",
        "country_24",
        "country_44",
        "colour_3",
        "colour_4",
        "colour_6",
        "colour_9",
        "colour_14",
        "colour_Other",
        "location_2",
        "location_3",
        "location_4",
        "location_5",
        "location_6",
        "model_photography_2",
        "page1_main_category_2",
        "page1_main_category_3",
        "first_page_freq",
        "last_page_freq",
        "price",
    ]

    clustering_features = [
        "month",
        "day",
        "order",
        "price",
        "price_2",
        "page",
        "sessions_per_day",
        "page_views_per_session_id",
        "is_bounce",
        "is_revisit",
        "is_exit",
        "exit_rate",
        "day_of_week",
        "time_spent_per_category",
        "total_clicks_per_session_id",
        "page2_grouped_Category 1",
        "page2_grouped_Category 2",
        "page2_grouped_Category 3",
        "page2_grouped_Other",
        "country_16",
        "country_21",
        "country_24",
        "country_29",
        "country_34",
        "country_41",
        "country_44",
        "country_46",
        "country_Other",
        "colour_3",
        "colour_4",
        "colour_6",
        "colour_9",
        "colour_14",
        "colour_Other",
        "location_2",
        "location_3",
        "location_4",
        "location_5",
        "location_6",
        "model_photography_2",
        "page1_main_category_2",
        "page1_main_category_3",
        "page1_main_category_4",
        "path_length",
        "first_page_freq",
        "last_page_freq",
    ]

    # Select the correct feature list based on problem type
    if problem_type == "Classification":
        feature_list = classification_features
    elif problem_type == "Regression":
        feature_list = regression_features
    elif problem_type == "Clustering":
        feature_list = clustering_features
    else:
        raise ValueError(
            "Invalid problem type! Must be Classification, Regression, or Clustering."
        )

    # Add missing columns with default value 0
    for feature in feature_list:
        if feature not in df.columns:
            df[feature] = 0

    # Keep only the columns that were used in training (removes unexpected columns)
    df = df[feature_list]

    return df
