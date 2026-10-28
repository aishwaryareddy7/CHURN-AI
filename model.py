import os

import pandas as pd
import numpy as np

from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# DATASET PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ff.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset loaded successfully: "
    f"{df.shape[0]} rows, {df.shape[1]} columns"
)


# ============================================================
# PREPROCESSING
# ============================================================

data = df.copy()

customer_ids = data["customerID"].copy()

data = data.drop(
    columns=["customerID"]
)


# Convert TotalCharges to numeric

data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"]
    .astype(str)
    .str.strip(),
    errors="coerce"
).fillna(0)


# Convert target

data["Churn"] = data["Churn"].map({
    "Yes": 1,
    "No": 0
})


# ============================================================
# SERVICE VALUE CLEANING
# ============================================================

data["MultipleLines"] = data[
    "MultipleLines"
].replace({
    "No phone service": "No"
})


service_columns = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]


for column in service_columns:

    data[column] = data[column].replace({
        "No internet service": "No"
    })


# ============================================================
# YES / NO ENCODING
# ============================================================

yes_no_columns = [
    column
    for column in data.columns
    if set(
        data[column]
        .dropna()
        .unique()
    ).issubset({"Yes", "No"})
]


for column in yes_no_columns:

    data[column] = data[column].map({
        "Yes": 1,
        "No": 0
    })


# ============================================================
# GENDER ENCODING
# ============================================================

data["gender_Male"] = (
    data["gender"] == "Male"
).astype(int)


data = data.drop(
    columns=["gender"]
)


# ============================================================
# ONE-HOT ENCODING
# ============================================================

data = pd.get_dummies(
    data,
    columns=[
        "InternetService",
        "Contract",
        "PaymentMethod"
    ],
    dtype=int
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

target_column = "Churn"


feature_columns = [
    column
    for column in data.columns
    if column != target_column
]


X = data[
    feature_columns
].astype(float)


y = data[
    target_column
].astype(int)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(

    X,
    y,

    test_size=0.40,

    stratify=y,

    random_state=42
)


X_validation, X_test, y_validation, y_test = train_test_split(

    X_temp,
    y_temp,

    test_size=0.50,

    stratify=y_temp,

    random_state=42
)


print(
    "\nDataset Split:"
)

print(
    "Training:",
    len(X_train)
)

print(
    "Validation:",
    len(X_validation)
)

print(
    "Test:",
    len(X_test)
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

logistic_model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "model",
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    )

])


# ============================================================
# RANDOM FOREST
# ============================================================

random_forest_model = RandomForestClassifier(

    n_estimators=400,

    min_samples_leaf=3,

    class_weight="balanced_subsample",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# XGBOOST
# ============================================================

negative_count = (
    y_train == 0
).sum()


positive_count = (
    y_train == 1
).sum()


scale_pos_weight = (
    negative_count /
    positive_count
)


xgboost_model = XGBClassifier(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=4,

    subsample=0.8,

    colsample_bytree=0.8,

    scale_pos_weight=scale_pos_weight,

    eval_metric="logloss",

    random_state=42
)


# ============================================================
# TRAIN BASELINE MODELS
# ============================================================

print(
    "\nTraining Logistic Regression..."
)

logistic_model.fit(
    X_train,
    y_train
)


print(
    "Training Random Forest..."
)

random_forest_model.fit(
    X_train,
    y_train
)


print(
    "Training XGBoost..."
)

xgboost_model.fit(
    X_train,
    y_train
)


# ============================================================
# TABPFN
# ============================================================

TABPFN_TOKEN = os.getenv(
    "TABPFN_TOKEN"
)


tabpfn_available = False

tabpfn_model = None


if TABPFN_TOKEN:

    try:

        from tabpfn_client import (
            TabPFNClassifier
        )


        print(
            "\nInitializing TabPFN..."
        )


        tabpfn_model = (
            TabPFNClassifier()
        )


        print(
            "Training TabPFN..."
        )


        tabpfn_model.fit(
            X_train,
            y_train
        )


        tabpfn_available = True


        print(
            "TabPFN model trained successfully."
        )


    except Exception as e:

        print(
            "\nTabPFN initialization/training failed."
        )

        print(
            "Reason:",
            str(e)
        )


        tabpfn_available = False

        tabpfn_model = None


else:

    print(
        "\nTabPFN token not found."
    )


# ============================================================
# PRIMARY AND BACKUP MODEL
# ============================================================

PRIMARY_MODEL = "TabPFN"

BACKUP_MODEL = "XGBoost"


if tabpfn_available:

    ACTIVE_MODEL = PRIMARY_MODEL

    print(
        "\nPrimary Model:",
        PRIMARY_MODEL
    )

    print(
        "Backup Model:",
        BACKUP_MODEL
    )

    print(
        "TabPFN Status: Available"
    )


else:

    ACTIVE_MODEL = BACKUP_MODEL

    print(
        "\nPrimary Model:",
        PRIMARY_MODEL
    )

    print(
        "Backup Model:",
        BACKUP_MODEL
    )

    print(
        "TabPFN Status: Unavailable"
    )

    print(
        "Fallback Model Activated:",
        BACKUP_MODEL
    )


# ============================================================
# MODEL DICTIONARY
# ============================================================

models = {

    "Logistic Regression":
        logistic_model,

    "Random Forest":
        random_forest_model,

    "XGBoost":
        xgboost_model
}


if (
    tabpfn_available
    and tabpfn_model is not None
):

    models["TabPFN"] = (
        tabpfn_model
    )


# ============================================================
# MODEL STATUS
# ============================================================

def get_model_status():

    return {

        "primary_model":
            PRIMARY_MODEL,

        "backup_model":
            BACKUP_MODEL,

        "active_model":
            ACTIVE_MODEL,

        "tabpfn_available":
            tabpfn_available,

        "dataset_rows":
            len(df),

        "dataset_columns":
            len(df.columns),

        "feature_count":
            len(feature_columns)
    }


# ============================================================
# FINAL SETUP MESSAGE
# ============================================================

print(
    "\n----------------------------------------"
)

print(
    "ML MODEL SETUP COMPLETE"
)

print(
    "----------------------------------------"
)

print(
    "Dataset:",
    len(df),
    "customers"
)

print(
    "Features:",
    len(feature_columns)
)

print(
    "Training rows:",
    len(X_train)
)

print(
    "Validation rows:",
    len(X_validation)
)

print(
    "Test rows:",
    len(X_test)
)

print(
    "Active Model:",
    ACTIVE_MODEL
)

print(
    "----------------------------------------"
)