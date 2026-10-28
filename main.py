import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from ml.model import (
    models,
    ACTIVE_MODEL,
    PRIMARY_MODEL,
    BACKUP_MODEL,
    tabpfn_available,
    tabpfn_model,
    get_model_status,
    feature_columns,
    df,
    X_test,
    y_test,
)


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Telecom Customer Churn Prediction using TabPFN and XGBoost",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SHAP CACHE
# ============================================================

_shap_explainer = None
_shap_background = None


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running",
        "active_model": ACTIVE_MODEL
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Customer Churn Prediction API"
    }


# ============================================================
# MODEL STATUS
# ============================================================

@app.get("/api/model-status")
def model_status():
    return get_model_status()


# ============================================================
# DATASET SUMMARY
# ============================================================

@app.get("/api/dataset-summary")
def dataset_summary():

    total_customers = len(df)

    churn_count = int(
        (df["Churn"] == "Yes").sum()
    )

    non_churn_count = int(
        (df["Churn"] == "No").sum()
    )

    churn_rate = (
        churn_count / total_customers
        if total_customers > 0
        else 0
    )

    return {
        "total_customers": total_customers,
        "total_features": len(feature_columns),
        "total_columns": len(df.columns),
        "churned_customers": churn_count,
        "retained_customers": non_churn_count,
        "churn_rate": round(churn_rate * 100, 2)
    }


# ============================================================
# FULL DATASET
# ============================================================

@app.get("/api/dataset")
def get_dataset(
    page: int = 1,
    page_size: int = 50,
    search: str = ""
):

    page_size = max(
        10,
        min(page_size, 100)
    )

    dataset = df.copy()

    if search.strip():

        search_value = search.strip().lower()

        mask = dataset.astype(str).apply(
            lambda column:
            column.str.lower().str.contains(
                search_value,
                na=False
            )
        ).any(axis=1)

        dataset = dataset[mask]

    total_rows = len(dataset)

    total_pages = max(
        1,
        int(np.ceil(total_rows / page_size))
    )

    page = max(
        1,
        min(page, total_pages)
    )

    start = (page - 1) * page_size
    end = start + page_size

    page_data = dataset.iloc[start:end]

    records = (
        page_data
        .where(pd.notna(page_data), None)
        .to_dict(orient="records")
    )

    return {
        "data": records,
        "columns": list(df.columns),
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
        "total_pages": total_pages,
        "search": search
    }


# ============================================================
# CUSTOMER INPUT MODEL
# ============================================================

class CustomerData(BaseModel):

    gender: str

    senior_citizen: int

    partner: int

    dependents: int

    tenure: int

    contract: str

    paperless_billing: int

    payment_method: str

    monthly_charges: float

    total_charges: float

    phone_service: int

    multiple_lines: int

    internet_service: str

    online_security: int

    online_backup: int

    device_protection: int

    tech_support: int

    streaming_tv: int

    streaming_movies: int


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_customer_features(
    customer: CustomerData
):

    data = {

        "gender_Male":
            1 if customer.gender == "Male" else 0,

        "SeniorCitizen":
            customer.senior_citizen,

        "Partner":
            customer.partner,

        "Dependents":
            customer.dependents,

        "tenure":
            customer.tenure,

        "PhoneService":
            customer.phone_service,

        "MultipleLines":
            customer.multiple_lines,

        "OnlineSecurity":
            customer.online_security,

        "OnlineBackup":
            customer.online_backup,

        "DeviceProtection":
            customer.device_protection,

        "TechSupport":
            customer.tech_support,

        "StreamingTV":
            customer.streaming_tv,

        "StreamingMovies":
            customer.streaming_movies,

        "PaperlessBilling":
            customer.paperless_billing,

        "MonthlyCharges":
            customer.monthly_charges,

        "TotalCharges":
            customer.total_charges,

        "InternetService_DSL":
            1 if customer.internet_service == "DSL" else 0,

        "InternetService_Fiber optic":
            1 if customer.internet_service == "Fiber optic" else 0,

        "InternetService_No":
            1 if customer.internet_service == "No" else 0,

        "Contract_Month-to-month":
            1 if customer.contract == "Month-to-month" else 0,

        "Contract_One year":
            1 if customer.contract == "One year" else 0,

        "Contract_Two year":
            1 if customer.contract == "Two year" else 0,

        "PaymentMethod_Bank transfer (automatic)":
            1 if customer.payment_method ==
            "Bank transfer (automatic)"
            else 0,

        "PaymentMethod_Credit card (automatic)":
            1 if customer.payment_method ==
            "Credit card (automatic)"
            else 0,

        "PaymentMethod_Electronic check":
            1 if customer.payment_method ==
            "Electronic check"
            else 0,

        "PaymentMethod_Mailed check":
            1 if customer.payment_method ==
            "Mailed check"
            else 0,
    }

    input_df = pd.DataFrame([data])

    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return input_df.astype(float)


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(
    probability: float
):

    if probability >= 0.60:
        return "High"

    elif probability >= 0.30:
        return "Medium"

    else:
        return "Low"


# ============================================================
# RETENTION RECOMMENDATION
# ============================================================

def get_recommendation(
    customer: CustomerData
):

    if customer.contract == "Month-to-month":

        return (
            "Offer a discounted one-year or two-year "
            "contract to improve customer retention."
        )

    elif customer.tenure <= 12:

        return (
            "Provide an early-life customer check-in "
            "and onboarding support."
        )

    elif customer.internet_service == "Fiber optic":

        return (
            "Review service quality and consider a "
            "loyalty or pricing offer."
        )

    elif customer.payment_method == "Electronic check":

        return (
            "Offer an incentive to switch to "
            "automatic payment."
        )

    elif customer.tech_support == 0:

        return (
            "Offer a complimentary technical "
            "support trial."
        )

    elif customer.online_security == 0:

        return (
            "Offer a free online security trial "
            "or bundle."
        )

    elif customer.online_backup == 0:

        return (
            "Offer an online backup bundle "
            "or promotional plan."
        )

    elif customer.device_protection == 0:

        return (
            "Offer a device protection bundle."
        )

    elif customer.monthly_charges >= 80:

        return (
            "Review the customer's plan and consider "
            "right-sizing or a loyalty discount."
        )

    else:

        return (
            "Route the customer to the retention team "
            "for personalized outreach."
        )


# ============================================================
# CHURN PREDICTION
# ============================================================

@app.post("/api/predict")
def predict_churn(
    customer: CustomerData
):

    input_df = prepare_customer_features(
        customer
    )

    if (
        ACTIVE_MODEL == "TabPFN"
        and "TabPFN" in models
    ):

        try:

            model = models["TabPFN"]

            probability = float(
                model.predict_proba(
                    input_df
                )[0][1]
            )

            model_used = "TabPFN"

        except Exception as e:

            print(
                "TabPFN prediction failed:",
                str(e)
            )

            print(
                "Using XGBoost fallback."
            )

            model = models["XGBoost"]

            probability = float(
                model.predict_proba(
                    input_df
                )[0][1]
            )

            model_used = "XGBoost (Fallback)"

    else:

        model = models["XGBoost"]

        probability = float(
            model.predict_proba(
                input_df
            )[0][1]
        )

        model_used = "XGBoost (Fallback)"

    risk_level = get_risk_level(
        probability
    )

    recommendation = get_recommendation(
        customer
    )

    return {

        "churn_probability":
            round(probability, 4),

        "churn_percentage":
            round(probability * 100, 2),

        "risk_level":
            risk_level,

        "model_used":
            model_used,

        "recommendation":
            recommendation
    }


# ============================================================
# SHAP HELPER FUNCTIONS
# ============================================================

def humanize_feature_name(feature_name: str):

    names = {

        "gender_Male":
            "Gender (Male)",

        "SeniorCitizen":
            "Senior Citizen",

        "Partner":
            "Partner",

        "Dependents":
            "Dependents",

        "tenure":
            "Customer Tenure",

        "PhoneService":
            "Phone Service",

        "MultipleLines":
            "Multiple Lines",

        "OnlineSecurity":
            "Online Security",

        "OnlineBackup":
            "Online Backup",

        "DeviceProtection":
            "Device Protection",

        "TechSupport":
            "Tech Support",

        "StreamingTV":
            "Streaming TV",

        "StreamingMovies":
            "Streaming Movies",

        "PaperlessBilling":
            "Paperless Billing",

        "MonthlyCharges":
            "Monthly Charges",

        "TotalCharges":
            "Total Charges",

        "InternetService_DSL":
            "Internet Service: DSL",

        "InternetService_Fiber optic":
            "Internet Service: Fiber optic",

        "InternetService_No":
            "Internet Service: None",

        "Contract_Month-to-month":
            "Contract: Month-to-month",

        "Contract_One year":
            "Contract: One year",

        "Contract_Two year":
            "Contract: Two year",

        "PaymentMethod_Bank transfer (automatic)":
            "Payment: Bank transfer (automatic)",

        "PaymentMethod_Credit card (automatic)":
            "Payment: Credit card (automatic)",

        "PaymentMethod_Electronic check":
            "Payment: Electronic check",

        "PaymentMethod_Mailed check":
            "Payment: Mailed check",
    }

    return names.get(
        feature_name,
        feature_name.replace("_", " ")
    )


def get_shap_background():

    global _shap_background

    if _shap_background is None:

        background_size = min(
            20,
            len(X_test)
        )

        _shap_background = X_test.sample(
            n=background_size,
            random_state=42
        )

    return _shap_background


def get_tabpfn_shap_explainer():

    global _shap_explainer

    if _shap_explainer is None:

        background = get_shap_background()

        print(
            "Creating SHAP KernelExplainer for TabPFN..."
        )

        def tabpfn_predict(data_array):

            data_frame = pd.DataFrame(
                data_array,
                columns=feature_columns
            )

            probabilities = tabpfn_model.predict_proba(
                data_frame
            )

            return probabilities[:, 1]

        _shap_explainer = shap.KernelExplainer(
            tabpfn_predict,
            background
        )

    return _shap_explainer


# ============================================================
# SHAP CUSTOMER EXPLANATION
# ============================================================

@app.post("/api/explain")
def explain_customer(
    customer: CustomerData
):

    input_df = prepare_customer_features(
        customer
    )

    # --------------------------------------------------------
    # First calculate the actual prediction
    # --------------------------------------------------------

    model_used = ACTIVE_MODEL

    if (
        ACTIVE_MODEL == "TabPFN"
        and "TabPFN" in models
    ):

        try:

            prediction_model = models["TabPFN"]

            probability = float(
                prediction_model.predict_proba(
                    input_df
                )[0][1]
            )

            model_used = "TabPFN"

        except Exception as e:

            print(
                "TabPFN explanation prediction failed:",
                str(e)
            )

            prediction_model = models["XGBoost"]

            probability = float(
                prediction_model.predict_proba(
                    input_df
                )[0][1]
            )

            model_used = "XGBoost (Fallback)"

    else:

        prediction_model = models["XGBoost"]

        probability = float(
            prediction_model.predict_proba(
                input_df
            )[0][1]
        )

        model_used = "XGBoost (Fallback)"

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    try:

        if (
            model_used == "TabPFN"
            and tabpfn_model is not None
        ):

            print(
                "Calculating TabPFN SHAP explanation..."
            )

            explainer = get_tabpfn_shap_explainer()

            shap_values = explainer.shap_values(
                input_df,
                nsamples=100
            )

            if isinstance(shap_values, list):

                shap_values = shap_values[-1]

            shap_values = np.asarray(
                shap_values
            )

            if shap_values.ndim == 3:

                shap_values = shap_values[0, :, -1]

            elif shap_values.ndim == 2:

                shap_values = shap_values[0]

        else:

            print(
                "Calculating XGBoost SHAP explanation..."
            )

            explainer = shap.TreeExplainer(
                models["XGBoost"]
            )

            shap_values = explainer.shap_values(
                input_df
            )

            if isinstance(shap_values, list):

                shap_values = shap_values[-1]

            shap_values = np.asarray(
                shap_values
            )

            if shap_values.ndim > 1:

                shap_values = shap_values[0]

        # ----------------------------------------------------
        # Build feature explanations
        # ----------------------------------------------------

        explanations = []

        for index, feature in enumerate(feature_columns):

            value = float(
                input_df.iloc[0][feature]
            )

            contribution = float(
                shap_values[index]
            )

            if contribution > 0:

                direction = "increases"

                impact = "Higher churn risk"

            elif contribution < 0:

                direction = "decreases"

                impact = "Lower churn risk"

            else:

                direction = "has little effect"

                impact = "Low impact"

            explanations.append({

                "feature":
                    feature,

                "name":
                    humanize_feature_name(
                        feature
                    ),

                "value":
                    value,

                "shap_value":
                    round(
                        contribution,
                        6
                    ),

                "absolute_impact":
                    round(
                        abs(contribution),
                        6
                    ),

                "direction":
                    direction,

                "impact":
                    impact
            })

        # ----------------------------------------------------
        # Sort by absolute SHAP impact
        # ----------------------------------------------------

        explanations.sort(
            key=lambda item:
            item["absolute_impact"],
            reverse=True
        )

        top_factors = explanations[:8]

        positive_factors = [
            item
            for item in explanations
            if item["shap_value"] > 0
        ][:5]

        negative_factors = [
            item
            for item in explanations
            if item["shap_value"] < 0
        ][:5]

        return {

            "success":
                True,

            "model_used":
                model_used,

            "churn_probability":
                round(
                    probability,
                    4
                ),

            "churn_percentage":
                round(
                    probability * 100,
                    2
                ),

            "risk_level":
                get_risk_level(
                    probability
                ),

            "top_factors":
                top_factors,

            "positive_factors":
                positive_factors,

            "negative_factors":
                negative_factors,

            "explanation_method":
                (
                    "SHAP KernelExplainer"
                    if model_used == "TabPFN"
                    else "SHAP TreeExplainer"
                ),

            "background_samples":
                20,

            "note":
                (
                    "SHAP values indicate how each feature "
                    "contributed to this individual prediction. "
                    "Positive values increase predicted churn "
                    "risk, while negative values decrease it."
                )
        }

    except Exception as e:

        print(
            "SHAP explanation failed:",
            str(e)
        )

        return {

            "success":
                False,

            "model_used":
                model_used,

            "churn_probability":
                round(
                    probability,
                    4
                ),

            "churn_percentage":
                round(
                    probability * 100,
                    2
                ),

            "risk_level":
                get_risk_level(
                    probability
                ),

            "top_factors":
                [],

            "positive_factors":
                [],

            "negative_factors":
                [],

            "explanation_method":
                "SHAP",

            "error":
                str(e)
        }


# ============================================================
# TEST PREDICTION
# ============================================================

@app.get("/api/test-prediction")
def test_prediction():

    model = models["XGBoost"]

    sample = {

        "gender": "Male",

        "senior_citizen": 0,

        "partner": 1,

        "dependents": 0,

        "tenure": 12,

        "contract": "Month-to-month",

        "paperless_billing": 1,

        "payment_method": "Electronic check",

        "monthly_charges": 80.0,

        "total_charges": 960.0,

        "phone_service": 1,

        "multiple_lines": 0,

        "internet_service": "Fiber optic",

        "online_security": 0,

        "online_backup": 0,

        "device_protection": 0,

        "tech_support": 0,

        "streaming_tv": 1,

        "streaming_movies": 1
    }

    customer = CustomerData(
        **sample
    )

    input_df = prepare_customer_features(
        customer
    )

    probability = float(
        model.predict_proba(
            input_df
        )[0][1]
    )

    return {

        "model": "XGBoost",

        "churn_probability":
            round(probability, 4),

        "churn_percentage":
            round(probability * 100, 2),

        "risk_level":
            get_risk_level(
                probability
            )
    }


# ============================================================
# REAL MODEL PERFORMANCE
# ============================================================

@app.get("/api/model-comparison")
def model_comparison():

    results = []

    for model_name, model in models.items():

        try:

            y_pred = model.predict(
                X_test
            )

            y_prob = model.predict_proba(
                X_test
            )[:, 1]

            tn, fp, fn, tp = confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1]
            ).ravel()

            results.append({

                "name": model_name,

                "role": (
                    "Primary"
                    if model_name == PRIMARY_MODEL
                    else (
                        "Backup"
                        if model_name == BACKUP_MODEL
                        else "Baseline"
                    )
                ),

                "status": "Available",

                "accuracy": round(
                    accuracy_score(
                        y_test,
                        y_pred
                    ) * 100,
                    2
                ),

                "precision": round(
                    precision_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    ) * 100,
                    2
                ),

                "recall": round(
                    recall_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    ) * 100,
                    2
                ),

                "f1_score": round(
                    f1_score(
                        y_test,
                        y_pred,
                        zero_division=0
                    ) * 100,
                    2
                ),

                "roc_auc": round(
                    roc_auc_score(
                        y_test,
                        y_prob
                    ) * 100,
                    2
                ),

                "pr_auc": round(
                    average_precision_score(
                        y_test,
                        y_prob
                    ) * 100,
                    2
                ),

                "confusion_matrix": {

                    "true_negative":
                        int(tn),

                    "false_positive":
                        int(fp),

                    "false_negative":
                        int(fn),

                    "true_positive":
                        int(tp)
                }
            })

        except Exception as e:

            print(
                f"Evaluation failed for {model_name}:",
                str(e)
            )

            results.append({

                "name": model_name,

                "role": (
                    "Primary"
                    if model_name == PRIMARY_MODEL
                    else (
                        "Backup"
                        if model_name == BACKUP_MODEL
                        else "Baseline"
                    )
                ),

                "status":
                    "Evaluation failed",

                "error":
                    str(e)
            })

    return {

        "models": results,

        "primary_model":
            PRIMARY_MODEL,

        "backup_model":
            BACKUP_MODEL,

        "active_model":
            ACTIVE_MODEL,

        "test_set_size":
            len(y_test)
    }


# ============================================================
# ANALYTICS
# ============================================================

@app.get("/api/analytics")
def analytics():

    contract_churn = (
        df.groupby("Contract")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .round(2)
        .to_dict()
    )

    internet_churn = (
        df.groupby("InternetService")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .round(2)
        .to_dict()
    )

    payment_churn = (
        df.groupby("PaymentMethod")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .round(2)
        .to_dict()
    )

    senior_churn = (
        df.groupby("SeniorCitizen")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .round(2)
        .to_dict()
    )

    return {

        "contract_churn":
            contract_churn,

        "internet_service_churn":
            internet_churn,

        "payment_method_churn":
            payment_churn,

        "senior_citizen_churn":
            senior_churn
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )