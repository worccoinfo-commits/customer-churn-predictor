import pandas as pd
import numpy as np
import joblib
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def load_model():
    model_path = MODELS_DIR / "churn_model.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    features_path = MODELS_DIR / "feature_names.csv"

    for p in [model_path, scaler_path, features_path]:
        if not p.exists():
            print(f"Error: {p} not found. Run 'python src/train.py' first.")
            sys.exit(1)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = pd.read_csv(features_path, header=None)[0].tolist()

    return model, scaler, feature_names


def preprocess_single(customer: dict, feature_names: list) -> pd.DataFrame:
    df = pd.DataFrame([customer])

    df["Churn"] = 0
    df["TotalCharges"] = pd.to_numeric(df.get("TotalCharges", 0), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df["Gender"] = (df.get("gender", "Male").iloc[0] == "Male").astype(int)
    df = df.drop("gender", axis=1, errors="ignore")

    for suffix in ["MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        col = f"{suffix}_Yes"
        value = customer.get(suffix.lower(), "No")
        df[col] = 1 if value == "Yes" else 0
        if suffix in df.columns:
            df = df.drop(suffix, axis=1)

    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        val = customer.get(col.lower(), "No")
        df[col] = 1 if val == "Yes" else 0

    cat_map = {
        "InternetService": {"DSL": [1, 0], "Fiber optic": [0, 1], "No": [0, 0]},
        "Contract": {"Month-to-month": [1, 0], "One year": [0, 1], "Two year": [0, 0]},
        "PaymentMethod": {
            "Electronic check": [1, 0, 0],
            "Mailed check": [0, 1, 0],
            "Bank transfer (automatic)": [0, 0, 1],
            "Credit card (automatic)": [0, 0, 0]
        }
    }

    cols_lookup = {
        "InternetService": ["InternetService_DSL", "InternetService_Fiber optic"],
        "Contract": ["Contract_Month-to-month", "Contract_One year"],
        "PaymentMethod": [
            "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed check",
            "PaymentMethod_Bank transfer (automatic)"
        ]
    }

    for cat, cats_map in cat_map.items():
        val = customer.get(cat.lower(), list(cats_map.keys())[0])
        mapping = cats_map.get(val, cats_map[list(cats_map.keys())[0]])
        for j, col_name in enumerate(cols_lookup[cat]):
            df[col_name] = mapping[j]

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    for col in numeric_cols:
        if col in df.columns:
            df[col] = scaler.transform(df[[col]])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[[c for c in feature_names if c in df.columns]]
    df = df.reindex(columns=feature_names, fill_value=0)

    return df


def predict_single(customer: dict):
    model, _, feature_names = load_model()
    X = preprocess_single(customer, feature_names)
    proba = model.predict_proba(X)[0, 1]
    pred = model.predict(X)[0]

    risk = "HIGH" if proba >= 0.7 else ("MEDIUM" if proba >= 0.3 else "LOW")

    recommendations = []
    if risk == "HIGH":
        recommendations = [
            "Immediate retention offer recommended",
            "Assign dedicated account manager",
            "Offer discounted annual contract",
            "Proactive customer satisfaction call"
        ]
    elif risk == "MEDIUM":
        recommendations = [
            "Send targeted retention email campaign",
            "Offer loyalty rewards or discounts",
            "Check support ticket history for unresolved issues"
        ]
    else:
        recommendations = [
            "No immediate action required",
            "Enroll in standard loyalty program",
            "Continue regular engagement"
        ]

    return {
        "churn_probability": round(float(proba), 4),
        "prediction": "Churn" if pred == 1 else "Stay",
        "risk_level": risk,
        "recommendations": recommendations
    }


def predict_csv(csv_path: str):
    model, _, feature_names = load_model()
    df_input = pd.read_csv(csv_path)

    results = []
    for idx, row in df_input.iterrows():
        customer = row.to_dict()
        result = predict_single(customer)
        result["customer_id"] = idx
        results.append(result)
        print(f"  Customer {idx}: {result['prediction']} ({result['risk_level']} risk, probability: {result['churn_probability']:.2%})")

    return results


def interactive_mode():
    print("\n" + "="*50)
    print("  CUSTOMER CHURN PREDICTOR")
    print("  Interactive Mode")
    print("="*50)

    customer = {}
    fields = [
        ("gender", "Gender (Male/Female)", "Male"),
        ("seniorcitizen", "Senior Citizen? (0/1)", 0),
        ("partner", "Has Partner? (Yes/No)", "Yes"),
        ("dependents", "Has Dependents? (Yes/No)", "No"),
        ("tenure", "Tenure (months)", 1),
        ("phoneservice", "Phone Service? (Yes/No)", "Yes"),
        ("multiplelines", "Multiple Lines? (Yes/No/No phone service)", "No"),
        ("internetservice", "Internet Service? (DSL/Fiber optic/No)", "Fiber optic"),
        ("onlinesecurity", "Online Security? (Yes/No/No internet service)", "No"),
        ("onlinebackup", "Online Backup? (Yes/No/No internet service)", "No"),
        ("deviceprotection", "Device Protection? (Yes/No/No internet service)", "No"),
        ("techsupport", "Tech Support? (Yes/No/No internet service)", "No"),
        ("streamingtv", "Streaming TV? (Yes/No/No internet service)", "No"),
        ("streamingmovies", "Streaming Movies? (Yes/No/No internet service)", "No"),
        ("contract", "Contract? (Month-to-month/One year/Two year)", "Month-to-month"),
        ("paperlessbilling", "Paperless Billing? (Yes/No)", "Yes"),
        ("paymentmethod", "Payment Method? (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic))", "Electronic check"),
        ("monthlycharges", "Monthly Charges ($)", 65),
        ("totalcharges", "Total Charges ($)", 500),
    ]

    try:
        for key, prompt, default in fields:
            val = input(f"  {prompt} [{default}]: ").strip()
            if not val:
                val = default
            if isinstance(default, int) and key not in ["partner", "dependents", "phoneservice", "paperlessbilling"]:
                try:
                    val = int(val)
                except ValueError:
                    val = default
            elif isinstance(default, float):
                try:
                    val = float(val)
                except ValueError:
                    val = default
            customer[key.lower()] = val

        result = predict_single(customer)

        print("\n" + "="*50)
        print("  PREDICTION RESULT")
        print("="*50)
        print(f"  Churn Probability: {result['churn_probability']:.2%}")
        print(f"  Prediction:        {result['prediction']}")
        print(f"  Risk Level:        {result['risk_level']}")
        print("\n  Recommendations:")
        for rec in result['recommendations']:
            print(f"    • {rec}")
        print("="*50 + "\n")

    except KeyboardInterrupt:
        print("\n\n  Exiting...")
        sys.exit(0)


def main():
    if len(sys.argv) < 2:
        print("\nCustomer Churn Predictor - Prediction Tool")
        print("="*50)
        print("Usage:")
        print("  python src/predict.py               Interactive mode")
        print("  python src/predict.py --csv <file>   Predict from CSV")
        print("  python src/predict.py --json <json>  Predict from JSON string")
        print()
        interactive_mode()
        return

    if sys.argv[1] == "--csv" and len(sys.argv) > 2:
        print(f"\nPredicting from CSV: {sys.argv[2]}")
        results = predict_csv(sys.argv[2])
        print(f"\nProcessed {len(results)} customers")
        output_path = Path(sys.argv[2]).stem + "_predictions.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    elif sys.argv[1] == "--json" and len(sys.argv) > 2:
        customer = json.loads(sys.argv[2])
        result = predict_single(customer)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
