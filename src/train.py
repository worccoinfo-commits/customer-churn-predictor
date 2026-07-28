import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

DATA_URL = ""  # Uses kagglehub download
DATA_PATH = DATA_DIR / "telco_churn.csv"
RANDOM_STATE = 42


def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded from cached file: {DATA_PATH}")
        return df

    print(f"Downloading dataset from Kaggle...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        import glob
        csv_files = glob.glob(str(path) + "/*.csv")
        if csv_files:
            df = pd.read_csv(csv_files[0])
            df.to_csv(DATA_PATH, index=False)
            print(f"Saved to {DATA_PATH}")
            return df
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Falling back to direct URL...")

    df = pd.read_csv(DATA_URL)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved to {DATA_PATH}")
    return df


def quick_eda(df):
    print(f"\n{'='*60}")
    print("EXPLORATORY DATA ANALYSIS")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"\nChurn Distribution:")
    print(df["Churn"].value_counts())
    print(f"\nChurn Rate: {df['Churn'].value_counts(normalize=True)['Yes']*100:.2f}%")

    print(f"\nData Types:")
    print(df.dtypes.value_counts())

    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("None")

    print(f"\nNumeric Features Summary:")
    print(df.describe())

    print(f"\nCategorical Features:")
    cat_cols = df.select_dtypes(include=["object"]).columns.drop("customerID", errors="ignore")
    for col in cat_cols:
        print(f"  {col}: {df[col].nunique()} unique values - {df[col].unique()[:5]}")

    churn_by_contract = df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack()
    print(f"\nChurn Rate by Contract Type:")
    print(churn_by_contract)

    return df


def generate_plots(df):
    print(f"\nGenerating visualizations...")

    plt.style.use("seaborn-v0_8-darkgrid")
    colors = ["#00e676", "#f44336"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Customer Churn Analysis - Key Insights", fontsize=16, fontweight="bold", y=1.02)

    df["Churn"].value_counts().plot(
        kind="bar", ax=axes[0, 0], color=colors, edgecolor="white",
        title="Churn Distribution"
    )
    axes[0, 0].set_xticklabels(["Stayed", "Churned"], rotation=0)
    for i, v in enumerate(df["Churn"].value_counts()):
        axes[0, 0].text(i, v + 30, f"{v}", ha="center", fontweight="bold")

    tenure_churn = df.groupby("Churn")["tenure"]
    bp = axes[0, 1].boxplot(
        [tenure_churn.get_group("No"), tenure_churn.get_group("Yes")],
        patch_artist=True, medianprops=dict(color="black")
    )
    axes[0, 1].set_xticklabels(["Stayed", "Churned"])
    for patch in bp["boxes"]:
        patch.set_facecolor("#00e676")
        patch.set_alpha(0.6)
    axes[0, 1].set_title("Tenure Distribution by Churn")
    axes[0, 1].set_ylabel("Months")

    contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index")
    contract_churn.plot(kind="bar", ax=axes[0, 2], color=colors, edgecolor="white", legend=False)
    axes[0, 2].set_title("Churn Rate by Contract")
    axes[0, 2].set_ylabel("Rate")
    axes[0, 2].set_xlabel("")

    monthly_by_churn = df.groupby("Churn")["MonthlyCharges"].mean()
    axes[1, 0].bar(["Stayed", "Churned"], monthly_by_churn.values, color=colors, edgecolor="white")
    axes[1, 0].set_title("Avg Monthly Charges by Churn")
    axes[1, 0].set_ylabel("$")
    for i, v in enumerate(monthly_by_churn.values):
        axes[1, 0].text(i, v + 0.5, f"${v:.0f}", ha="center", fontweight="bold")

    payment_churn = pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index")
    payment_churn.plot(kind="bar", ax=axes[1, 1], color=colors, edgecolor="white", legend=False)
    axes[1, 1].set_title("Churn Rate by Payment Method")
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=30, ha="right")
    axes[1, 1].set_ylabel("Rate")

    internet_product_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    avg_churn_by_feature = {}
    for col in internet_product_cols:
        rates = df.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean())
        if "No" in rates.index:
            avg_churn_by_feature[col] = rates.get("No", 0)
    if avg_churn_by_feature:
        feat_df = pd.Series(avg_churn_by_feature).sort_values(ascending=True)
        feat_df.plot(kind="barh", ax=axes[1, 2], color="#00e676", edgecolor="white")
        axes[1, 2].set_title("Churn Rate (Customers Without Service)")
        axes[1, 2].set_xlabel("Churn Rate")

    for ax in axes.flat:
        ax.set_facecolor("#f8f9fa")

    plt.tight_layout()
    plot_path = REPORTS_DIR / "churn_analysis_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {plot_path}")


def preprocess_data(df):
    print(f"\n{'='*60}")
    print("DATA PREPROCESSING")
    print(f"{'='*60}")

    df = df.drop("customerID", axis=1)

    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df[col] = (df[col] == "Yes").astype(int)

    cat_cols = ["InternetService", "Contract", "PaymentMethod"]

    df["Gender"] = (df["gender"] == "Male").astype(int)
    df = df.drop("gender", axis=1)

    for col in ["MultipleLines", "OnlineSecurity", "OnlineBackup",
                 "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        df[f"{col}_Yes"] = (df[col] == "Yes").astype(int)
        df = df.drop(col, axis=1)

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    df = df.astype({c: "int64" for c in df.select_dtypes(include=["object"]).columns if c != "Churn"})

    print(f"Final feature count: {df.shape[1] - 1}")
    print(f"Features: {[c for c in df.columns if c != 'Churn']}")

    return df


def train_models(X_train, X_test, y_train, y_test):
    print(f"\n{'='*60}")
    print("MODEL TRAINING")
    print(f"{'='*60}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=RANDOM_STATE, eval_metric="auc", use_label_encoder=False
        )
    }

    results = {}
    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "CV ROC-AUC (mean)": round(cv_scores.mean(), 4),
            "CV ROC-AUC (std)": round(cv_scores.std(), 4),
        }

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        if auc > best_score:
            best_score = auc
            best_model = model
            best_name = name

    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name} (ROC-AUC: {best_score:.4f})")
    print(f"{'='*60}")

    return results, best_model, best_name


def plot_model_comparison(results):
    metrics_df = pd.DataFrame(results).T
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    multiplier = 0

    for model_name in metrics_df.index:
        offset = width * multiplier
        values = [metrics_df.loc[model_name, m] for m in metrics_to_plot]
        bars = ax.bar(x + offset, values, width, label=model_name, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        multiplier += 1

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Performance Comparison", fontweight="bold")
    ax.legend(loc="lower right")
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5)
    ax.set_facecolor("#f8f9fa")

    plot_path = REPORTS_DIR / "model_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Model comparison saved to {plot_path}")


def plot_feature_importance(model, feature_names, model_name, top_n=15):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(top_n), importances[indices], color="#00e676", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Features - {model_name}", fontweight="bold")
    ax.set_facecolor("#f8f9fa")

    plot_path = REPORTS_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance saved to {plot_path}")


def plot_roc_curves(models_trained, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models_trained:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves Comparison", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_facecolor("#f8f9fa")

    plot_path = REPORTS_DIR / "roc_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curves saved to {plot_path}")


def generate_business_report(results, feature_names, model):
    print(f"\n{'='*60}")
    print("BUSINESS IMPACT ANALYSIS")
    print(f"{'='*60}")

    avg_revenue_per_customer = 65  
    retention_cost = 20  
    total_customers = 7043

    churn_rate = results["XGBoost"]["Recall"]
    false_positive_rate = 1 - results["XGBoost"]["Precision"]

    prevented_churn = int(total_customers * churn_rate * 0.6)
    saved_revenue = prevented_churn * avg_revenue_per_customer * 12
    program_cost = total_customers * retention_cost
    net_impact = saved_revenue - program_cost
    roi = (net_impact / program_cost) * 100

    report = f"""
{'='*60}
CUSTOMER CHURN PREDICTOR - BUSINESS IMPACT REPORT
{'='*60}

ASSUMPTIONS:
  - Average monthly revenue per customer: ${avg_revenue_per_customer}
  - Retention program cost per customer: ${retention_cost}
  - 60% of predicted churners can be retained with intervention
  - Total customer base: {total_customers:,}

MODEL PERFORMANCE (Best: {list(results.keys())[-1] if results else "XGBoost"}):
  - Accuracy: {results.get(list(results.keys())[-1], {}).get('Accuracy', 0)*100:.2f}%
  - Precision: {results.get(list(results.keys())[-1], {}).get('Precision', 0)*100:.2f}%
  - Recall: {results.get(list(results.keys())[-1], {}).get('Recall', 0)*100:.2f}%
  - ROC-AUC: {results.get(list(results.keys())[-1], {}).get('ROC-AUC', 0):.4f}

BUSINESS IMPACT:
  - Customers identified as high-risk: {int(total_customers * churn_rate):,}
  - Estimated preventable churns: {prevented_churn:,}
  - Annual revenue saved: ${saved_revenue:,.0f}
  - Retention program cost: ${program_cost:,.0f}
  - Net annual impact: ${net_impact:,.0f}
  - ROI: {roi:.1f}%

TOP RISK FACTORS TO MONITOR:
"""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-5:]
        for idx in reversed(indices):
            report += f"  - {feature_names[idx].replace('_', ' ')} (importance: {importances[idx]:.3f})\n"

    report += f"""
RECOMMENDATIONS:
  1. Implement early warning system for customers reaching 12-month tenure
  2. Offer loyalty discounts for month-to-month contract customers
  3. Bundle online security and tech support for higher retention
  4. Target customers with high monthly charges (>$70) with retention offers
  5. Improve onboarding for first 6 months - highest churn period
{'='*60}
"""
    print(report)

    report_path = REPORTS_DIR / "business_impact_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Business report saved to {report_path}")

    report_data = {
        "churn_rate": float(churn_rate),
        "precision": float(results.get(list(results.keys())[-1], {}).get('Precision', 0)),
        "recall": float(results.get(list(results.keys())[-1], {}).get('Recall', 0)),
        "roc_auc": float(results.get(list(results.keys())[-1], {}).get('ROC-AUC', 0)),
        "preventable_churns": prevented_churn,
        "annual_savings": float(saved_revenue),
        "roi_percent": float(roi),
    }
    with open(REPORTS_DIR / "business_impact.json", "w") as f:
        json.dump(report_data, f, indent=2)

    return report


def main():
    print(f"\n{'#'*60}")
    print("#  CUSTOMER CHURN PREDICTOR - TRAINING PIPELINE")
    print(f"{'#'*60}")

    df = load_data()
    df = quick_eda(df)

    generate_plots(df)

    df_processed = preprocess_data(df)
    X = df_processed.drop("Churn", axis=1)
    y = df_processed["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        X_train[col] = scaler.fit_transform(X_train[[col]])
        X_test[col] = scaler.transform(X_test[[col]])

    results, best_model, best_name = train_models(X_train, X_test, y_train, y_test)

    plot_model_comparison(results)
    plot_feature_importance(best_model, X.columns, best_name)

    models_trained = []
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced")
        if name == "Random Forest":
            model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
        elif name == "XGBoost":
            model = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=RANDOM_STATE, eval_metric="auc")
        model.fit(X_train, y_train)
        models_trained.append((name, model))

    plot_roc_curves(models_trained, X_test, y_test)

    results_df = pd.DataFrame(results).T
    results_df.to_csv(REPORTS_DIR / "model_results.csv")
    print(f"\nResults saved to {REPORTS_DIR / 'model_results.csv'}")

    joblib.dump(best_model, MODELS_DIR / "churn_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    X.columns.to_series().to_csv(MODELS_DIR / "feature_names.csv", index=False)
    print(f"Best model saved to {MODELS_DIR / 'churn_model.joblib'}")

    report = generate_business_report(results, X.columns, best_model)

    print(f"\n{'#'*60}")
    print("#  TRAINING COMPLETE")
    print(f"{'#'*60}")
    print(f"\nSaved artifacts:")
    print(f"  - Model:       models/churn_model.joblib")
    print(f"  - Scaler:      models/scaler.joblib")
    print(f"  - Features:    models/feature_names.csv")
    print(f"  - Plots:       reports/*.png")
    print(f"  - Results:     reports/model_results.csv")
    print(f"  - Report:      reports/business_impact_report.txt")


if __name__ == "__main__":
    main()
