import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import shap
import joblib


from imblearn.over_sampling import SMOTE


from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


from sklearn.inspection import PartialDependenceDisplay

from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

import matplotlib
matplotlib.use("TkAgg")

warnings.simplefilter("ignore", category=FitFailedWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter("ignore")



os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)


df_eda = pd.read_csv("data/raw/Breast_Cancer.csv")
df_eda.dropna(inplace=True) 


df_eda["Status"] = df_eda["Status"].astype(str)


numeric_cols_eda = ["Age", "Tumor Size", "Survival Months", 
                    "Regional Node Examined", "Reginol Node Positive"]
plt.figure()
sns.pairplot(df_eda, vars=numeric_cols_eda, hue="Status", diag_kind="hist")
plt.savefig("plots/eda_pairplot.png")
plt.close()


plt.figure(figsize=(6, 4))
sns.countplot(data=df_eda, x="T Stage", hue="Status")
plt.title("T Stage by Status")
plt.savefig("plots/eda_tstage_status_countplot.png")
plt.close()


plt.figure(figsize=(6,5))
sns.violinplot(data=df_eda, x="Status", y="Age")
plt.title("Age distribution by Status")
plt.savefig("plots/eda_violin_age_status.png")
plt.close()


df = pd.read_csv("data/raw/Breast_Cancer.csv")
print("Data preview:")
print(df.head())

print("Dataset size:", df.shape)
print("Column information:")
print(df.info())
print("Unique column values:")
print(df.apply(lambda x: x.nunique()))
print("Summary statistics:")
print(df.describe())
print("Missing values:")
print(df.isnull().sum())




df.dropna(inplace=True)


df["Grade"] = df["Grade"].replace({" anaplastic; Grade IV": "4"}).astype(int)


categorical_columns = [
    "Race", "Marital Status", "T Stage", "N Stage", "6th Stage",
    "differentiate", "A Stage", "Estrogen Status", "Progesterone Status",
]
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


df["Status"] = df["Status"].replace({"Alive": 1, "Dead": 0})


df["NodeRatio"] = df["Reginol Node Positive"] / (df["Regional Node Examined"] + 1e-9)


numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


print("\nChecking class balance for Status:")
print(df["Status"].value_counts())
ratio_dead = df["Status"].value_counts(normalize=True)[0]
imbalance_threshold = 0.30

X = df.drop("Status", axis=1)
y = df["Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if ratio_dead < imbalance_threshold or ratio_dead > (1 - imbalance_threshold):
    print(f"\nApplying SMOTE... (Dead ratio={ratio_dead:.2f})")
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(pd.Series(y_train).value_counts())



def plot_confusion_matrix_save(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"plots/{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

def plot_roc_pr_curves_save(y_true, y_pred_prob, model_name):
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    plt.title(f"{model_name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision)
    plt.title(f"{model_name} Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.tight_layout()
    plt.savefig(f"plots/{model_name.lower().replace(' ', '_')}_roc_pr_curves.png")
    plt.close()

def evaluate_model_verbose(model, X_train, X_test, y_train, y_test, model_name, is_linear_reg=False):
    model.fit(X_train, y_train)

    if not is_linear_reg:
        y_pred = model.predict(X_test)
        y_pred_prob = (
            model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        )
    else:
        # 0.5 threshold
        y_reg_pred = model.predict(X_test)
        y_pred = (y_reg_pred >= 0.5).astype(int)
        y_pred_prob = np.clip(y_reg_pred, 0, 1)

    print(f"\n==== {model_name} RESULTS ====")
    print(classification_report(y_test, y_pred))

 
    plot_confusion_matrix_save(y_test, y_pred, model_name)


    if y_pred_prob is not None and len(np.unique(y_pred_prob)) > 1:
        plot_roc_pr_curves_save(y_test, y_pred_prob, model_name)

def evaluate_model_metrics(model, X_train, X_test, y_train, y_test, is_linear_reg=False):
    model.fit(X_train, y_train)
    if not is_linear_reg:
        y_pred = model.predict(X_test)
        y_pred_prob = (
            model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        )
    else:
        y_reg_pred = model.predict(X_test)
        y_pred = (y_reg_pred >= 0.5).astype(int)
        y_pred_prob = np.clip(y_reg_pred, 0, 1)

    acc = accuracy_score(y_test, y_pred)
    f1_m = f1_score(y_test, y_pred, average="macro")
    f1_w = f1_score(y_test, y_pred, average="weighted")
    p0 = precision_score(y_test, y_pred, pos_label=0)
    p1 = precision_score(y_test, y_pred, pos_label=1)
    r0 = recall_score(y_test, y_pred, pos_label=0)
    r1 = recall_score(y_test, y_pred, pos_label=1)
    rocauc = (roc_auc_score(y_test, y_pred_prob) 
              if y_pred_prob is not None and len(np.unique(y_pred_prob)) > 1 
              else np.nan)

    return {
        "Accuracy": acc,
        "F1 (macro)": f1_m,
        "F1 (weighted)": f1_w,
        "Precision_0 (Dead)": p0,
        "Precision_1 (Alive)": p1,
        "Recall_0 (Dead)": r0,
        "Recall_1 (Alive)": r1,
        "ROC_AUC": rocauc,
    }


lin_reg_default = LinearRegression()
evaluate_model_verbose(
    lin_reg_default, X_train, X_test, y_train, y_test, 
    "Linear Regression (Default)", 
    is_linear_reg=True
)

log_reg_default = LogisticRegression(random_state=42)
evaluate_model_verbose(
    log_reg_default, X_train, X_test, y_train, y_test, 
    "Logistic Regression (Default)"
)

rf_default = RandomForestClassifier(random_state=42)
evaluate_model_verbose(
    rf_default, X_train, X_test, y_train, y_test, 
    "Random Forest (Default)"
)

xgb_default = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
evaluate_model_verbose(
    xgb_default, X_train, X_test, y_train, y_test, 
    "XGBoost (Default)"
)

default_results = {}
default_results["LinearRegression_Default"] = evaluate_model_metrics(
    lin_reg_default, X_train, X_test, y_train, y_test, is_linear_reg=True
)
default_results["LogisticRegression_Default"] = evaluate_model_metrics(
    log_reg_default, X_train, X_test, y_train, y_test
)
default_results["RandomForest_Default"] = evaluate_model_metrics(
    rf_default, X_train, X_test, y_train, y_test
)
default_results["XGBoost_Default"] = evaluate_model_metrics(
    xgb_default, X_train, X_test, y_train, y_test
)


log_param_grid = {
    "penalty": ["l1", "l2"],
    "C": [0.01, 0.1, 1, 10],
    "solver": ["saga", "liblinear"],
}
logistic_base = LogisticRegression(max_iter=1000, random_state=42)
log_grid = GridSearchCV(
    logistic_base, 
    param_grid=log_param_grid, 
    scoring="f1", 
    cv=5,
    n_jobs=-1
)
log_grid.fit(X_train, y_train)
best_log_model = log_grid.best_estimator_
print("\nBest Logistic Regression params:", log_grid.best_params_)

evaluate_model_verbose(
    best_log_model, X_train, X_test, y_train, y_test, 
    "Logistic Regression (Tuned)"
)

rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["auto", "sqrt"],
}
rf_base = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(
    rf_base,
    param_grid=rf_param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)
best_rf_model = rf_grid.best_estimator_
print("\nBest Random Forest params:", rf_grid.best_params_)

evaluate_model_verbose(
    best_rf_model, X_train, X_test, y_train, y_test, 
    "Random Forest (Tuned)"
)

tuned_results = {}
tuned_results["LogisticRegression_Tuned"] = evaluate_model_metrics(
    best_log_model, X_train, X_test, y_train, y_test
)
tuned_results["RandomForest_Tuned"] = evaluate_model_metrics(
    best_rf_model, X_train, X_test, y_train, y_test
)


all_model_objects = {
    "LinearRegression_Default.pkl": lin_reg_default,
    "LogisticRegression_Default.pkl": log_reg_default,
    "RandomForest_Default.pkl": rf_default,
    "XGBoost_Default.pkl": xgb_default,
    "LogisticRegression_Tuned.pkl": best_log_model,
    "RandomForest_Tuned.pkl": best_rf_model
}

for model_filename, model_object in all_model_objects.items():
    save_path = os.path.join("models", model_filename)
    joblib.dump(model_object, save_path)
    print(f"Model saved to: {save_path}")


all_results = {**default_results, **tuned_results}
results_df = pd.DataFrame(all_results).T
print("\n================ COMPARISON TABLE ================")
print(results_df)




import os


os.makedirs("results", exist_ok=True)


results_df.to_csv("results/comparison_results.csv", index_label="Model")
print("Comparison results saved to 'results/comparison_results.csv'.")


plt.figure(figsize=(10, 6))
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(len(indices)), importances[indices], align="center")
feat_names_sorted = [X.columns[i] for i in indices]
plt.xticks(range(len(indices)), feat_names_sorted, rotation=90)
plt.title("Feature Importances (Random Forest - Tuned)")
plt.tight_layout()
plt.savefig("plots/random_forest_tuned_feature_importances.png")
plt.close()

explainer = shap.TreeExplainer(best_rf_model)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list) and len(shap_values) > 1:
    shap_values_to_plot = shap_values[1]  
else:
    shap_values_to_plot = shap_values

shap.summary_plot(
    shap_values_to_plot,
    X_test,
    feature_names=X.columns,
    show=False
)
plt.savefig("plots/shap_summary_plot_rf_tuned.png")
plt.close()

shap.summary_plot(
    shap_values_to_plot,
    X_test,
    feature_names=X.columns,
    plot_type="bar",
    show=False
)
plt.savefig("plots/shap_summary_bar_plot_rf_tuned.png")
plt.close()


pdp_features = ["Age", "Tumor Size", "NodeRatio"]
fig, ax = plt.subplots(nrows=1, ncols=len(pdp_features), figsize=(18, 5))

PartialDependenceDisplay.from_estimator(
    best_rf_model,
    X_train, 
    features=pdp_features,
    kind="average",  
    target=1,        
    ax=ax
)
plt.suptitle("Partial Dependence Plots (RandomForest - Tuned)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("plots/random_forest_tuned_pdp.png")
plt.close()
