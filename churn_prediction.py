"""
Customer Churn Analysis and Prediction — Streamlit App
SaiKet Systems - ML Internship Project

Covers all 6 project tasks as interactive tabs:
  1. Data Preparation
  2. Train/Test Split
  3. Feature Selection
  4. Model Selection
  5. Model Training
  6. Model Evaluation

Run with:  streamlit run churn_prediction.py
Keep Telco_Customer_Churn_Dataset.csv in the same folder, or upload it in the app.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

TARGET_COL = "Churn"
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

st.set_page_config(page_title="Churn Analysis & Prediction", layout="wide")

# ----------------------------------------------------------------------
# Session state init
# ----------------------------------------------------------------------
defaults = {
    "raw_df": None, "df": None, "label_encoders": {}, "feature_cols": None,
    "X_train": None, "X_test": None, "y_train": None, "y_test": None,
    "scaler": None, "model": None, "model_name": None, "trained": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_downstream_of_features():
    st.session_state.X_train = st.session_state.X_test = None
    st.session_state.y_train = st.session_state.y_test = None
    st.session_state.model = None
    st.session_state.model_name = None
    st.session_state.trained = False


def reset_downstream_of_split():
    st.session_state.model = None
    st.session_state.model_name = None
    st.session_state.trained = False


# ----------------------------------------------------------------------
# Sidebar: status
# ----------------------------------------------------------------------
st.sidebar.title("Pipeline status")
st.sidebar.write("Data prepared:", "✅" if st.session_state.df is not None else "❌")
st.sidebar.write("Features selected:", len(st.session_state.feature_cols) if st.session_state.feature_cols else "not set")
st.sidebar.write("Split done:", "✅" if st.session_state.X_train is not None else "❌")
st.sidebar.write("Model chosen:", st.session_state.model_name or "none")
st.sidebar.write("Model trained:", "✅" if st.session_state.trained else "❌")

st.title("📉 Customer Churn Analysis and Prediction")
st.caption("SaiKet Systems ML Internship — interactive pipeline")

tabs = st.tabs([
    "1️⃣ Data Preparation", "2️⃣ Train/Test Split", "3️⃣ Feature Selection",
    "4️⃣ Model Selection", "5️⃣ Model Training", "6️⃣ Model Evaluation",
])

# ----------------------------------------------------------------------
# Tab 1: Data Preparation
# ----------------------------------------------------------------------
with tabs[0]:
    st.header("Task 1: Data Preparation")
    st.write("Load the dataset, handle missing values, and encode categorical variables.")

    uploaded = st.file_uploader("Upload Telco_Customer_Churn_Dataset.csv", type="csv")
    use_default = st.checkbox("Use file already in this folder instead (Telco_Customer_Churn_Dataset.csv)")

    source = None
    if uploaded is not None:
        source = uploaded
    elif use_default:
        source = "Telco_Customer_Churn_Dataset.csv"

    if st.button("Load and prepare data", type="primary"):
        if source is None:
            st.error("Upload a file or check the box to use the local CSV.")
        else:
            try:
                raw_df = pd.read_csv(source)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                raw_df = None

            if raw_df is not None:
                st.session_state.raw_df = raw_df.copy()
                df = raw_df.copy()

                if "customerID" in df.columns:
                    df = df.drop(columns=["customerID"])

                # TotalCharges: blank/whitespace strings hidden as non-null text
                if "TotalCharges" in df.columns and df["TotalCharges"].dtype == object:
                    df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)
                    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

                missing = df.isnull().sum()
                missing = missing[missing > 0]
                if len(missing) > 0:
                    st.write("Missing values found and imputed:")
                    fill_log = []
                    for col in df.columns:
                        if df[col].isnull().sum() == 0:
                            continue
                        if pd.api.types.is_numeric_dtype(df[col]):
                            fv = df[col].median()
                            df[col] = df[col].fillna(fv)
                            fill_log.append(f"- `{col}`: filled with median = {fv:.2f}")
                        else:
                            fv = df[col].mode()[0]
                            df[col] = df[col].fillna(fv)
                            fill_log.append(f"- `{col}`: filled with mode = '{fv}'")
                    st.markdown("\n".join(fill_log))
                else:
                    st.success("No missing values detected after cleaning TotalCharges.")

                if df[TARGET_COL].dtype == object:
                    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

                cat_cols = df.select_dtypes(include="object").columns.tolist()
                encoders = {}
                for col in cat_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    encoders[col] = le

                st.session_state.df = df
                st.session_state.label_encoders = encoders
                st.session_state.feature_cols = [c for c in df.columns if c != TARGET_COL]
                reset_downstream_of_features()

                st.success(f"Prepared dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
                st.write(f"Encoded categorical columns: {cat_cols}")
                st.dataframe(df.head(10))

    if st.session_state.df is not None:
        st.subheader("Churn distribution")
        counts = st.session_state.df[TARGET_COL].value_counts().rename({0: "No", 1: "Yes"})
        st.bar_chart(counts)

# ----------------------------------------------------------------------
# Tab 2: Train/Test Split
# ----------------------------------------------------------------------
with tabs[1]:
    st.header("Task 2: Split Data for Training and Testing")
    if st.session_state.df is None:
        st.warning("Complete Task 1 first.")
    else:
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
        if st.button("Split data", type="primary"):
            feature_cols = st.session_state.feature_cols
            X = st.session_state.df[feature_cols]
            y = st.session_state.df[TARGET_COL]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            st.session_state.X_train, st.session_state.X_test = X_train, X_test
            st.session_state.y_train, st.session_state.y_test = y_train, y_test
            reset_downstream_of_split()
            st.success(f"Training rows: {X_train.shape[0]} | Testing rows: {X_test.shape[0]}")
            c1, c2 = st.columns(2)
            c1.metric("Train churn rate", f"{y_train.mean():.1%}")
            c2.metric("Test churn rate", f"{y_test.mean():.1%}")

# ----------------------------------------------------------------------
# Tab 3: Feature Selection
# ----------------------------------------------------------------------
with tabs[2]:
    st.header("Task 3: Feature Selection")
    if st.session_state.df is None:
        st.warning("Complete Task 1 first.")
    else:
        all_features = [c for c in st.session_state.df.columns if c != TARGET_COL]
        mode = st.radio(
            "Selection strategy",
            ["Use all features", "Top features by correlation with Churn", "Domain-driven picks"],
        )

        if mode == "Top features by correlation with Churn":
            corr = st.session_state.df[all_features + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
            corr = corr.abs().sort_values(ascending=False)
            st.write("Absolute correlation with Churn:")
            st.dataframe(corr)
            top_n = st.slider("Number of top features", 3, len(corr), min(10, len(corr)))
            candidate_cols = corr.head(top_n).index.tolist()
        elif mode == "Domain-driven picks":
            candidates = [
                "Contract", "MonthlyCharges", "tenure", "TotalCharges",
                "InternetService", "TechSupport", "OnlineSecurity",
                "PaymentMethod", "PaperlessBilling", "SeniorCitizen",
            ]
            candidate_cols = [c for c in candidates if c in all_features]
        else:
            candidate_cols = all_features

        selected = st.multiselect("Final feature set (edit if needed)", all_features, default=candidate_cols)

        if st.button("Apply feature selection", type="primary"):
            if len(selected) == 0:
                st.error("Select at least one feature.")
            else:
                st.session_state.feature_cols = selected
                reset_downstream_of_features()
                st.success(f"Active feature set ({len(selected)}): {selected}")

# ----------------------------------------------------------------------
# Tab 4: Model Selection
# ----------------------------------------------------------------------
with tabs[3]:
    st.header("Task 4: Model Selection")
    model_choice = st.selectbox("Choose a binary classification algorithm", list(MODELS.keys()))
    st.caption({
        "Logistic Regression": "Simple, interpretable, good baseline. Assumes a roughly linear relationship.",
        "Decision Tree": "Easy to interpret, captures non-linear splits, prone to overfitting alone.",
        "Random Forest": "Ensemble of trees, generally strong out-of-the-box performance, less interpretable.",
        "Gradient Boosting": "Sequential boosting, often the highest accuracy, slower to train, more tuning-sensitive.",
    }[model_choice])

    if st.button("Select this model", type="primary"):
        st.session_state.model = MODELS[model_choice]
        st.session_state.model_name = model_choice
        st.session_state.trained = False
        st.success(f"Selected model: {model_choice}")

# ----------------------------------------------------------------------
# Tab 5: Model Training
# ----------------------------------------------------------------------
with tabs[4]:
    st.header("Task 5: Model Training")
    if st.session_state.model is None:
        st.warning("Complete Task 4 first.")
    elif st.session_state.X_train is None:
        st.warning("Complete Task 2 first.")
    else:
        st.write(f"Ready to train **{st.session_state.model_name}** on "
                  f"{st.session_state.X_train.shape[0]} rows using "
                  f"{st.session_state.X_train.shape[1]} features.")
        if st.button("Train model", type="primary"):
            if st.session_state.model_name == "Logistic Regression":
                scaler = StandardScaler()
                X_fit = scaler.fit_transform(st.session_state.X_train)
                st.session_state.scaler = scaler
            else:
                X_fit = st.session_state.X_train
                st.session_state.scaler = None

            with st.spinner("Training..."):
                st.session_state.model.fit(X_fit, st.session_state.y_train)
            st.session_state.trained = True
            st.success("Training complete. Go to Task 6 to evaluate.")

# ----------------------------------------------------------------------
# Tab 6: Model Evaluation
# ----------------------------------------------------------------------
with tabs[5]:
    st.header("Task 6: Model Evaluation")
    if not st.session_state.trained:
        st.warning("Complete Task 5 first.")
    else:
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        X_test_fit = st.session_state.scaler.transform(X_test) if st.session_state.scaler is not None else X_test

        y_pred = model.predict(X_test_fit)
        y_proba = model.predict_proba(X_test_fit)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1-score", f"{f1:.3f}")
        c5.metric("ROC-AUC", f"{auc:.3f}")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Confusion matrix")
            fig, ax = plt.subplots(figsize=(4, 3.5))
            ax.imshow(cm, cmap="Blues")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["No", "Yes"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["No", "Yes"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            st.pyplot(fig)

        with col_b:
            st.subheader("ROC curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig2, ax2 = plt.subplots(figsize=(4, 3.5))
            ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.legend()
            st.pyplot(fig2)

        st.subheader("Classification report")
        report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"],
                                        zero_division=0, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Feature importance")
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=st.session_state.feature_cols)
            imp = imp.sort_values(ascending=False).head(10)
            st.bar_chart(imp)
        elif hasattr(model, "coef_"):
            coef = pd.Series(model.coef_[0], index=st.session_state.feature_cols)
            coef = coef.sort_values(key=np.abs, ascending=False).head(10)
            st.bar_chart(coef)

        st.download_button(
            "Download evaluation summary (.txt)",
            data=(
                f"Model: {st.session_state.model_name}\n"
                f"Features: {st.session_state.feature_cols}\n"
                f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\n"
                f"F1-score: {f1:.4f}\nROC-AUC: {auc:.4f}\n\n"
                f"{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'], zero_division=0)}"
            ),
            file_name="evaluation_summary.txt",
        )
