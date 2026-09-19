# Machine-Learning-Intership
# Customer Churn Analysis and Prediction

Interactive Streamlit app for the SaiKet Systems ML Internship project. Analyzes customer churn in a telecom dataset and lets you build, train, and evaluate a prediction model — all through the browser, no code editing required.

## Files

| File | Purpose |
|---|---|
| `churn_app.py` | The Streamlit application |
| `Telco_Customer_Churn_Dataset.csv` | Dataset (7,043 customers, 21 columns) |

## Setup

Requires Python 3.8+.

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

## Run

From the folder containing both files:

```bash
streamlit run churn_prediction.py
```

This opens the app in your browser at `http://localhost:8501`. If it doesn't open automatically, go to that address manually.

## How to use it

The app is organized into 6 tabs, one per project task. Work through them in order — each one depends on the previous.

### 1. Data Preparation
Upload the CSV, or check the box to use the copy already in the folder. Click **Load and prepare data**. This:
- Drops `customerID` (identifier, not a predictor)
- Fixes `TotalCharges` — 11 rows store blank/whitespace text instead of a real missing value, which looks "clean" until you try to convert it to a number
- Fills missing numeric values with the median, missing categorical values with the mode
- Label-encodes all categorical columns and the `Churn` target (Yes/No → 1/0)

### 2. Train/Test Split
Pick a test set size (default 20%) and click **Split data**. The split is stratified, so the churn rate stays consistent between the training and testing sets.

### 3. Feature Selection
Choose one of three strategies, then adjust the multiselect if you want to add or remove specific columns:
- **Use all features** — everything except the target
- **Top by correlation** — ranks features by absolute correlation with churn, pick how many to keep
- **Domain-driven picks** — contract type, monthly/total charges, tenure, and other known churn drivers

### 4. Model Selection
Choose one of four algorithms: Logistic Regression, Decision Tree, Random Forest, or Gradient Boosting. A short note explains the tradeoff for each.

### 5. Model Training
Click **Train model**. Logistic Regression gets its inputs standardized first (it's scale-sensitive); the tree-based models don't need that.

### 6. Model Evaluation
Shows:
- Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrix and ROC curve
- Full classification report
- Feature importances (tree models) or coefficients (Logistic Regression)
- A download button for a text summary of the results

## Notes

- Changing the feature set or the train/test split resets any model you've already trained — you'll need to retrain after either change.
- Random Forest with all features typically lands around 80% accuracy and 0.83–0.84 ROC-AUC on this dataset, with no hyperparameter tuning. Recall on the churn class is the weaker metric — worth keeping in mind if the business priority is catching as many at-risk customers as possible rather than overall accuracy.
