# Loan Default Prediction with XGBoost 🏦📉

A machine learning project using the [Home Credit Default Risk dataset](https://www.kaggle.com/competitions/home-credit-default-risk) to predict whether a loan applicant is likely to default. Built with `XGBoost`, feature engineering from multiple sources, and model evaluation using ROC AUC.

---

## 🔍 Objective

Predict loan default (`TARGET = 1`) using structured customer data, including credit bureau history, previous loans, and credit card behavior.

---

## 📂 Dataset Sources (from Kaggle)

- `application_train.csv` — main training set with target variable
- `bureau.csv` & `bureau_balance.csv` — external credit bureau history
- `previous_application.csv` — previous loan applications
- `POS_CASH_balance.csv` — point-of-sale loan performance
- `credit_card_balance.csv` — credit card balance behavior
- `installments_payments.csv` — payment history

---

## 🧠 Techniques Used

- **Data cleaning**: Missing values, infinities, merging multi-table relational data
- **Feature engineering**: Aggregates (mean, max, size), ratios (credit/income), behavioral indicators
- **Modeling**: XGBoost Classifier
- **Evaluation**: ROC AUC score of **0.70**
- **Tools**: `pandas`, `scikit-learn`, `xgboost`, `matplotlib`

---

## 📈 Model Performance

- ROC AUC: **0.70**
- Evaluated on stratified train/test split

---

## 🛠️ How to Run

1. Clone the repo
2. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)
3. Place CSVs in a `data/` folder
4. Run the notebook:

```bash
jupyter notebook notebooks/loan_default_project.ipynb
