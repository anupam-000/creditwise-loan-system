# 💳 CreditWise – Loan Approval Prediction System

CreditWise is a **Machine Learning project** that predicts whether a loan application will be **Approved or Rejected** using an applicant's financial, personal, employment, and loan-related information.

## 📌 Overview

The project demonstrates a complete Machine Learning workflow, from data preprocessing and EDA to model training and evaluation.

### 🔄 Workflow

**Data → Preprocessing → EDA → Encoding → Scaling → Feature Engineering → Model Training → Evaluation**

## 📊 Dataset

The dataset contains **1,000 loan applications** with features such as:

* Applicant & Coapplicant Income
* Age & Dependents
* Credit Score
* Existing Loans
* DTI Ratio
* Savings & Collateral Value
* Loan Amount & Term
* Employment & Education
* Loan Purpose & Property Area

**Target:** `Loan_Approved` — Approved / Rejected

## 🔧 Preprocessing & EDA

* Handled missing values using **Mean** and **Most Frequent Imputation**
* Removed `Applicant_ID`
* Applied **Label Encoding** and **One-Hot Encoding**
* Used **StandardScaler** for feature scaling
* Analyzed class distribution, income, credit score, outliers, and correlations

The strongest correlations with loan approval were **Credit Score (+0.45)** and **DTI Ratio (-0.44)**.

## ⚙️ Feature Engineering

Created squared features for DTI Ratio and Credit Score:

```python
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2
```

The dataset was split into **80% training** and **20% testing**.

## 🤖 Models Used

* **Logistic Regression**
* **K-Nearest Neighbors (KNN)**
* **Gaussian Naive Bayes**

## 📈 Results

| Model                   |  Accuracy |  Precision |     Recall |   F1-Score |
| ----------------------- | --------: | ---------: | ---------: | ---------: |
| **Logistic Regression** | **88.0%** |     78.46% | **83.61%** | **80.95%** |
| KNN                     |     78.5% |     67.31% |     57.38% |     61.95% |
| Naive Bayes             |     86.0% | **81.13%** |     70.49% |     75.44% |

🏆 **Logistic Regression** achieved the best overall performance with **88% accuracy** and an **80.95% F1-score**.

## 🛠️ Tech Stack

**Python • Pandas • NumPy • Scikit-learn • Matplotlib • Seaborn **

## 🚀 How to Run

```bash
git clone https://github.com/anupam-000/creditwise-loan-system.git
cd creditwise-loan-system
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 👨‍💻 Author

**Anupam Gupta**

GitHub: `anupam-000`
