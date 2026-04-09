import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)
n = 1000

MonthlyIncome = np.clip(np.random.normal(130000, 60000, n), 10000, 250000)
CreditScore = np.clip(np.random.normal(680, 80, n), 300, 850)
EmploymentStatus = np.random.binomial(1, 0.85, n)
Debt = MonthlyIncome * np.random.beta(2, 5, n)
DebtToIncomeRatio = Debt / MonthlyIncome
PaymentHistory = np.clip(np.random.normal(0.8, 0.15, n), 0, 1)

df = pd.DataFrame({
    "Monthly Income": MonthlyIncome,
    "Credit Score": CreditScore,
    "Employment Status": EmploymentStatus,
    "Debt-to-Income Ratio": DebtToIncomeRatio,
    "PaymentHistory": PaymentHistory
})

df.head()

score = (0.00003 * df["Monthly Income"] +
    0.01 * df["Credit Score"] +
    0.8 * df["Employment Status"] +
    2.5 * df["PaymentHistory"] -
    5.0 * df["Debt-to-Income Ratio"] - 8
)

probs = 1 / (1 + np.exp(-score))

df["LoanApproved"] = np.random.binomial(1, probs)

df.head()

X = df[["Monthly Income", "Credit Score", "Employment Status", "PaymentHistory", "Debt-to-Income Ratio"]]
y = df["LoanApproved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print ("Predictions: ", predictions)
print("Accuracy: ", accuracy_score(y_test, predictions))

new_data = pd.DataFrame({
    "Monthly Income":[80000],
    "Credit Score": [850],
    "Employment Status": [1],
    "PaymentHistory": [0.477819],
    "Debt-to-Income Ratio": [0.867546]})
result = model.predict(new_data)
print("Approved" if result[0] == 1 else "Rejected")