#%%
#Load libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
#%%
# Load data
#balance data so prevalence is equal
df = pd.read_csv("creditcard.csv")

print(df.info())
print(df.describe())

X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

#%%
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#%%
# Scale features (important for logistic regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit model — class_weight='balanced' handles the heavy class imbalance
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train, y_train)

#%%
# Evaluate
y_pred = model.predict(X_test)
 
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

# %%
#lower threshold
