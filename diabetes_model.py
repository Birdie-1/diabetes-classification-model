# =======================================================
# 1) IMPORT LIBRARIES
# =======================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# =======================================================
# 2) LOAD CLEANED DATASET
# =======================================================
df = pd.read_csv("diabetes_cleaned.csv")

# แยก X (features) และ y (target)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =======================================================
# 3) TRAIN-TEST SPLIT
# =======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # ใช้ 20% สำหรับ test
    random_state=42,
    stratify=y           # ทำให้สัดส่วน Outcome 0/1 เท่าเดิม
)

# =======================================================
# 4) BUILD RANDOM FOREST MODEL
# =======================================================
model = RandomForestClassifier(
    n_estimators=200,      # จำนวนต้นไม้
    random_state=42
)

# ฝึกโมเดล
model.fit(X_train, y_train)

# =======================================================
# 5) PREDICT
# =======================================================
y_pred = model.predict(X_test)

# =======================================================
# 6) EVALUATION METRICS
# =======================================================
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("==== MODEL PERFORMANCE ====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =======================================================
# 7) CONFUSION MATRIX
# =======================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =======================================================
# 8) FEATURE IMPORTANCE
# =======================================================
importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

print("\n==== FEATURE IMPORTANCE ====")
print(importances)
