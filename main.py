# =======================================================
# 1) IMPORT LIBRARIES
# =======================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# โหลดข้อมูลที่ทำความสะอาดแล้ว
df = pd.read_csv("diabetes_cleaned.csv")

# ตั้งค่าธีมกราฟสวยๆ
sns.set(style="whitegrid")

# =======================================================
# HISTOGRAM ของตัวแปรสำคัญ
# =======================================================
features = ["Glucose", "BMI", "Age", "BloodPressure"]

plt.figure(figsize=(12, 8))
df[features].hist(bins=20, figsize=(12, 8))
plt.suptitle("Histogram of Glucose, BMI, Age, BloodPressure")
plt.show()


# =======================================================
# BOXPLOT เปรียบเทียบตัวแปรตาม Outcome
# =======================================================

# Glucose vs Outcome
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Outcome", y="Glucose")
plt.title("Glucose vs Outcome (Boxplot)")
plt.show()

# BMI vs Outcome
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Outcome", y="BMI")
plt.title("BMI vs Outcome (Boxplot)")
plt.show()


# =======================================================
# CORRELATION HEATMAP
# =======================================================
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# =======================================================
# BAR CHART: จำนวนผู้ป่วยเบาหวาน
# =======================================================
plt.figure(figsize=(6, 4))
df['Outcome'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Count of Diabetes Outcome")
plt.xlabel("Outcome (0 = No, 1 = Yes)")
plt.ylabel("Number of People")
plt.show()


# =======================================================
# PAIRPLOT (Optional)
# =======================================================
sns.pairplot(df[["Glucose","BMI","Age","BloodPressure","Outcome"]], hue="Outcome")
plt.show()
