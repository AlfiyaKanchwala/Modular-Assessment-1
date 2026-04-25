import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_excel(r"C:\Users\Alifya\Desktop\DMA\credit_fraud.csv.xlsx")

# Display Dataset
print(df.head())

# Dataset Shape
print(df.shape)

# Column Names
print(df.columns)

# Dataset Information
print(df.info())

# Statistical Summary
print(df.describe())

# Check Missing Values
print(df.isnull().sum())

# Handle Missing Values
numeric_columns = df.select_dtypes(include=np.number).columns

imputer = SimpleImputer(strategy='mean')

df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# Remove Duplicate Records
df = df.drop_duplicates()

print("Dataset Shape After Removing Duplicates:")
print(df.shape)

# Outlier Detection
plt.figure(figsize=(12,6))
sns.boxplot(data=df[numeric_columns])
plt.xticks(rotation=90)
plt.title("Outlier Detection")
plt.show()

# Fraud vs Non Fraud Distribution
sns.countplot(x='is_fraud', data=df)
plt.title("Fraud vs Non Fraud Transactions")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(15,10))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(numeric_df.corr(), cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()

# Feature Selection
X = df.drop('is_fraud', axis=1)

# Keep only numeric columns
X = X.select_dtypes(include=np.number)

y = df['is_fraud']

# Data Normalization
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# Decision Tree Classification
# -----------------------------------

model_dt = DecisionTreeClassifier(random_state=42)

model_dt.fit(X_train, y_train)

y_pred_dt = model_dt.predict(X_test)

print("\nDecision Tree Results")

print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall:", recall_score(y_test, y_pred_dt))
print("F1 Score:", f1_score(y_test, y_pred_dt))

print(classification_report(y_test, y_pred_dt))

# Decision Tree Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(6,5))

sns.heatmap(cm_dt, annot=True, fmt='d')

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# -----------------------------------
# Naive Bayes Classification
# -----------------------------------

model_nb = GaussianNB()

model_nb.fit(X_train, y_train)

y_pred_nb = model_nb.predict(X_test)

print("\nNaive Bayes Results")

print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Precision:", precision_score(y_test, y_pred_nb))
print("Recall:", recall_score(y_test, y_pred_nb))
print("F1 Score:", f1_score(y_test, y_pred_nb))

print(classification_report(y_test, y_pred_nb))

# Naive Bayes Confusion Matrix
cm_nb = confusion_matrix(y_test, y_pred_nb)

plt.figure(figsize=(6,5))

sns.heatmap(cm_nb, annot=True, fmt='d')

plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# -----------------------------------
# K-Means Clustering
# -----------------------------------

kmeans = KMeans(n_clusters=2, random_state=42)

clusters = kmeans.fit_predict(X_scaled)

silhouette = silhouette_score(X_scaled, clusters)

print("\nSilhouette Score:", silhouette)

# Cluster Visualization
plt.figure(figsize=(8,6))

plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters)

plt.title("K-Means Clustering")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()

# -----------------------------------
# Association Rule Mining
# -----------------------------------

binary_df = X.copy()

for col in binary_df.columns:
    binary_df[col] = pd.cut(binary_df[col], bins=2, labels=[0,1])

binary_df = binary_df.astype(int)

frequent_items = apriori(
    binary_df,
    min_support=0.1,
    use_colnames=True
)

rules = association_rules(
    frequent_items,
    metric="confidence",
    min_threshold=0.5
)

print("\nAssociation Rules")

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

print("\nProject Execution Completed Successfully")