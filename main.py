import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

# 1. Load Dataset
data = pd.read_csv("winequality-red.csv")
print("Dataset Head:")
print(data.head())

# 2. Exploratory Data Analysis (EDA)
# Check for missing values
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull())
plt.title("Missing Values Heatmap")
plt.show()

# Target distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="quality", data=data)
plt.title("Wine Quality Class Distribution")
plt.show()

# 3. Features & Target Extraction
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 4. Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ---------------------------------------------------------
# Model 1: K-Nearest Neighbors (KNN)
# ---------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)

knn_acc = accuracy_score(y_test, knn_pred)
print(f"=== K-Nearest Neighbors ===")
print(f"Accuracy: {knn_acc}")
cmknn = confusion_matrix(y_test, knn_pred)
sns.heatmap(cmknn, annot=True)
plt.title("KNN Confusion Matrix")
plt.show()
print(classification_report(y_test, knn_pred))

# ---------------------------------------------------------
# Model 2: Logistic Regression
# ---------------------------------------------------------
logm = LogisticRegression()
logm.fit(x_train, y_train)
logm_pred = logm.predict(x_test)

aclogm = accuracy_score(y_test, logm_pred)
print(f"=== Logistic Regression ===")
print(f"Accuracy: {aclogm}")
cmlogm = confusion_matrix(y_test, logm_pred)
sns.heatmap(cmlogm, annot=True)
plt.title("Logistic Regression Confusion Matrix")
plt.show()
print(classification_report(y_test, logm_pred))

# ---------------------------------------------------------
# Model 3: Gaussian Naïve Bayes
# ---------------------------------------------------------
nb = GaussianNB()
nb.fit(x_train, y_train)
nb_pred = nb.predict(x_test)

acnb = accuracy_score(y_test, nb_pred)
print(f"=== Gaussian Naïve Bayes ===")
print(f"Accuracy: {acnb}")
cmnb = confusion_matrix(y_test, nb_pred)
sns.heatmap(cmnb, annot=True)
plt.title("Gaussian Naïve Bayes Confusion Matrix")
plt.show()
print(classification_report(y_test, nb_pred))

# ---------------------------------------------------------
# Model 4: Support Vector Machine (Linear Kernel)
# ---------------------------------------------------------
svmmodel = SVC(kernel="linear")
svmmodel.fit(x_train, y_train)
svm_pred = svmmodel.predict(x_test)

acsvm = accuracy_score(y_test, svm_pred)
print(f"=== Support Vector Machine ===")
print(f"Accuracy: {acsvm}")
cmsvm = confusion_matrix(y_test, svm_pred)
sns.heatmap(cmsvm, annot=True)
plt.title("SVM Confusion Matrix")
plt.show()
print(classification_report(y_test, svm_pred))

# ---------------------------------------------------------
# Model 5: Decision Tree Classifier
# ---------------------------------------------------------
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
dt_pred = dt.predict(x_test)

dtac = accuracy_score(y_test, dt_pred)
print(f"=== Decision Tree ===")
print(f"Accuracy: {dtac}")
cmdt = confusion_matrix(y_test, dt_pred)
sns.heatmap(cmdt, annot=True)
plt.title("Decision Tree Confusion Matrix")
plt.show()
print(classification_report(y_test, dt_pred))

# ---------------------------------------------------------
# Model 6: Random Forest Classifier
# ---------------------------------------------------------
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)

rf_acc = accuracy_score(y_test, rf_pred)
print(f"=== Random Forest Classifier ===")
print(f"Accuracy: {rf_acc}")
cmrf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cmrf, annot=True)
plt.title("Random Forest Confusion Matrix")
plt.show()
print(classification_report(y_test, rf_pred))
