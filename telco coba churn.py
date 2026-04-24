import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset dari URL GitHub IBM
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print("Shape awal data:", df.shape)

# 2. Basic Data Cleaning (Data Understanding / Preparation)
# Kolom TotalCharges bertipe object (string) karena ada spasi kosong. Kita ubah ke numerik.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop missing values yang dihasilkan dari spasi kosong tadi (sekitar 11 baris)
df.dropna(inplace=True)

print("Shape setelah cleaning:", df.shape)
# TAHAP 3: DATA PREPARATION (BALANCING)

# 1. Menampilkan Distribusi Kelas Sebelum Balancing
print("\n--- Sebelum Balancing ---")
print(df['Churn'].value_counts())

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.countplot(x='Churn', data=df, palette='viridis')
plt.title('Distribusi Churn (Sebelum)')

# 2. Random Undersampling
# Memisahkan data berdasarkan kelas
df_majority = df[df['Churn'] == 'No']
df_minority = df[df['Churn'] == 'Yes']

# Undersample kelas mayoritas agar sama dengan jumlah kelas minoritas
df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)

# Gabungkan dan acak urutannya
df_balanced = pd.concat([df_majority_downsampled, df_minority])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Menampilkan Distribusi Kelas Sesudah Balancing
print("\n--- Sesudah Balancing ---")
print(df_balanced['Churn'].value_counts())

plt.subplot(1, 2, 2)
sns.countplot(x='Churn', data=df_balanced, palette='viridis')
plt.title('Distribusi Churn (Sesudah)')
plt.tight_layout()
plt.show()
# TAHAP 3 (Lanjutan): Encoding & Splitting

# Drop 'customerID' karena tidak relevan untuk pemodelan
X = df_balanced.drop(columns=['customerID', 'Churn'])

# Encode target variable 'Churn' (Yes -> 1, No -> 0)
y = df_balanced['Churn'].map({'Yes': 1, 'No': 0})

# One-Hot Encoding untuk fitur kategorikal lainnya
X = pd.get_dummies(X, drop_first=True)

# Train-Test Split (80% Train, 20% Test)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)

# TAHAP 4: MODELING

# -----------------------------------------------------------------------------------------
# CATATAN ANALIS: MENGAPA LOGISTIC REGRESSION?
# Meskipun bernama "Regression", Logistic Regression adalah algoritma KLASIFIKASI. 
# Algoritma ini memetakan nilai ke dalam fungsi Sigmoid yang menghasilkan probabilitas 
# absolut antara 0 dan 1. Linear Regression murni tidak membatasi outputnya (bisa < 0 
# atau > 1), sehingga kurang pantas secara matematis untuk kasus klasifikasi biner seperti Churn.
# -----------------------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# --- Model 1: Logistic Regression ---
log_reg = LogisticRegression(max_iter=2000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# --- Model 2: Random Forest ---
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

print("Training model selesai!")

# TAHAP 5: EVALUATION
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(y_true, y_pred, model_name):
    """Fungsi pembantu untuk print metrik dan plot Confusion Matrix"""
    print(f"=== Evaluasi {model_name} ===")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y_true, y_pred):.4f}\n")
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
    
    fig, ax = plt.subplots(figsize=(4,3))
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f'CM - {model_name}')
    plt.show()

# Evaluasi Model
evaluate_model(y_test, y_pred_log, "Logistic Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")