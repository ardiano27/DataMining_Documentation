import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

#  PREPROCESSING
# Hapus kolom customerID karena nggak ngaruh ke tebakan model
df.drop('customerID', axis=1, inplace=True)

# Ubah TotalCharges jadi angka (karena di data aslinya ada yang berupa spasi kosong)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Hapus data yang TotalCharges-nya kosong (cuma dikit kok, sekitar 11 baris)
df.dropna(inplace=True)

# BALANCING DATA (UNDERSAMPLING)
# Pisahkan pelanggan yang setia (No) dan yang kabur (Yes)
df_no_churn = df[df['Churn'] == 'No'] 
df_churn = df[df['Churn'] == 'Yes']

# Pangkas yang 'No' biar jumlahnya sama persis kayak yang 'Yes' (sekitar 1869)
df_no_churn_downsampled = resample(df_no_churn, 
                                   replace=False, 
                                   n_samples=len(df_churn), # Ngikutin jumlah Churn 'Yes'
                                   random_state=42) 

# Gabungkan lagi jadi satu dataset yang udah seimbang
df_balanced = pd.concat([df_no_churn_downsampled, df_churn])

print("Jumlah data setelah di-balance:")
print(df_balanced['Churn'].value_counts())
print("-" * 30)

# 4. UBAH TEKS JADI ANGKA (ENCODING)
# Pisahkan Fitur (X) dan Target (y)
X = df_balanced.drop('Churn', axis=1)
y = df_balanced['Churn'].map({'No': 0, 'Yes': 1}) # Target diubah jadi 0 dan 1

# Ubah semua fitur teks jadi angka (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# 5. SPLIT DATA & MODELING (DECISION TREE)
# Bagi data 80% untuk Training, 20% untuk Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Panggil model Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)

# Mulai belajar (Training)
dt_model.fit(X_train, y_train)

# Coba tebak data Testing
y_pred = dt_model.predict(X_test)

# 6. LIHAT HASIL RAPORNYA
print("\n=== Evaluasi Decision Tree (Balanced) ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Atur ukuran kanvas gambarnya (biar nggak kekecilan)
plt.figure(figsize=(20, 10))

# Bikin gambar pohonnya
plot_tree(dt_model, 
          feature_names=X.columns, 
          class_names=['No Churn', 'Churn'], 
          filled=True, 
          rounded=True, 
          fontsize=10,
          max_depth=3) # Dibatasi 3 tingkat aja biar gambarnya rapi dan bisa dibaca

# Tampilkan gambarnya
plt.show()