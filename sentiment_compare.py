import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC           
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)

# ==========================================
# 1. LOAD DATASET
# ==========================================
file_path = "dataset_konekin/sentiment_analyst.csv"   
try:
    df = pd.read_csv(file_path, encoding="latin-1")
    print("✅ File berhasil dibaca!")
except FileNotFoundError:
    print("❌ File tidak ditemukan. Pastikan path benar.")
    exit()

df.dropna(subset=['text', 'sentiment'], inplace=True)
print(f" Dimensi dataset: {df.shape}")

# ==========================================
# 2. EKSPLORASI DATA (EDA)
# ==========================================
print("\n Distribusi Sentimen:")
print(df['sentiment'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='sentiment', order=['positive','negative','neutral'],
              palette='Set2')
plt.title('Distribusi Sentimen')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah Tweet')
plt.show()

# WordCloud
STOPWORDS = set(stopwords.words('english'))
def plot_wordcloud(sentiment, color='white'):
    text = ' '.join(df[df['sentiment'] == sentiment]['text'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color=color,
                          stopwords=STOPWORDS, collocations=False).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'WordCloud - Sentimen {sentiment.upper()}')
    plt.axis('off')
    plt.show()

# ==========================================
# 3. TEXT PREPROCESSING + TF-IDF
# ==========================================
print("\n🧹 Preprocessing teks...")
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub(r'@\w+', '', text)          # hapus mention
    text = re.sub(r'http\S+|www\S+', '', text) # hapus link
    text = re.sub(r'[^a-zA-Z\s]', '', text)   # hanya huruf & spasi
    text = text.lower().strip()
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF dengan unigram + bigram
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))  # N-Grams (1,2)
X = tfidf.fit_transform(df['clean_text'])
y = df['sentiment']

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f" Dimensi TF-IDF: {X.shape}")
print(f" Data latih: {X_train.shape[0]} | 🔹 Data uji: {X_test.shape[0]}")

# ==========================================
# 4. BASELINE: LOGISTIC REGRESSION
# ==========================================
print("\n" + "="*60)
print("🔷 1. LOGISTIC REGRESSION (BASELINE)")
print("="*60)

lr = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"✅ Akurasi Logistic Regression: {acc_lr:.4f}")
print("\n Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

cv_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring='accuracy')
print(f" 5-Fold CV Akurasi: {cv_lr.mean():.4f} (+/- {cv_lr.std()*2:.4f})")

# Confusion Matrix LR
plt.figure(figsize=(6,5))
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==========================================
# 5. MODEL SVM (LinearSVC)
# ==========================================
print("\n" + "="*60)
print(" 2. SUPPORT VECTOR MACHINE (LinearSVC)")
print("="*60)

svm = LinearSVC(random_state=42, max_iter=2000, C=1.0, dual='auto') 
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"✅ Akurasi SVM: {acc_svm:.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

cv_svm = cross_val_score(svm, X_train, y_train, cv=5, scoring='accuracy')
print(f" 5-Fold CV Akurasi: {cv_svm.mean():.4f} (+/- {cv_svm.std()*2:.4f})")

# Confusion Matrix SVM
plt.figure(figsize=(6,5))
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==========================================
# 6. PERBANDINGAN
# ==========================================
print("\n" + "="*60)
print(" PERBANDINGAN MODEL")
print("="*60)
print(f"Logistic Regression Accuracy : {acc_lr:.4f}  (CV: {cv_lr.mean():.4f})")
print(f"SVM (LinearSVC) Accuracy     : {acc_svm:.4f}  (CV: {cv_svm.mean():.4f})")

if acc_svm > acc_lr:
    print("\n SVM UNGGUL dalam akurasi!")
elif acc_svm < acc_lr:
    print("\n Logistic Regression UNGGUL dalam akurasi!")
else:
    print("\n Keduanya SAMA akurasinya.")

# Visualisasi perbandingan
models = ['Logistic Regression', 'SVM (LinearSVC)']
accs = [acc_lr, acc_svm]
cv_means = [cv_lr.mean(), cv_svm.mean()]

fig, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].bar(models, accs, color=['skyblue','orange'])
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Test Accuracy')
for i, v in enumerate(accs):
    ax[0].text(i, v+0.005, f"{v:.4f}", ha='center')

ax[1].bar(models, cv_means, color=['skyblue','orange'])
ax[1].set_ylabel('CV Mean Accuracy')
ax[1].set_title('5-Fold CV Accuracy')
for i, v in enumerate(cv_means):
    ax[1].text(i, v+0.005, f"{v:.4f}", ha='center')
plt.suptitle('Perbandingan Logistic Regression vs SVM')
plt.tight_layout()
plt.show()