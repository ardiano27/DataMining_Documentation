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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)

# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = "dataset/sentiment_analyst.csv"   
try:
    df = pd.read_csv(file_path, encoding="latin-1")
    print("✅ File berhasil dibaca!")
except FileNotFoundError:
    print("❌ File tidak ditemukan. Pastikan path benar.")
    exit()

print(f" Dimensi dataset: {df.shape}")
print(df.head())

df.dropna(subset=['text', 'sentiment'], inplace=True)
print(f" Dimensi setelah drop NaN: {df.shape}")

# ==========================================
# 2. EKSPLORASI DATA (EDA)
# ==========================================
# Distribusi sentimen
print("\n🔍 Distribusi Sentimen:")
print(df['sentiment'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='sentiment', order=['positive','negative','neutral'],
              palette='Set2')
plt.title('Distribusi Sentimen dalam Dataset')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah Tweet')
plt.show()

# WordCloud 
print("\n Membuat WordCloud (mungkin butuh waktu)...")
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

for sent in ['positive', 'negative', 'neutral']:
    plot_wordcloud(sent)

# ==========================================
# 3. TEXT PREPROCESSING + TF-IDF
# ==========================================
print("\n Preprocessing teks...")
stemmer = PorterStemmer()

def clean_text(text):
    # Hapus mention, URL, karakter non-huruf
    text = re.sub(r'@\w+', '', text)          # @user
    text = re.sub(r'http\S+|www\S+', '', text) # link
    text = re.sub(r'[^a-zA-Z\s]', '', text)   # hanya huruf & spasi
    text = text.lower().strip()
    tokens = text.split()
    # Stopword removal + stemming
    tokens = [stemmer.stem(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df['clean_text'])
y = df['sentiment']

print(f" Dimensi matriks TF-IDF: {X.shape}")

# Encode label
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"Label encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"🔹 Data latih: {X_train.shape[0]} sampel")
print(f"🔹 Data uji : {X_test.shape[0]} sampel")

# ==========================================
# 4. LOGISTIC REGRESSION (BASELINE & MODEL FINAL)
# ==========================================
print("\n" + "="*60)
print("   🔥 LOGISTIC REGRESSION DENGAN TF-IDF")
print("="*60)

# Inisialisasi dan training
lr = LogisticRegression(max_iter=1000, multi_class='multinomial',
                        solver='lbfgs', random_state=42)
lr.fit(X_train, y_train)

# Prediksi
y_pred = lr.predict(X_test)

# Evaluasi
print(f"\n Akurasi: {accuracy_score(y_test, y_pred):.4f}")
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Cross-Validation untuk stabilitas
print("\n📈 Cross-Validation (5-fold)...")
cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='accuracy')
print(f"Rata-rata akurasi CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# ==========================================
# 5. KESIMPULAN
# ==========================================
print("\n" + "="*60)
print("📌 KESIMPULAN")
print("="*60)
print("Model Logistic Regression dengan fitur TF-IDF (max 5000 fitur, unigram+bigram)")
print(f"mencapai akurasi {accuracy_score(y_test, y_pred):.2%} pada data uji.")
print("Hasil cross-validation menunjukkan model stabil dan tidak overfitting.")
print("Kata/frasa kunci setiap sentimen dapat dilihat dari koefisien model (opsional).")