import pandas as pd

df = pd.read_csv("dataset/pokemon.csv")
df['Name_clean'] = df['Name'].str.lower().str.replace('[^a-z0-9]', '_', regex=True)
df = df.dropna(subset=['Type1'])   
df['Type2'] = df['Type2'].fillna('')
df['Evolution'] = df['Evolution'].fillna('')

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Type1_label'] = le.fit_transform(df['Type1'])
num_classes = len(le.classes_)

import os

image_dir = "pokemon_images"
valid_exts = ('.png')

image_files = {f.rsplit('.', 1)[0].lower(): os.path.join(image_dir, f)
               for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)}

# Cocokkan: untuk setiap Name_clean, cari di image_files
df['image_path'] = df['Name_clean'].map(image_files)
# simpan hanya yang memiliki gambar
df = df.dropna(subset=['image_path'])
print(f"Jumlah data yang cocok: {len(df)}")

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# Load model dasar tanpa top
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
# Bekukan bobot base_model (optional, nanti bisa fine‑tune)
base_model.trainable = False

# Tambahkan pooling
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)

def load_and_preprocess(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    # preprocess_input khas EfficientNet
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# Batch processing
def extract_image_features(df, batch_size=32):
    features = []
    for idx in range(0, len(df), batch_size):
        batch_paths = df.iloc[idx:idx+batch_size]['image_path'].values
        batch_imgs = np.array([load_and_preprocess(p) for p in batch_paths])
        batch_feat = feature_extractor.predict(batch_imgs, verbose=0)
        features.append(batch_feat)
    return np.vstack(features)

from sklearn.feature_extraction.text import TfidfVectorizer

# Buat string dokumen
df['text_doc'] = df['Type2'] + ' ' + df['Evolution']

vectorizer = TfidfVectorizer(
    lowercase=True,
    analyzer='word',
    token_pattern=r'(?u)\b\w+\b',  # hanya kata
    max_features=500                # batasi fitur (dataset kecil)
)

tfidf_matrix = vectorizer.fit_transform(df['text_doc'])
# Ubah ke array numpy
text_features = tfidf_matrix.toarray()
print("Dimensi TF-IDF:", text_features.shape)

from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model

# ---- Cabang Gambar ----
image_input = Input(shape=(224, 224, 3), name='image_input')
# Gunakan base_model yang sudah dibekukan
x_img = base_model(image_input, training=False)
x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
x_img = Dense(256, activation='relu')(x_img)
x_img = Dropout(0.5)(x_img)

# ---- Cabang Teks ----
text_input = Input(shape=(text_features.shape[1],), name='text_input')
x_txt = Dense(128, activation='relu')(text_input)
x_txt = Dropout(0.3)(x_txt)

# ---- Gabungan ----
combined = Concatenate()([x_img, x_txt])
combined = Dense(128, activation='relu')(combined)
combined = Dropout(0.4)(combined)
output = Dense(num_classes, activation='softmax', name='type_output')(combined)

model = Model(inputs=[image_input, text_input], outputs=output)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# gambar dalam array
from sklearn.model_selection import train_test_split

X_images = np.array([load_and_preprocess(p) for p in df['image_path']])
X_text = text_features
y = df['Type1_label'].values

X_img_train, X_img_val, X_txt_train, X_txt_val, y_train, y_val = train_test_split(
    X_images, X_text, y, test_size=0.2, stratify=y, random_state=42
)

history = model.fit(
    [X_img_train, X_txt_train], y_train,
    validation_data=([X_img_val, X_txt_val], y_val),
    epochs=30,
    batch_size=16,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
)

val_loss, val_acc = model.evaluate([X_img_val, X_txt_val], y_val)
print(f"Validation Accuracy: {val_acc:.4f}")