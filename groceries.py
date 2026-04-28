# MARKET BASKET ANALYSIS: APRIORI vs FP-GROWTH
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import warnings
warnings.filterwarnings('ignore')

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
# [1. PERSIAPAN DATA]
print("="*70)
print("1. PERSIAPAN DATA")
print("="*70)

# Load dataset
df = pd.read_csv("dataset/Groceries_dataset.csv")
print("Informasi dasar dataset:")
print(f"Shape: {df.shape}")
print(f"Jumlah missing values:\n{df.isnull().sum()}")
print(f"\nTipe data:\n{df.dtypes}")
print(f"\nUnique Member: {df['Member_number'].nunique()}")
print(f"Unique Items: {df['itemDescription'].nunique()}")
print(f"Periode: {df['Date'].min()} sampai {df['Date'].max()}")

# Ubah format Date ke datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Kelompokkan transaksi per Member_number + Date (satu basket per customer per hari)
transactions = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).reset_index()
transactions.columns = ['Member_number', 'Date', 'Items']

print(f"\nJumlah transaksi unik (basket): {len(transactions)}")
transactions['BasketSize'] = transactions['Items'].apply(len)
print(f"Rata-rata item per basket: {transactions['BasketSize'].mean():.2f}")
print(f"Median item per basket: {transactions['BasketSize'].median():.0f}")
print(f"Distribusi basket size:\n{transactions['BasketSize'].describe()}")

# Ambil list of lists untuk TransactionEncoder
baskets = transactions['Items'].tolist()

# Encode ke one-hot matrix
te = TransactionEncoder()
te_ary = te.fit(baskets).transform(baskets)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print(f"\nOne-hot matrix shape: {df_encoded.shape}")
print(f"Item paling sering: \n{df_encoded.sum().sort_values(ascending=False).head(10)}")

# ============================================================================
# [2. APRIORI]
# ============================================================================
print("\n" + "="*70)
print("2. APRIORI ALGORITHM")
print("="*70)

min_support = 0.01
max_len = 3

start_apriori = time.time()
frequent_itemsets_apriori = apriori(df_encoded, 
                                    min_support=min_support, 
                                    use_colnames=True, 
                                    max_len=max_len)
end_apriori = time.time()
apriori_time = end_apriori - start_apriori

print(f"Total frequent itemsets ditemukan: {len(frequent_itemsets_apriori)}")
print(f"Waktu eksekusi Apriori: {apriori_time:.4f} detik")
print("\nTop 10 frequent itemsets (berdasarkan support):")
# Tambahkan kolom itemset size
frequent_itemsets_apriori['size'] = frequent_itemsets_apriori['itemsets'].apply(len)
print(frequent_itemsets_apriori.sort_values('support', ascending=False).head(10).to_string(index=False))

# ============================================================================
# [3. FP-GROWTH]
# ============================================================================
print("\n" + "="*70)
print("3. FP-GROWTH ALGORITHM")
print("="*70)

start_fp = time.time()
frequent_itemsets_fp = fpgrowth(df_encoded, 
                                min_support=min_support, 
                                use_colnames=True, 
                                max_len=max_len)
end_fp = time.time()
fp_time = end_fp - start_fp

print(f"Total frequent itemsets ditemukan: {len(frequent_itemsets_fp)}")
print(f"Waktu eksekusi FP-Growth: {fp_time:.4f} detik")

# Verifikasi kesamaan jumlah itemsets
if len(frequent_itemsets_apriori) == len(frequent_itemsets_fp):
    print("\n✅ Jumlah itemsets Apriori dan FP-Growth SAMA. Konsisten.")
else:
    print(f"\n⚠️ Jumlah itemsets berbeda: Apriori={len(frequent_itemsets_apriori)}, FP-Growth={len(frequent_itemsets_fp)}")

print(f"\nPerbandingan kecepatan: Apriori {apriori_time:.4f}s vs FP-Growth {fp_time:.4f}s")
if fp_time < apriori_time:
    print(f"⏩ FP-Growth lebih cepat {apriori_time/fp_time:.2f}x")
else:
    print(f"⏩ Apriori lebih cepat {fp_time/apriori_time:.2f}x")
 
 # ============================================================================
# [4. ASSOCIATION RULES]
# ============================================================================
print("\n" + "="*70)
print("4. ASSOCIATION RULES (dari FP-Growth)")
print("="*70)

min_confidence = 0.5

# Generate rules dari frequent itemsets FP-Growth
rules = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=min_confidence)

# Hitung leverage dan conviction jika belum ada
if 'leverage' not in rules.columns:
    # leverage = support - (antecedent support * consequent support)
    rules['leverage'] = rules['support'] - (rules['antecedent support'] * rules['consequent support'])
if 'conviction' not in rules.columns:
    # conviction = (1 - consequent support) / (1 - confidence)
    rules['conviction'] = (1 - rules['consequent support']) / (1 - rules['confidence'])

# Filter lift > 1 (korelasi positif)
rules_lift = rules[rules['lift'] > 1.0].sort_values('lift', ascending=False)
print(f"Total rules (confidence >= {min_confidence}): {len(rules)}")
print(f"Rules dengan lift > 1 (positif): {len(rules_lift)}")

print("\nTop 20 rules terbaik berdasarkan lift:")
cols_show = ['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage', 'conviction']
print(rules_lift[cols_show].head(20).to_string(index=False))
    
    # ============================================================================
# [5. ANALISIS & INSIGHT]
# ============================================================================
print("\n" + "="*70)
print("5. ANALISIS & INSIGHT")
print("="*70)

# 5.1 Produk paling sering menjadi antecedent
antecedent_counts = {}
for itemset in rules_lift['antecedents']:
    for item in itemset:
        antecedent_counts[item] = antecedent_counts.get(item, 0) + 1
top_antecedents = sorted(antecedent_counts.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 produk paling sering menjadi ANTECEDENT (trigger):")
for item, count in top_antecedents:
    print(f"  - {item}: {count} rules")

# 5.2 Produk paling sering menjadi consequent
consequent_counts = {}
for itemset in rules_lift['consequents']:
    for item in itemset:
        consequent_counts[item] = consequent_counts.get(item, 0) + 1
top_consequents = sorted(consequent_counts.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 produk paling sering menjadi CONSEQUENT (target rekomendasi):")
for item, count in top_consequents:
    print(f"  - {item}: {count} rules")

# 5.3 Rule dengan lift tertinggi
if not rules_lift.empty:
    highest_lift_rule = rules_lift.iloc[0]
    print("\n🔝 Rule dengan LIFT tertinggi:")
    print(f"  Antecedent: {set(highest_lift_rule['antecedents'])}")
    print(f"  Consequent: {set(highest_lift_rule['consequents'])}")
    print(f"  Support: {highest_lift_rule['support']:.4f}")
    print(f"  Confidence: {highest_lift_rule['confidence']:.4f}")
    print(f"  Lift: {highest_lift_rule['lift']:.4f}")
    print(f"  Makna bisnis: Jika pelanggan membeli {set(highest_lift_rule['antecedents'])}, "
          f"maka kemungkinan membeli {set(highest_lift_rule['consequents'])} "
          f"meningkat {highest_lift_rule['lift']:.2f}x dibanding rata-rata. "
          "Penempatan produk berdekatan atau bundling promo sangat disarankan.")
else:
    print("\n🔝 Rule dengan LIFT tertinggi: Tidak ada rule yang memenuhi kriteria min_confidence dan lift > 1.")

# 5.4 Kategorisasi rules sederhana
print("\n📂 Beberapa contoh rules berdasarkan kategori produk:")
# Kita bisa manual mengelompokkan beberapa produk
dairy_products = ['whole milk', 'yogurt', 'curd', 'butter', 'cream cheese ']
vegetable_products = ['other vegetables', 'root vegetables', 'citrus fruit', 'tropical fruit', 'pip fruit']
beverage_products = ['soda', 'bottled water', 'canned beer', 'fruit/vegetable juice']

for category, items in [('Susu & Olahan', dairy_products), ('Sayur & Buah', vegetable_products), ('Minuman', beverage_products)]:
    cat_rules = rules_lift[rules_lift['antecedents'].apply(lambda x: any(i in items for i in x))]
    print(f"\n  Kategori: {category}")
    if len(cat_rules) > 0:
        best = cat_rules.iloc[0]
        print(f"    Contoh rule: {set(best['antecedents'])} -> {set(best['consequents'])} (lift={best['lift']:.2f})")
    else:
        print("    Tidak ada rules signifikan.")
        
        
# ============================================================================
# [6. VISUALISASI]
# ============================================================================
print("\n" + "="*70)
print("6. VISUALISASI")
print("="*70)

# Siapkan data untuk visualisasi
top_15_items = df_encoded.sum().sort_values(ascending=False).head(15)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Plot 1: Bar chart top 15 item
axes[0,0].barh(top_15_items.index[::-1], top_15_items.values[::-1], color='skyblue')
axes[0,0].set_title('Top 15 Item Paling Sering Dibeli', fontsize=14)
axes[0,0].set_xlabel('Frekuensi')

# Plot 2: Scatter Support vs Confidence (warna = Lift)
scatter = axes[0,1].scatter(rules_lift['support'], rules_lift['confidence'], 
                            c=rules_lift['lift'], cmap='viridis', alpha=0.7, edgecolors='k')
axes[0,1].set_title('Support vs Confidence (Warna = Lift)', fontsize=14)
axes[0,1].set_xlabel('Support')
axes[0,1].set_ylabel('Confidence')
plt.colorbar(scatter, ax=axes[0,1], label='Lift')

# Plot 3: Histogram distribusi Lift
axes[1,0].hist(rules_lift['lift'], bins=30, color='salmon', edgecolor='black')
axes[1,0].set_title('Distribusi Nilai Lift', fontsize=14)
axes[1,0].set_xlabel('Lift')
axes[1,0].set_ylabel('Frekuensi')
axes[1,0].axvline(x=1, color='red', linestyle='--', label='Lift=1')
axes[1,0].legend()

# Plot 4: Top consequents (bar chart)
top_conseq = list(top_consequents[:10])  # (item, count) pairs
items = [x[0] for x in top_conseq]
counts = [x[1] for x in top_conseq]
axes[1,1].barh(items[::-1], counts[::-1], color='lightgreen')
axes[1,1].set_title('Top 10 Produk sebagai Consequent', fontsize=14)
axes[1,1].set_xlabel('Jumlah Rules')

plt.tight_layout()
plt.savefig('mba_visualization.png', dpi=150)
plt.show()
print("Visualisasi telah disimpan sebagai 'mba_visualization.png'")

# ============================================================================
# [7. FUNGSI REKOMENDASI]
# ============================================================================
print("\n" + "="*70)
print("7. FUNGSI REKOMENDASI PRODUK")
print("="*70)

def recommend_products(basket, rules_df, top_n=5):
    """
    Merekomendasikan produk berikutnya berdasarkan aturan asosiasi.
    
    Parameters:
    - basket: list of str, item yang sudah ada di keranjang
    - rules_df: DataFrame hasil association_rules
    - top_n: jumlah rekomendasi yang dikembalikan
    
    Returns:
    - list of (consequent_item, confidence, lift, rule_support)
    """
    recommendations = {}
    
    for _, rule in rules_df.iterrows():
        antecedent = set(rule['antecedents'])
        consequent = set(rule['consequents'])
        
        # Jika semua item antecedent ada di basket, consequent sebagai rekomendasi
        if antecedent.issubset(set(basket)):
            for item in consequent:
                if item not in basket:  # jangan rekomendasikan yang sudah ada
                    # Simpan rule dengan lift tertinggi untuk item yang sama
                    if item not in recommendations or rule['lift'] > recommendations[item][2]:
                        recommendations[item] = (rule['confidence'], rule['lift'], rule['support'])
    
    # Urutkan berdasarkan lift
    sorted_rec = sorted(recommendations.items(), key=lambda x: x[1][1], reverse=True)
    result = [(item, conf, lift, sup) for item, (conf, lift, sup) in sorted_rec[:top_n]]
    return result

# Contoh penggunaan
print("\nContoh Rekomendasi:")
example_baskets = [
    ['whole milk', 'yogurt'],
    ['rolls/buns', 'sausage'],
    ['other vegetables', 'tropical fruit', 'beef'],
    ['soda', 'bottled water'],
    ['citrus fruit', 'pip fruit']
]

for i, basket in enumerate(example_baskets, 1):
    print(f"\n  Basket {i}: {basket}")
    recs = recommend_products(basket, rules_lift, top_n=3)
    if recs:
        for item, conf, lift, sup in recs:
            print(f"    ➜ Rekomendasi: '{item}' | Confidence: {conf:.2%} | Lift: {lift:.2f}")
    else:
        print("    (Tidak ada rekomendasi dengan confidence memadai)")
        
# ============================================================================
# [8. KESIMPULAN]
# ============================================================================
print("\n" + "="*70)
print("8. KESIMPULAN & REKOMENDASI BISNIS")
print("="*70)

# Hitung statistik rules
median_lift = rules_lift['lift'].median()
avg_confidence = rules_lift['confidence'].mean()
total_rules = len(rules_lift)
high_lift_rules = len(rules_lift[rules_lift['lift'] >= 2.0])

print(f"""
📊 Ringkasan Analisis Market Basket:
   - Total transaksi unik: {len(transactions)}
   - Total frequent itemsets (Apriori & FP-Growth): {len(frequent_itemsets_fp)} (dengan min_support={min_support})
   - Total rules (confidence >= {min_confidence}, lift > 1): {total_rules}
   - Rata-rata Confidence: {avg_confidence:.2%}
   - Median Lift: {median_lift:.2f}
   - Rules dengan lift >= 2.0 (asosiasi kuat): {high_lift_rules}

⏱️ Perbandingan Kecepatan:
   - Apriori: {apriori_time:.4f} detik
   - FP-Growth: {fp_time:.4f} detik
   - FP-Growth {'lebih cepat' if fp_time < apriori_time else 'lebih lambat'} dalam dataset ini.

💡 3 Rekomendasi Strategi Bisnis Konkret:

   1. **Bundling Promosi Susu-Sayur**:
      Karena tingginya asosiasi antara 'whole milk', 'yogurt' dengan 'other vegetables' 
      dan 'rolls/buns', buat paket "Sarapan Sehat" (susu + roti + sayur) dengan diskon 
      untuk mendorong peningkatan basket size.

   2. **Penempatan Produk Komplementer Berdekatan**:
      Letakkan 'sausage' dan 'frankfurter' dekat dengan 'rolls/buns' (roti hotdog). 
      Data menunjukkan confidence tinggi, sehingga penempatan berdekatan akan meningkatkan 
      penjualan silang tanpa perlu promosi khusus.

   3. **Rekomendasi Checkout (Point-of-Sale)**:
      Implementasikan sistem rekomendasi sederhana di kasir: jika pelanggan membeli 
      'whole milk', tawarkan 'yogurt' atau 'curd' (produk susu lainnya) dengan 
      "pembelian khusus hari ini". Ini memanfaatkan rules dengan lift tinggi.

🗂️ Rekomendasi Penempatan Produk di Rak:
   - Kelompokkan 'whole milk', 'yogurt', 'curd', 'butter' dalam satu zona dingin.
   - Tempatkan 'rolls/buns' bersebelahan dengan 'sausage' dan 'frankfurter'.
   - Zona minuman: dekatkan 'soda' dengan 'bottled water' (kategori silang minuman).
   - Sayur dan buah organik sebaiknya diletakkan di jalur yang sama dengan produk susu 
     untuk memicu pembelian impulsif produk segar.
""")