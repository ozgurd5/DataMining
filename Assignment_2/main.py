import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Excel dosyasını oku
df = pd.read_excel("ham_veri.xlsx")

# Tüm satır ve sütunları göstermek için pandas ayarlarını güncelle
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)

# Varsayılan ayarlara dönmek için
# pd.reset_option("display.max_rows")
# pd.reset_option("display.max_columns")

# Duplikeleri Link sütununa göre tespit et ve yazdır
duplikeler = df[df.duplicated(subset="Link", keep="first")]
print("Duplikeler (aynı linke sahip satırlar): ", len(duplikeler))
print(duplikeler[["Başlık", "Fiyat", "Link"]])

# Boş fiyatları tespit et ve yazdır
bos_fiyatlar = df[df["Fiyat"].isna()]
print("\nBoş Fiyatı Olan Satırlar:", len(bos_fiyatlar))
print(bos_fiyatlar[["Başlık", "Fiyat", "Link"]])

# Temizlik işlemleri
# Aynı linke sahip satırları kaldır
df = df.drop_duplicates(subset="Link", keep="first")

# Fiyatı olmayan satırları kaldır
df = df.dropna(subset=["Fiyat"])

# Fiyatları düz sayı formatına çevir
df["Fiyat"] = df["Fiyat"].astype(str).str.replace(r"\D", "", regex=True).astype(int)

# Fiyatları K-ortalama ile kümelere ayır
kmeans = KMeans(n_clusters=3, random_state=0)
df["Küme"] = kmeans.fit_predict(df[["Fiyat"]])

# Küme merkezlerine göre sıralayıp kategorilere ayır
cluster_centers = kmeans.cluster_centers_.flatten()
sorted_clusters = sorted(range(3), key=lambda x: cluster_centers[x])

# Küme numarasını açıklayıcı bir kategoriye dönüştür
cluster_map = {sorted_clusters[0]: "Ucuz", sorted_clusters[1]: "Orta", sorted_clusters[2]: "Pahalı"}
df["Fiyat_Kategorisi"] = df["Küme"].map(cluster_map)

# Sadece index, fiyat ve kümeyi içeren yeni bir Excel dosyası oluştur
df.reset_index()[["index", "Fiyat", "Küme", "Fiyat_Kategorisi"]].to_excel("temizlenmiş_veri.xlsx", index=False)
print("\nSadece index, fiyat, küme ve fiyat kategorisini içeren 'temizlenmiş_veri.xlsx' dosyası oluşturuldu.")

# Kümelenmiş veriyi görselleştirmek için bir grafik oluştur
plt.figure(figsize=(10, 6))

# Her küme için farklı bir renk kullanarak veriyi çiz
colors = ["green", "orange", "red"]
labels = ["Ucuz", "Orta", "Pahalı"]

for i, label in enumerate(labels):
    cluster_data = df[df["Fiyat_Kategorisi"] == label]
    plt.scatter(cluster_data.index, cluster_data["Fiyat"], color=colors[i], label=label, alpha=0.6, s=50)

# Grafik başlıkları ve etiketleri
plt.title("Fiyatlara Göre Kümelenmiş Kiralık İlanlar")
plt.xlabel("İlan Index")
plt.ylabel("Fiyat")
plt.legend(title="Kategoriler")
plt.grid(True)

# Grafiği göster
plt.show()
