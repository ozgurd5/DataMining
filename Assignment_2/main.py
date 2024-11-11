# Pandas modülü
import pandas as pd

# Matplotlib modülü
import matplotlib.pyplot as plt

# K-Means algoritması
from k_means import k_means

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

# Fiyatları küçükten büyüğe sırala
df = df.sort_values("Fiyat")

# Temizlenmiş veriyi yeni bir Excel dosyasına yaz
print("\nSadece index ve fiyatı içeren 'temizlenmiş_veri_0.xlsx' dosyası oluşturuldu.")
df.reset_index()[["index", "Fiyat"]].to_excel("temizlenmiş_veri_0.xlsx", index=False)

# K-Means algoritması
k = 3
kümeler, merkezler = k_means(df, k)

# Kümeleri ve merkezleri yazdır
for i in range(k):
    print(f"\nKüme {i + 1} ({len(kümeler[i])} eleman):")
    print(kümeler[i])

print("\nMerkezler:")
print(merkezler)


