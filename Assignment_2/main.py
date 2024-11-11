# Pandas modülü
import pandas as pd

# Matplotlib modülü
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
# Değerler numpy değeri olarka dönüyor, bunu string'e çevir, ardından regex kullanarak sadece sayıları al ve integer'a çevir
# \D sayı olmayan karakterleri temsil eder
df["Fiyat"] = df["Fiyat"].astype(str).str.replace(r"\D", "", regex=True).astype(int)

# Fiyatları küçükten büyüğe sırala
#df = df.sort_values("Fiyat")

# Kümelenmemiş ama temizlenmiş veriyi excel dosyasına yaz
# print("\nSadece index ve fiyatı içeren 'temizlenmiş_veri_0.xlsx' dosyası oluşturuldu.")
# df.reset_index()[["index", "Fiyat"]].to_excel("temizlenmiş_veri_0.xlsx", index=False)

# K-Means algoritmasına başla
k = 3

# Veri kümesindeki fiyat sütununu al.
# Değerler numpy değeri olarak dönüyor, bunu integer'a cast et, ardından listeye çevir
fiyatlar = df["Fiyat"].values.astype(int).tolist()

# K değeri kadar rastgele merkez seç
# Merkezlerin tipini float yap ve listeye çevir
merkezler = df.sample(n=k)["Fiyat"].values.astype(float).tolist()

# Önceki merkezleri tut, başlangıçta tüm değerleri 0 yap
önceki_merkezler = [0] * k

# Merkezler değişene kadar döngüyü devam ettir
while True:
    # K değeri kadar küme oluştur
    kümeler = []
    for i in range(k):
        kümeler.append([])

    # Veri kümesindeki her fiyat için en yakın merkezi bul
    for fiyat in fiyatlar:
        en_yakin_merkez_index = 0
        en_kucuk_fark = abs(fiyat - merkezler[0]) # Öklit mesafesi

        for i in range(1, k):
            fark = abs(fiyat - merkezler[i]) # Öklit mesafesi
            if fark < en_kucuk_fark:
                en_yakin_merkez_index = i
                en_kucuk_fark = fark

        kümeler[en_yakin_merkez_index].append(fiyat)

    # Her kümenin ortalamasını alarak yeni merkezleri hesapla
    for i in range(k):
        önceki_merkezler[i] = merkezler[i]
        merkezler[i] = sum(kümeler[i]) / len(kümeler[i])

    # Merkezler değişmediyse döngüyü sonlandır
    if önceki_merkezler == merkezler:
        break

# Kümeleri ve merkezleri yazdır
# for i in range(k):
#     print(f"\nKüme {i + 1} ({len(kümeler[i])} eleman):")
#     print(kümeler[i])
#
# print("\nMerkezler:")
# print(merkezler)

# Kümeleri ve merkezleri görselleştirmeye başla

# Plot oluştur
plt.figure(figsize=(10, 6), dpi=300)

# Plot başlığını belirle
plt.title(f"K-Means")

# Index sütununu göster
plt.xlabel("Index")

# Fiyat sütununu göster
plt.ylabel("Fiyat")

# Grid çiz
plt.grid(True)

# Veri kümesindeki index değerlerini x ekseninde göstermek için al
x_values = df.index

# Veri kümesindeki fiyat değerlerini y ekseninde göstermek için al
y_values = df["Fiyat"]

# Her küme için farklı renkler belirle
colors = ["red", "green", "blue"]
color_values = []
for i in range(k):
    color_values.extend([colors[i]] * len(kümeler[i]))

# Veri noktalarını çiz
plt.scatter(x_values, y_values, color=color_values, s=2)

# Merkezleri çiz
plt.scatter(range(k), merkezler, color="black", s=100, marker="x")

# Plot'u kaydet
plt.savefig("plot.png", dpi=300, bbox_inches='tight')

# Plot'u göster
plt.show()
