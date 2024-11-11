from pandas import DataFrame

def k_means(dataframe : DataFrame, k : int) -> tuple[list[list[int]], list[int]]:
    # K değeri 0'dan küçükse hata ver
    if k < 1: raise ValueError("K değeri 0'dan küçük olamaz.")

    # K değeri veri kümesinden fazlaysa hata ver
    if k > len(dataframe): raise ValueError("K değeri veri kümesinden fazla olamaz.")

    # Veri kümesindeki fiyat sütununu al.
    # Değerler numpy değeri olarak dönüyor, bunu integer'a cast et, ardından listeye çevir
    fiyatlar = dataframe["Fiyat"].values.astype(int).tolist()

    # K değeri kadar rastgele merkez seç
    # Merkezlerin tipini float yap ve listeye çevir
    merkezler = dataframe.sample(n=k)["Fiyat"].values.astype(float).tolist()

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

    # Kümeleri ve merkezleri döndür
    return kümeler, merkezler