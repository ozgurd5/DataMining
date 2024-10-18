import requests
from bs4 import BeautifulSoup
import pandas as pd

# İlanları tutmak için bir liste
ilan_listesi = []

# İlanların linklerini tutmak için bir liste. Eşya ve doğalgaz durumlarına buradan ulaşacağız
ilan_linkleri = []

# 22 sayfa boyunca gezinme işlemi
for page in range(1, 2):
    # Sayfa URL'si oluşturuluyor ve siteye gidiliyor
    url = f'https://www.hepsiemlak.com/buca-kiralik?page={page}'
    print(f"Sayfa {page} yükleniyor: {url}")

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Sayfa yüklenirken hata oluştu: {response.status_code}")
        continue

    soup = BeautifulSoup(response.content, "html.parser")

    print("Sayfa yüklendi.")

    # İlan kartlarını bulma
    ilanlar = soup.find_all("div", class_="listing-item")
    print(f"{len(ilanlar)} ilan bulundu.")

    # Her bir ilan kartını işleme
    for ilan in ilanlar:
        try:
            # Fiyat çıkarılıyor
            fiyat = ilan.find("span", class_="list-view-price").text

            # Fiyat bilgisi düzenleniyor: 'TL' ve virgüller kaldırılıyor, boşluklar siliniyor
            fiyat = fiyat.replace("TL", "").replace(",", "").strip()

            # İlan başlığı çıkarılıyor
            baslik = ilan.find("a", class_="card-link")["title"]

            # İlan konumu çıkarılıyor
            konum = ilan.find("span", class_="list-view-location").text

            # Detaylar çıkarılıyor
            detay = ilan.find("div", class_="short-property").text.strip().split("\n")

            # Detaylar listesi kontrol edilerek gerekli alanlar çıkarılıyor
            # Eğer detaylar listesi boş ise, ilgili alanlar boş olarak atanıyor
            oda_sayisi = detay[1] if len(detay) > 1 else ''
            alan = detay[2] if len(detay) > 2 else ''
            bina_yasi = detay[3] if len(detay) > 3 else ''
            kat_sayisi = detay[4] if len(detay) > 4 else ''

            # İlan tarihi çıkarılıyor
            ilan_tarihi = ilan.find("span", class_="list-view-date").text

            # İlan linkini almak ve tam URL'yi oluşturmak
            ilan_linki = "https://www.hepsiemlak.com" + ilan.find("a", class_="card-link")["href"]
            ilan_linkleri.append(ilan_linki)

            # İlan verileri ilana ekleniyor
            ilan_listesi.append({
                'Fiyat': fiyat,
                'Başlık': baslik,
                'Konum': konum,
                'Oda Sayısı': oda_sayisi,
                'Alan': alan,
                'Bina Yaşı': bina_yasi,
                'Kat': kat_sayisi,
                'İlan Tarihi': ilan_tarihi,
                'İlan Linki': ilan_linki,
            })
            print(f"İlan eklendi: {baslik}")
            print(f"İlan linki eklendi: {ilan_linki}")

        except Exception as e:
            # Hata durumunda mesaj yazdırılıyor, işleme bir sonraki sayfadan devam ediliyor
            print(f'Error: {e}')
            continue

for ilan_linki in ilan_linkleri:
    print(f"İlan linki yükleniyor: {ilan_linki}")
    response = requests.get(ilan_linki)
    soup = BeautifulSoup(response.content, "html.parser")

    try:
        # Tüm span'ları bul ve eşya durumu, yakıt tipi ve ısınma tipini çıkart
        spans = soup.find_all("span", class_="txt")

        esya_durumu = ''
        yakit_tipi = ''
        isinma_tipi = ''

        for span in spans:
            if 'Eşya Durumu' in span.text:
                esya_durumu = span.find_next("span").text

            elif 'Yakıt Tipi' in span.text:
                yakit_tipi = span.find_next("span").text

            elif 'Isınma Tipi' in span.text:
                isinma_tipi = span.find_next("span").text

            if esya_durumu and yakit_tipi and isinma_tipi:
                break

        # İlan listesine 'Eşya Durumu', 'Yakıt Tipi' ve 'Isınma Tipi' ekleniyor
        ilan_index = ilan_linkleri.index(ilan_linki)
        ilan_listesi[ilan_index]['Eşya Durumu'] = esya_durumu
        ilan_listesi[ilan_index]['Yakıt Tipi'] = yakit_tipi
        ilan_listesi[ilan_index]['Isınma Tipi'] = isinma_tipi

        print(f"İlan verileri eklendi: {esya_durumu}, {yakit_tipi}, {isinma_tipi}")

    except Exception as e:
        print(f'Error: {e}')
        continue

# İlan verileri DataFrame'e dönüştürülüyor ve Excel dosyasına kaydediliyor
df = pd.DataFrame(ilan_listesi)
df.to_excel('ilanlar.xlsx', index=False)  # Excel dosyası kaydediliyor
print("İlanlar başarıyla kaydedildi: ilanlar.xlsx")  # İşlem başarı mesajı
