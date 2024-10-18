from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time

# WebDriver başlatılıyor - Chrome tarayıcı
print("Tarayıcı başlatılıyor...")
driver = webdriver.Chrome()

# İlanları tutmak için bir liste
ilan_listesi = []

# 20 sayfa boyunca gezinme işlemi
for page in range(1, 21):
    # Sayfa URL'si oluşturuluyor ve siteye gidiliyor
    url = f'https://www.hepsiemlak.com/buca-kiralik?page={page}'
    print(f"Sayfa {page} yükleniyor: {url}")
    driver.get(url)
    time.sleep(5)  # Sayfanın tamamen yüklenmesi için 5 saniye bekleniyor, süre uzatılabilir

    # İlan kartlarını bulma
    ilanlar = driver.find_elements(By.CLASS_NAME, 'listing-item')
    print(f"{len(ilanlar)} ilan bulundu.")

    # Her bir ilan kartını işleme
    for ilan in ilanlar:
        try:
            # Fiyat çıkarılıyor
            fiyat = ilan.find_element(By.CLASS_NAME, 'list-view-price').text

            # Fiyat bilgisi düzenleniyor: 'TL' ve virgüller kaldırılıyor, boşluklar siliniyor
            fiyat = fiyat.replace("TL", "").replace(",", "").strip()

            # İlan başlığı çıkarılıyor
            baslik = ilan.find_element(By.CLASS_NAME, 'card-link').get_attribute('title')

            # İlan konumu çıkarılıyor
            konum = ilan.find_element(By.CLASS_NAME, 'list-view-location').text

            # Detaylar çıkarılıyor
            detay = ilan.find_element(By.CLASS_NAME, 'short-property').text.split("\n")

            # Detaylar listesi kontrol edilerek gerekli alanlar çıkarılıyor
            # Eğer detaylar listesi boş ise, ilgili alanlar boş olarak atanıyor
            oda_sayisi = detay[1] if len(detay) > 1 else ''
            alan = detay[2] if len(detay) > 2 else ''
            bina_yasi = detay[3] if len(detay) > 3 else ''
            kat_sayisi = detay[4] if len(detay) > 4 else ''

            # İlan tarihi çıkarılıyor
            ilan_tarihi = ilan.find_element(By.CLASS_NAME, 'list-view-date').text

            # İlan verileri ilana ekleniyor
            ilan_listesi.append({
                'Fiyat': fiyat,
                'Başlık': baslik,
                'Konum': konum,
                'Oda Sayısı': oda_sayisi,
                'Alan': alan,
                'Bina Yaşı': bina_yasi,
                'Kat': kat_sayisi,
                'İlan Tarihi': ilan_tarihi
            })
            print(f"İlan eklendi: {baslik}")
        except Exception as e:
            # Hata durumunda mesaj yazdırılıyor, işleme bir sonraki sayfadan devam ediliyor
            print(f'Error: {e}')
            continue

print("Tarayıcı kapatılıyor...")
driver.quit()

# İlan verileri DataFrame'e dönüştürülüyor ve Excel dosyasına kaydediliyor
df = pd.DataFrame(ilan_listesi)
df.to_excel('ilanlar.xlsx', index=False)  # Excel dosyası kaydediliyor
print("İlanlar başarıyla kaydedildi: ilanlar.xlsx")  # İşlem başarı mesajı
