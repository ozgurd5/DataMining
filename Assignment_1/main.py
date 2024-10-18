from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time

from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions

# WebDriver başlatılıyor - Chrome tarayıcı
print("Tarayıcı başlatılıyor...")
driver = webdriver.Chrome()

# İlanları tutmak için bir liste
ilan_listesi = []

# İlanların linklerini tutmak için bir liste. Eşya ve doğalgaz durumlarına buradan ulaşacağız
ilan_linkleri = []

# 22 sayfa boyunca gezinme işlemi
for page in range(1, 22):
    # Sayfa URL'si oluşturuluyor ve siteye gidiliyor
    url = f'https://www.hepsiemlak.com/buca-kiralik?page={page}'
    print(f"Sayfa {page} yükleniyor: {url}")
    driver.get(url)
    WebDriverWait(driver, 10).until(expected_conditions.presence_of_element_located((By.CLASS_NAME, 'listing-item')))
    print("Sayfa yüklendi.")

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

            # İlan linkini almak ve tam URL'yi oluşturmak
            ilan_linki = ilan.find_element(By.CLASS_NAME, 'card-link').get_attribute('href')
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
    print("İlan linki yükleniyor: ", ilan_linki)
    driver.get(ilan_linki)
    WebDriverWait(driver, 10).until(expected_conditions.presence_of_element_located((By.CLASS_NAME, 'txt')))
    print("İlan linki yüklendi.")

    try:
        # Tüm span'ları bul
        spans = driver.find_elements(By.CLASS_NAME, 'txt')

        esya_durumu = ''
        yakit_tipi = ''
        isinma_tipi = ''

        for span in spans:
            if span.text == 'Eşya Durumu':  # Eğer span 'Eşya Durumu' metnini içeriyorsa
                # Kardeş öğesi olan diğer span'dan Eşya Durumu'nu al
                esya_durumu = span.find_element(By.XPATH, "following-sibling::span").text

            elif span.text == 'Yakıt Tipi':  # Eğer span 'Yakıt Tipi' metnini içeriyorsa
                # Kardeş öğesi olan diğer span'dan Yakıt Tipi'ni al
                yakit_tipi = span.find_element(By.XPATH, "following-sibling::span").text

            elif span.text == 'Isınma Tipi':  # Eğer span 'Isınma Tipi' metnini içeriyorsa
                # Kardeş öğesi olan diğer span'dan Isınma Tipi'ni al
                isinma_tipi = span.find_element(By.XPATH, "following-sibling::span").text

            if esya_durumu and yakit_tipi and isinma_tipi != '':  # Eğer tüm bilgiler alındıysa
                break # Döngüden çık

        # İlan listesine 'Eşya Durumu', 'Yakıt Tipi' ve 'Isınma Tipi' ekleniyor
        ilan_index = ilan_linkleri.index(ilan_linki)
        ilan_listesi[ilan_index]['Eşya Durumu'] = esya_durumu
        ilan_listesi[ilan_index]['Yakıt Tipi'] = yakit_tipi
        ilan_listesi[ilan_index]['Isınma Tipi'] = isinma_tipi

        print(f"İlan verileri eklendi: {esya_durumu}, {yakit_tipi}, {isinma_tipi}")

    except Exception as e:
        print(f'Error: {e}')
        continue


print("Tarayıcı kapatılıyor...")
driver.quit()

# İlan verileri DataFrame'e dönüştürülüyor ve Excel dosyasına kaydediliyor
df = pd.DataFrame(ilan_listesi)
df.to_excel('ilanlar.xlsx', index=False)  # Excel dosyası kaydediliyor
print("İlanlar başarıyla kaydedildi: ilanlar.xlsx")  # İşlem başarı mesajı
