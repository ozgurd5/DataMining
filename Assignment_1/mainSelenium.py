# Pandas modülü
import pandas as pd

# Selenium modülü
from selenium import webdriver

# Sayfada filtreleme işlemi yapabilmek için gerekli modüller
from selenium.webdriver.common.by import By

# Sayfanın yüklendiğini algılamamız için gerekli modüller
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait

# WebDriver başlatılıyor - Chrome tarayıcı
print("Tarayıcı başlatılıyor...")
driver = webdriver.Chrome()

# İlanları tutmak için liste
ilan_listesi = []

# İlanların linklerini tutmak için liste. Eşya ve doğalgaz durumlarına buradan ulaşacağız
ilan_linkleri = []

sayfa_sayisi = 2 # Toplam sayfa sayısı. Her sayfada 24 ilan var. 21 sayfa için 504 ilan var.

for page in range(1, sayfa_sayisi + 1):
    # Sayfa URL'si oluşturuluyor
    url = f'https://www.hepsiemlak.com/buca-kiralik?page={page}'

    # Sayfa yükleniyor
    print(f"Sayfa {page} yükleniyor: {url}")
    driver.get(url)

    # Sayfa yüklenene kadar bekleniyor
    WebDriverWait(driver, 10).until(expected_conditions.presence_of_element_located((By.CLASS_NAME, 'listing-item')))
    print("Sayfa yüklendi.")

    # İlanlar bulunuyor
    ilanlar = driver.find_elements(By.CLASS_NAME, 'listing-item')
    print(f"{len(ilanlar)} ilan bulundu.")

    # İlanlar işleniyor
    for ilan in ilanlar:
        # Başlık çıkarılıyor
        baslik = ilan.find_element(By.CLASS_NAME, 'card-link').get_attribute('title')

        # Tarih çıkarılıyor
        tarih = ilan.find_element(By.CLASS_NAME, 'list-view-date').text

        # Fiyat çıkarılıyor
        fiyat = ilan.find_element(By.CLASS_NAME, 'list-view-price').text

        # Özellikler çıkarılıyor
        özellikler = ilan.find_element(By.CLASS_NAME, 'short-property').text.split("\n")

        # Özellikler listesi kontrol edilerek gerekli alanlar çıkarılıyor
        # Eğer karşılık gelen index boş ise, ilgili alanlar boş olarak atanıyor
        # Sıfırıncı indexte Kiralık Daire yazısı yer almakta, o yüzden birinci indexten başlıyoruz
        oda_sayisi = özellikler[1] if len(özellikler) > 1 else ''
        alan = özellikler[2] if len(özellikler) > 2 else ''
        bina_yasi = özellikler[3] if len(özellikler) > 3 else ''
        kat_sayisi = özellikler[4] if len(özellikler) > 4 else ''

        # Konum çıkarılıyor
        konum = ilan.find_element(By.CLASS_NAME, 'list-view-location').text

        # İlan linki alınıyor ve ilan_linkleri listesine ekleniyor
        ilan_linki = ilan.find_element(By.CLASS_NAME, 'card-link').get_attribute('href')
        ilan_linkleri.append(ilan_linki)

        # İlan verileri listeye ekleniyor
        ilan_listesi.append({
            'Başlık': baslik,
            'Konum': konum,
            'Fiyat': fiyat,
            'Oda Sayısı': oda_sayisi,
            'Alan': alan,
            'Bina Yaşı': bina_yasi,
            'Kat': kat_sayisi,
            'Tarihi': tarih,
            'Link': ilan_linki,
        })

        print(f"İlan eklendi: {baslik}")
        print(f"İlan linki eklendi: {ilan_linki}")

for ilan_linki in ilan_linkleri:
    # İlan linkine gidiliyor
    print("İlan linki yükleniyor: ", ilan_linki)
    driver.get(ilan_linki)

    # Sayfa yüklenene kadar bekleniyor
    WebDriverWait(driver, 10).until(expected_conditions.presence_of_element_located((By.CLASS_NAME, 'txt')))
    print("İlan linki yüklendi.")

    # Tüm spanlar alınıyor
    spans = driver.find_elements(By.CLASS_NAME, 'txt')

    esya_durumu = ''
    isinma_tipi = ''
    yakit_tipi = ''

    for span in spans:
        if span.text == 'Eşya Durumu':  # Eğer span 'Eşya Durumu' metnini içeriyorsa
            # Kardeş öğesi olan diğer span'dan Eşya Durumu'nu al
            esya_durumu = span.find_element(By.XPATH, "following-sibling::span").text

        elif span.text == 'Isınma Tipi':  # Eğer span 'Isınma Tipi' metnini içeriyorsa
            # Kardeş öğesi olan diğer span'dan Isınma Tipi'ni al
            isinma_tipi = span.find_element(By.XPATH, "following-sibling::span").text

        elif span.text == 'Yakıt Tipi':  # Eğer span 'Yakıt Tipi' metnini içeriyorsa
            # Kardeş öğesi olan diğer span'dan Yakıt Tipi'ni al
            yakit_tipi = span.find_element(By.XPATH, "following-sibling::span").text

        if esya_durumu and yakit_tipi and isinma_tipi != '':  # Eğer tüm bilgiler alındıysa
            break # Döngüden çık

    # İlan listesine 'Eşya Durumu', 'Yakıt Tipi' ve 'Isınma Tipi' ekleniyor
    ilan_index = ilan_linkleri.index(ilan_linki)
    ilan_listesi[ilan_index]['Eşya Durumu'] = esya_durumu
    ilan_listesi[ilan_index]['Isınma Tipi'] = isinma_tipi
    ilan_listesi[ilan_index]['Yakıt Tipi'] = yakit_tipi

    print(f"İlan verileri eklendi: {esya_durumu}, {isinma_tipi}, {yakit_tipi}")


print("Tarayıcı kapatılıyor...")
driver.quit()

# İlan verileri DataFrame'e dönüştürülüyor ve Excel dosyasına kaydediliyor
df = pd.DataFrame(ilan_listesi)
df.to_excel('ilanlar.xlsx', index=False)  # Excel dosyası kaydediliyor
print("İlanlar başarıyla kaydedildi: ilanlar.xlsx")  # İşlem başarı mesajı
